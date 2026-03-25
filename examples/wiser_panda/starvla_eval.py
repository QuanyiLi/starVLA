"""
Evaluate a trained starVLA (Qwen-GR00T) model across 24 config subsets on train/test splits.

Uses the official `rollout()` function from vla_align for env management,
metrics collection, and optional video/data saving.

Usage (single GPU, subsets 0-5, both splits):

    CUDA_VISIBLE_DEVICES=0 python starvla_eval.py \
        --checkpoint /path/to/steps_XXXXX_pytorch_model.pt \
        --result_dir /path/to/starvla_eval_result \
        --start_subset 0 --end_subset 6 --split both \
        --eval_rounds 1
"""

import argparse
import glob
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Resolve starVLA root so that local imports work regardless of cwd
# ---------------------------------------------------------------------------
_STARVLA_ROOT = Path(__file__).resolve().parents[2]  # .../starVLA
if str(_STARVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(_STARVLA_ROOT))

# starVLA model loading
from starVLA.model.framework.base_framework import baseframework

# Environment utilities (shared with the vla repo)
_VLA_ROOT = Path(__file__).resolve().parents[3] / "vla"  # .../vla
if str(_VLA_ROOT) not in sys.path:
    sys.path.insert(0, str(_VLA_ROOT))

from vla_align.env.config import get_env_cfg, MAX_EPISODE_STEP_WORKSPACE_EVAL
from vla_align.utils.env import build_endless_env
from vla_align.utils.helpers import batch_tensor_to_string
from vla_align.utils.rollout import rollout

# ---------------------------------------------------------------------------
# Observation keys (must match the ManiSkill env wrapper)
# ---------------------------------------------------------------------------
IMAGE_1_KEY = "observation.images.image_1"
WRIST_IMAGE_KEY = "observation.images.wrist_image"
OBS_STATE_KEY = "observation.state"
TASK_KEY = "task"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ACTION_CHUNK_SIZE = 20  # action_horizon from starvla_cotrain_wiser.yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ======================================================================== #
#                          Helper Functions                                  #
# ======================================================================== #

def unnormalize_actions_min_max(normalized_actions, action_norm_stats):
    """Unnormalize using min_max mode (matching WiserPandaDataConfig).

    Training normalizes via: 2 * (x - min) / (max - min) - 1
    Inverse:                 (x + 1) / 2 * (max - min) + min

    WiserPandaDataConfig uses min_max for ALL 8 dims (7 joints + 1 gripper)
    uniformly — no binary gripper binarization, no q99 mode.
    """
    action_high = np.array(action_norm_stats["max"])
    action_low = np.array(action_norm_stats["min"])
    normalized_actions = np.clip(normalized_actions, -1, 1)
    return (normalized_actions + 1) / 2 * (action_high - action_low) + action_low


def normalize_state_min_max(raw_state, state_norm_stats):
    """Normalize raw state using min_max mode (matching WiserPandaDataConfig).

    Formula: 2 * (x - min) / (max - min) - 1  →  maps to [-1, 1]

    Args:
        raw_state: np.ndarray of shape (state_dim,) — raw env state.
        state_norm_stats: dict with 'min' and 'max' lists.
    Returns:
        np.ndarray of same shape, normalized to [-1, 1].
    """
    s_min = np.array(state_norm_stats["min"])
    s_max = np.array(state_norm_stats["max"])
    denom = s_max - s_min
    # Avoid division by zero where min == max (set to 0 as training does)
    mask = denom != 0
    normalized = np.zeros_like(raw_state, dtype=np.float32)
    normalized[..., mask] = 2.0 * (raw_state[..., mask] - s_min[mask]) / denom[mask] - 1.0
    return np.clip(normalized, -1, 1)


# ======================================================================== #
#                     StarVLA Policy Wrapper                                 #
# ======================================================================== #

class StarVLAChunkedPolicy:
    """
    Wraps a starVLA model into the policy interface expected by
    ``vla_align.utils.rollout.rollout``:

        action_to_take, expert_action, policy_info = policy(obs)

    Internally maintains per-env action queues. When a queue is empty,
    a fresh batch inference is performed to get a 20-step action chunk.
    One action is popped from the queue per call.

    Inputs to the model:
    - 2 images (main camera + wrist camera), resized to 224×224 inside predict_action().
    - State (9-dim q_pos), normalized with min_max before passing to model.
    - CoT prompt wrapping is handled automatically inside build_qwenvl_inputs().
    """

    def __init__(self, model, action_norm_stats, state_norm_stats,
                 action_chunk_size=ACTION_CHUNK_SIZE):
        self.model = model
        self.action_norm_stats = action_norm_stats
        self.state_norm_stats = state_norm_stats
        self.action_chunk_size = action_chunk_size
        self._action_queues = None  # lazily initialized
        self._first_inference_done = False

    def reset(self):
        """Clear action queues (call before each new episode/round)."""
        self._action_queues = None
        self._first_inference_done = False

    def __call__(self, obs):
        """
        Args:
            obs: TensorDict from ManiSkill env with batched observations.

        Returns:
            (action_to_take, expert_action, policy_info)
            action_to_take: (num_envs, action_dim) tensor to step the env
            expert_action:  same as action_to_take (no separate expert here)
            policy_info:    dict with timing / diagnostic info
        """
        num_envs = obs[IMAGE_1_KEY].shape[0]
        device = obs[OBS_STATE_KEY].device

        # Lazy init queues
        if self._action_queues is None:
            self._action_queues = [None] * num_envs

        # Check if any env needs a fresh inference
        needs_inference = any(
            self._action_queues[i] is None or len(self._action_queues[i]) == 0
            for i in range(num_envs)
        )

        policy_info = {"inference_time": 0.0}
        if needs_inference:
            t0 = time.perf_counter()

            # Convert env obs → model input format
            images_np = obs[IMAGE_1_KEY].cpu().numpy()  # (B, H, W, C) uint8
            wrist_np = obs[WRIST_IMAGE_KEY].cpu().numpy()  # (B, H, W, C) uint8
            state_np = obs[OBS_STATE_KEY].cpu().numpy()    # (B, state_dim) float
            instructions = batch_tensor_to_string(obs[TASK_KEY])

            examples = []
            for i in range(num_envs):
                # Normalize state with min_max (same as training)
                norm_state = normalize_state_min_max(
                    state_np[i], self.state_norm_stats
                )
                examples.append({
                    "image": [
                        Image.fromarray(images_np[i]),   # main camera
                        Image.fromarray(wrist_np[i]),    # wrist camera
                    ],
                    "lang": instructions[i],
                    "state": norm_state[np.newaxis, :].astype(np.float16),  # (1, state_dim)
                })

            with torch.no_grad():
                output = self.model.predict_action(examples=examples)
                norm_actions = output["normalized_actions"]  # (B, chunk, action_dim)

            # Diagnostic on first call
            if not self._first_inference_done:
                logger.info(f"[DIAG] norm_actions shape={norm_actions.shape}, "
                            f"range=[{norm_actions.min():.4f}, {norm_actions.max():.4f}]")
                for env_i, inst in enumerate(instructions):
                    logger.info(f"[DIAG] env {env_i} lang: '{inst}'")

            # Unnormalize and fill queues
            for i in range(num_envs):
                unnorm = unnormalize_actions_min_max(
                    norm_actions[i].copy(), self.action_norm_stats
                )
                self._action_queues[i] = unnorm

            if not self._first_inference_done:
                u0 = self._action_queues[0]
                logger.info(f"[DIAG] unnorm_actions[0] shape={u0.shape}, "
                            f"range=[{u0.min():.4f}, {u0.max():.4f}]")
                self._first_inference_done = True

            policy_info["inference_time"] = time.perf_counter() - t0

        # Pop one action per env
        actions = []
        for i in range(num_envs):
            actions.append(self._action_queues[i][0])
            self._action_queues[i] = self._action_queues[i][1:]

        actions_tensor = torch.tensor(
            np.stack(actions, axis=0), dtype=torch.float32, device=device
        )

        return actions_tensor, actions_tensor, policy_info


# ======================================================================== #
#                           Results Aggregation                              #
# ======================================================================== #

def aggregate_results(result_dir):
    """Aggregate per-subset eval results into final_results_{split}.json."""
    for split in ["train", "test"]:
        pattern = os.path.join(result_dir, f"*{split}*")
        directories = sorted(glob.glob(pattern))
        if not directories:
            logger.info(f"No results found for split '{split}'")
            continue

        metrics_sums, metrics_counts = {}, {}
        metrics_mins, metrics_maxs = {}, {}
        file_count = 0

        for d in directories:
            mf = os.path.join(d, "episode_metrics.json")
            if not os.path.exists(mf):
                continue
            try:
                with open(mf) as f:
                    summary = json.load(f).get("summary", {})
                for k, v in summary.items():
                    if isinstance(v, (int, float)):
                        metrics_sums[k] = metrics_sums.get(k, 0) + v
                        metrics_counts[k] = metrics_counts.get(k, 0) + 1
                        if k not in metrics_mins or v < metrics_mins[k]:
                            metrics_mins[k] = v
                        if k not in metrics_maxs or v > metrics_maxs[k]:
                            metrics_maxs[k] = v
                file_count += 1
            except json.JSONDecodeError:
                logger.warning(f"Bad JSON in {mf}, skipping")

        logger.info(f"[{split}] Aggregated {file_count} files")
        final = {}
        for k in sorted(metrics_sums):
            avg = metrics_sums[k] / metrics_counts[k]
            final[k] = {"avg": avg, "min": metrics_mins[k], "max": metrics_maxs[k]}
            logger.info(f"  {k}: {avg:.4f}  (min={metrics_mins[k]:.4f}, max={metrics_maxs[k]:.4f})")

        out = os.path.join(result_dir, f"final_results_{split}.json")
        with open(out, "w") as f:
            json.dump(final, f, indent=2)
        logger.info(f"  → saved to {out}")


# ======================================================================== #
#                           Main Eval Driver                                #
# ======================================================================== #

def run_eval(args):
    if args.aggregate_only:
        aggregate_results(args.result_dir)
        return

    os.makedirs(args.result_dir, exist_ok=True)

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # ------------------------------------------------------------------ #
    # Load model
    # ------------------------------------------------------------------ #
    logger.info(f"Loading starVLA model from {args.checkpoint}")
    model = baseframework.from_pretrained(args.checkpoint)
    model = model.to(device).eval()

    # ------------------------------------------------------------------ #
    # Retrieve action normalization statistics
    # ------------------------------------------------------------------ #
    norm_stats = model.norm_stats
    unnorm_key = next(iter(norm_stats.keys()))
    action_norm_stats = norm_stats[unnorm_key]["action"]
    state_norm_stats = norm_stats[unnorm_key]["state"]

    logger.info(f"unnorm_key = {unnorm_key}")
    logger.info(f"action min = {action_norm_stats.get('min', 'MISSING')}")
    logger.info(f"action max = {action_norm_stats.get('max', 'MISSING')}")
    logger.info(f"state min  = {state_norm_stats.get('min', 'MISSING')}")
    logger.info(f"state max  = {state_norm_stats.get('max', 'MISSING')}")

    assert "min" in action_norm_stats and "max" in action_norm_stats, (
        f"action_norm_stats must contain 'min' and 'max' keys for min_max "
        f"unnormalization, but got keys: {list(action_norm_stats.keys())}"
    )
    assert "min" in state_norm_stats and "max" in state_norm_stats, (
        f"state_norm_stats must contain 'min' and 'max' keys for min_max "
        f"normalization, but got keys: {list(state_norm_stats.keys())}"
    )

    # ------------------------------------------------------------------ #
    # Build policy wrapper
    # ------------------------------------------------------------------ #
    policy = StarVLAChunkedPolicy(model, action_norm_stats, state_norm_stats)

    # ------------------------------------------------------------------ #
    # Determine splits
    # ------------------------------------------------------------------ #
    splits = ["train", "test"] if args.split == "both" else [args.split]
    logger.info(
        f"Evaluating subsets {args.start_subset}–{args.end_subset - 1} "
        f"on splits: {splits}"
    )

    # ------------------------------------------------------------------ #
    # Evaluation loop
    # ------------------------------------------------------------------ #
    num_envs = args.num_envs
    save_video = args.save_video

    for split in splits:
        for cfg_idx in range(args.start_subset, args.end_subset):
            cfg_name = f"config_{cfg_idx}"
            subset_dir = os.path.join(args.result_dir, f"config_{cfg_idx}_{split}")

            # Skip if already evaluated
            metrics_file = os.path.join(subset_dir, "episode_metrics.json")
            if os.path.exists(metrics_file):
                logger.info(f"Skipping {cfg_name} ({split}) — already evaluated")
                continue

            if os.path.exists(subset_dir):
                shutil.rmtree(subset_dir)
            os.makedirs(subset_dir)

            # Build environment
            scene_cfg = dict(
                robot_init_qpos_noise=0.0,
                cube_size_noise=0.0,
                cfg_name=cfg_name,
                mode=split,
            )
            env_cfg = get_env_cfg(
                num_env=num_envs,
                max_steps=MAX_EPISODE_STEP_WORKSPACE_EVAL,
                obs_mode="rgb+segmentation",
                scene_cfg_to_overwrite=scene_cfg,
            )
            envs = build_endless_env(
                env_cfg,
                record_video=save_video,
                data_record_dir=subset_dir if save_video else None,
            )

            print("\n" + "=" * 60)
            print(f"  Rollout: {cfg_name} ({split})  |  "
                  f"{num_envs} envs × {args.eval_rounds} rounds")
            print("=" * 60)

            # Reset policy queues before each subset
            policy.reset()

            # Use the official rollout function
            # indices_to_save=None → save all envs when demo_saving_dir is set
            results = rollout(
                envs,
                policy,
                round_to_collect=args.eval_rounds,
                demo_saving_dir=subset_dir if save_video else None,
                indices_to_save=None,
            )

            print(f"\n  Performance:")
            for k, v in results.items():
                print(f"    {k}: {v}")

            # Save metrics (rollout already saves episode_metrics.json
            # in demo_saving_dir, but also save to subset_dir when
            # not saving video)
            if not save_video or not os.path.exists(metrics_file):
                with open(metrics_file, "w") as f:
                    json.dump({"summary": results}, f, indent=2)
                logger.info(f"Metrics saved to {metrics_file}")

            envs.unwrapped.close()

    logger.info("All subsets evaluated. Done.")


# ======================================================================== #
#                                CLI                                        #
# ======================================================================== #

def parse_args():
    p = argparse.ArgumentParser(description="StarVLA distributed evaluation")
    p.add_argument("--checkpoint", type=str, required=True,
                    help="Path to starVLA .pt or .safetensors checkpoint")
    p.add_argument("--result_dir", type=str, default="./starvla_eval_result",
                    help="Directory to store per-subset results")
    p.add_argument("--start_subset", type=int, default=0,
                    help="First config index (inclusive)")
    p.add_argument("--end_subset", type=int, default=24,
                    help="Last config index (exclusive)")
    p.add_argument("--split", type=str, default="both",
                    choices=["train", "test", "both"],
                    help="Which splits to evaluate")
    p.add_argument("--eval_rounds", type=int, default=1,
                    help="Number of rollout rounds per subset")
    p.add_argument("--num_envs", type=int, default=12,
                    help="Number of parallel environments")
    p.add_argument("--aggregate_only", action="store_true",
                    help="Skip rollouts, only aggregate existing results")
    p.add_argument("--save_video", action="store_true",
                    help="Save rollout videos and data to each subset dir")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval(args)
