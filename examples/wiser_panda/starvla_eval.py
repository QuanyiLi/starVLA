"""
Evaluate a trained starVLA (Qwen-GR00T) model across 24 config subsets on train/test splits.

Each GPU process evaluates a slice of subsets [start_subset, end_subset).
Inside each episode the model produces a 20-step action chunk; actions are
executed one-by-one until the chunk is exhausted, then a fresh inference is
performed with the latest observation.

Usage (single GPU, subsets 0-5, both splits):

    CUDA_VISIBLE_DEVICES=0 python starvla_eval.py \
        --checkpoint /path/to/steps_XXXXX_pytorch_model.pt \
        --dataset_root /path/to/wiser_dataset \
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
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

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

# ---------------------------------------------------------------------------
# Observation keys (must match the ManiSkill env wrapper)
# ---------------------------------------------------------------------------
IMAGE_1_KEY = "observation.images.image_1"
WRIST_IMAGE_KEY = "observation.images.wrist_image"
IMAGE_1_ROBOT_STATE = "observation.images.image_1_robot_state"
IMAGE_1_SEG_MASK = "observation.images.image_1_segmentation_mask"
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

def env_obs_to_model_inputs(obs, num_envs):
    """
    Convert a batched TensorDict observation from ManiSkill into a list of
    dicts that ``model.predict_action`` expects.

    Each element of the returned list corresponds to one parallel env and has:
        image : list[PIL.Image]   – single-view (image_1)
        lang  : str               – task instruction
        state : np.ndarray (1, state_dim) – current joint state
    """
    # --- images: (B, H, W, C) uint8 tensor → list[PIL.Image] ---
    images_t = obs[IMAGE_1_KEY]  # (B, H, W, C) uint8 on GPU
    images_np = images_t.cpu().numpy()

    # --- task instruction (byte-encoded tensor) → list[str] ---
    instructions = batch_tensor_to_string(obs[TASK_KEY])

    # --- state: (B, state_dim) → np.ndarray ---
    state_np = obs[OBS_STATE_KEY].cpu().numpy()  # (B, state_dim)

    examples = []
    for i in range(num_envs):
        pil_img = Image.fromarray(images_np[i])
        examples.append({
            "image": [pil_img],  # list of views; single view for wiser_panda
            "lang": instructions[i],
            "state": state_np[i:i+1],  # (1, state_dim)
        })
    return examples


def unnormalize_wiser_actions(normalized_actions, action_norm_stats):
    """Unnormalize using min_max mode (matching WiserPandaDataConfig).

    Training normalizes via: 2 * (x - min) / (max - min) - 1
    Inverse:                 (x + 1) / 2 * (max - min) + min

    No binary gripper binarization — wiser_panda normalizes all 8 dims
    (7 joints + 1 gripper) uniformly with min_max.
    """
    action_high = np.array(action_norm_stats["max"])
    action_low = np.array(action_norm_stats["min"])
    normalized_actions = np.clip(normalized_actions, -1, 1)
    return (normalized_actions + 1) / 2 * (action_high - action_low) + action_low


def rollout_with_chunking(
    envs,
    model,
    action_norm_stats,
    max_steps,
    num_envs,
    action_chunk_size=ACTION_CHUNK_SIZE,
):
    """
    Run one round of evaluation across ``num_envs`` parallel environments.

    Inference is performed once to get a 20-step action chunk. Actions are
    then executed **one at a time**. After all 20 are consumed, a new
    observation is taken and another inference is performed. This continues
    until ``max_steps`` environment steps have been taken.

    Returns:
        eval_infos : dict  – the info dict from the final env step (contains
                             episode metrics such as success, return, etc.)
    """
    obs, _ = envs.reset()

    # Pre-allocate per-env action queues (list of remaining actions)
    # Each element is an np.ndarray of shape (remaining, action_dim) or None
    action_queues = [None] * num_envs

    for step_idx in range(max_steps):
        # Check which envs need a fresh inference (queue empty or None)
        needs_inference = [
            i for i in range(num_envs)
            if action_queues[i] is None or len(action_queues[i]) == 0
        ]

        if needs_inference:
            # Build model inputs only for envs that need new actions
            # For simplicity we re-infer for ALL envs when any need it.
            # This is fine because the model is batched and the extra compute
            # is negligible compared to re-constructing partial batches.
            with torch.no_grad():
                examples = env_obs_to_model_inputs(obs, num_envs)
                output = model.predict_action(examples=examples)
                norm_actions = output["normalized_actions"]  # (B, chunk, action_dim)

            # Unnormalize per env and fill queues
            for i in range(num_envs):
                unnorm = unnormalize_wiser_actions(
                    norm_actions[i].copy(), action_norm_stats
                )  # (chunk, action_dim)
                action_queues[i] = unnorm  # full chunk

        # Pop the first action from each env queue
        actions = []
        for i in range(num_envs):
            action = action_queues[i][0]
            action_queues[i] = action_queues[i][1:]  # pop front
            actions.append(action)

        actions_tensor = torch.tensor(
            np.stack(actions, axis=0),
            dtype=torch.float32,
            device=obs[OBS_STATE_KEY].device,
        )

        obs, rew, done, eval_infos = envs.step(actions_tensor)

    return eval_infos, obs


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

    # Retrieve action normalization statistics
    # Load norm_stats directly from the checkpoint (same pattern as LIBERO/SimplerEnv evals)
    # because get_action_stats is a @classmethod and cannot access instance attributes.
    from starVLA.model.framework.share_tools import read_mode_config
    _, norm_stats = read_mode_config(Path(args.checkpoint))
    # Auto-resolve the dataset key when there's only one
    unnorm_key = next(iter(norm_stats.keys()))
    action_norm_stats = norm_stats[unnorm_key]["action"]

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
            envs = build_endless_env(env_cfg, record_video=False, data_record_dir="test")
            max_steps = envs.unwrapped.max_episode_steps

            print("\n" + "=" * 60)
            print(f"  Rollout: {cfg_name} ({split})  |  "
                  f"{num_envs} envs × {args.eval_rounds} rounds × {max_steps} steps")
            print("=" * 60)

            # Collect metrics over multiple rounds
            all_eval_infos = defaultdict(list)
            episode_records = []

            t0 = time.perf_counter()
            for round_idx in range(args.eval_rounds):
                eval_infos, last_obs = rollout_with_chunking(
                    envs,
                    model,
                    action_norm_stats,
                    max_steps=max_steps,
                    num_envs=num_envs,
                    action_chunk_size=ACTION_CHUNK_SIZE,
                )

                # Collect per-env episode records
                task_names = batch_tensor_to_string(last_obs[TASK_KEY])

                for env_idx in range(num_envs):
                    ep_info = eval_infos["episode"]
                    record = {
                        "round": round_idx,
                        "env_idx": env_idx,
                        "global_episode_idx": round_idx * num_envs + env_idx,
                        "task_prompt": task_names[env_idx],
                        "success_once": bool(ep_info["success_once"][env_idx].item()),
                        "success_at_end": bool(ep_info["success_at_end"][env_idx].item()),
                        "return": float(ep_info["return"][env_idx].item()),
                    }
                    # Optional keys
                    for extra_key in ["near_goal", "is_grasped", "tcp_near_goal"]:
                        if extra_key in eval_infos:
                            record[extra_key] = bool(eval_infos[extra_key][env_idx].item())
                    episode_records.append(record)

                # Accumulate summary metrics
                for k in ["success_once", "success_at_end", "return"]:
                    all_eval_infos[k].append(eval_infos["episode"][k])
                for k in ["near_goal", "is_grasped", "tcp_near_goal"]:
                    if k in eval_infos:
                        all_eval_infos[k].append(eval_infos[k])

            elapsed = time.perf_counter() - t0

            # Compute summary statistics
            summary = {}
            for k, vals in all_eval_infos.items():
                stacked = torch.cat(vals).float()
                summary[f"{k}_mean"] = stacked.mean().cpu().item()

            summary["rollout_time"] = round(elapsed, 2)
            total_steps = num_envs * args.eval_rounds * max_steps
            summary["rollout_fps"] = round(total_steps / elapsed, 2)

            print(f"\n  Performance ({elapsed:.1f}s):")
            for k, v in summary.items():
                print(f"    {k}: {v}")

            # Save metrics
            metrics_data = {
                "summary": summary,
                "episodes": episode_records,
                "config": {
                    "num_envs": num_envs,
                    "rounds": args.eval_rounds,
                    "max_steps_per_episode": max_steps,
                    "total_episodes": len(episode_records),
                    "action_chunk_size": ACTION_CHUNK_SIZE,
                },
            }
            with open(metrics_file, "w") as f:
                json.dump(metrics_data, f, indent=2)
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
    p.add_argument("--dataset_root", type=str, default=None,
                    help="Root of the wiser dataset (unused at runtime, "
                         "kept for compatibility)")
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
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval(args)
