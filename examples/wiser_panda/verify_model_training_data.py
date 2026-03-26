"""
Verify a trained starVLA model by forwarding one training data sample.

Loads the model identically to starvla_eval.py (via baseframework.from_pretrained),
then loads the training dataset identically to submit_starvla_wiser.run, and performs:
  1. model.forward([sample]) → action_loss  (training loss)
  2. model.predict_action([sample]) → normalized predicted actions

Logs shapes, dtypes, ranges, and saves input images + a diagnostic JSON.

Usage:
    CUDA_VISIBLE_DEVICES=0 python verify_model_training_data.py \
        --checkpoint /path/to/steps_XXXXX_pytorch_model.pt \
        --output_dir /tmp/starvla_debug
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Path setup (same as starvla_eval.py)
# ---------------------------------------------------------------------------
_STARVLA_ROOT = Path(__file__).resolve().parents[2]  # .../starVLA
if str(_STARVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(_STARVLA_ROOT))

_VLA_ROOT = Path(__file__).resolve().parents[3] / "vla"  # .../vla
if str(_VLA_ROOT) not in sys.path:
    sys.path.insert(0, str(_VLA_ROOT))

from omegaconf import OmegaConf

from starVLA.model.framework.base_framework import baseframework
from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def describe_tensor_or_array(name, x):
    """Return a diagnostic dict for a numpy array or torch tensor."""
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().float().numpy()
    elif isinstance(x, np.ndarray):
        x_np = x.astype(np.float64)
    else:
        return {"name": name, "type": str(type(x)), "value": str(x)[:200]}
    return {
        "name": name,
        "shape": list(x_np.shape),
        "dtype": str(x.dtype) if isinstance(x, torch.Tensor) else str(x.dtype),
        "min": float(x_np.min()),
        "max": float(x_np.max()),
        "mean": float(x_np.mean()),
        "std": float(x_np.std()),
    }


def main():
    parser = argparse.ArgumentParser(description="Verify starVLA model with training data")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to starVLA .pt or .safetensors checkpoint")
    parser.add_argument("--config_yaml", type=str,
                        default=str(_STARVLA_ROOT / "examples/wiser_panda/starvla_cotrain_wiser.yaml"),
                        help="Path to the training YAML config")
    parser.add_argument("--output_dir", type=str, default="/tmp/starvla_debug",
                        help="Directory to save images and diagnostic info")
    parser.add_argument("--data_root_dir", type=str, default=None,
                        help="Override dataset root dir (default: from YAML)")
    parser.add_argument("--sample_index", type=int, default=0,
                        help="Index of the training sample to use")
    # --- Env replay options ---
    parser.add_argument("--replay_in_env", action="store_true",
                        help="Replay predicted actions in ManiSkill sim env")
    parser.add_argument("--video_dir", type=str,
                        default="/work/vita/lanfeng/vlas/starvla_verify_debug",
                        help="Directory to save env replay videos")
    parser.add_argument("--num_envs", type=int, default=12,
                        help="Number of parallel environments for replay")
    parser.add_argument("--start_subset", type=int, default=0,
                        help="First config index (inclusive) for env replay")
    parser.add_argument("--end_subset", type=int, default=1,
                        help="Last config index (exclusive) for env replay")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "test"],
                        help="Which split to use for env replay")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diagnostics = {}

    # ==================================================================== #
    # 1. Load model (same as starvla_eval.py)
    # ==================================================================== #
    logger.info(f"Loading starVLA model from {args.checkpoint}")
    model = baseframework.from_pretrained(args.checkpoint)
    model = model.to(device).eval()
    logger.info("Model loaded successfully")

    # Log norm stats
    norm_stats = model.norm_stats
    unnorm_key = next(iter(norm_stats.keys()))
    action_norm_stats = norm_stats[unnorm_key]["action"]
    diagnostics["unnorm_key"] = unnorm_key
    diagnostics["action_norm_stats"] = {
        k: v if isinstance(v, list) else str(v) for k, v in action_norm_stats.items()
    }
    logger.info(f"unnorm_key = {unnorm_key}")
    logger.info(f"action_norm_stats keys = {list(action_norm_stats.keys())}")
    for k, v in action_norm_stats.items():
        # Use print() to avoid rich logging handler crash on long float lists
        print(f"  {k} = {v}")

    # ==================================================================== #
    # 2. Load training dataset (same as submit_starvla_wiser.run)
    # ==================================================================== #
    logger.info(f"Loading training config from {args.config_yaml}")
    cfg = OmegaConf.load(args.config_yaml)

    if args.data_root_dir:
        cfg.datasets.vla_data.data_root_dir = args.data_root_dir

    vla_dataset_cfg = cfg.datasets.vla_data
    logger.info(f"Building dataset with data_mix={vla_dataset_cfg.data_mix}, "
                f"data_root_dir={vla_dataset_cfg.data_root_dir}")

    dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
    logger.info(f"Dataset loaded: {len(dataset)} samples")

    # ==================================================================== #
    # 3. Get one training sample
    # ==================================================================== #
    idx = args.sample_index
    logger.info(f"Getting training sample at index {idx}")
    sample = dataset[idx]

    # ==================================================================== #
    # 4. Inspect and save inputs
    # ==================================================================== #
    logger.info("=" * 60)
    logger.info("INPUT DIAGNOSTICS")
    logger.info("=" * 60)

    input_diags = []

    # -- Action --
    action = sample["action"]
    d = describe_tensor_or_array("action", action)
    input_diags.append(d)
    logger.info(f"action: shape={d['shape']}, dtype={d['dtype']}, "
                f"range=[{d['min']:.6f}, {d['max']:.6f}], mean={d['mean']:.6f}")

    # -- State (if present) --
    if "state" in sample:
        state = sample["state"]
        d = describe_tensor_or_array("state", state)
        input_diags.append(d)
        logger.info(f"state: shape={d['shape']}, dtype={d['dtype']}, "
                    f"range=[{d['min']:.6f}, {d['max']:.6f}], mean={d['mean']:.6f}")
    else:
        logger.info("state: NOT present in sample")

    # -- Images --
    images = sample["image"]
    logger.info(f"images: {len(images)} PIL images")
    for i, img in enumerate(images):
        img_np = np.array(img)
        d = describe_tensor_or_array(f"image_{i}", img_np)
        input_diags.append(d)
        logger.info(f"  image_{i}: size={img.size}, mode={img.mode}, "
                    f"array shape={img_np.shape}, dtype={img_np.dtype}, "
                    f"range=[{img_np.min()}, {img_np.max()}]")

        # Save image
        save_path = os.path.join(args.output_dir, f"input_image_{i}.png")
        img.save(save_path)
        logger.info(f"  Saved to {save_path}")

    # -- Language --
    lang = sample.get("lang", sample.get("language", "N/A"))
    logger.info(f"lang: '{lang}'")
    input_diags.append({"name": "lang", "value": str(lang)})

    diagnostics["input_diagnostics"] = input_diags

    # ==================================================================== #
    # 5. Forward pass (training loss)
    # ==================================================================== #
    logger.info("=" * 60)
    logger.info("FORWARD PASS (Training Loss)")
    logger.info("=" * 60)

    batch = [sample]  # batch of 1

    # Log exact input format going into model.forward()
    logger.info("-" * 40)
    logger.info("Exact examples[0] fed to model.forward():")
    for key, val in sample.items():
        if isinstance(val, np.ndarray):
            logger.info(f"  {key}: np.ndarray  dtype={val.dtype}  shape={val.shape}  "
                        f"range=[{val.min():.6f}, {val.max():.6f}]")
        elif isinstance(val, torch.Tensor):
            logger.info(f"  {key}: torch.Tensor  dtype={val.dtype}  shape={tuple(val.shape)}  "
                        f"range=[{val.min():.6f}, {val.max():.6f}]")
        elif isinstance(val, list):
            logger.info(f"  {key}: list  len={len(val)}")
            for j, item in enumerate(val):
                if isinstance(item, Image.Image):
                    arr = np.array(item)
                    logger.info(f"    [{j}]: PIL.Image  size={item.size}  mode={item.mode}  "
                                f"array_dtype={arr.dtype}  array_shape={arr.shape}")
                elif isinstance(item, np.ndarray):
                    logger.info(f"    [{j}]: np.ndarray  dtype={item.dtype}  shape={item.shape}")
                else:
                    logger.info(f"    [{j}]: {type(item).__name__}  val={str(item)[:100]}")
        elif isinstance(val, str):
            logger.info(f"  {key}: str  val='{val[:100]}'")
        else:
            logger.info(f"  {key}: {type(val).__name__}  val={str(val)[:100]}")
    logger.info("-" * 40)

    try:
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model.forward(batch)
        action_loss = output["action_loss"]
        loss_val = action_loss.item()
        logger.info(f"action_loss = {loss_val}")
        diagnostics["forward_loss"] = loss_val

        if np.isnan(loss_val):
            logger.error("⚠️  Loss is NaN! Something is wrong.")
        elif np.isinf(loss_val):
            logger.error("⚠️  Loss is Inf! Something is wrong.")
        elif loss_val > 100:
            logger.warning(f"⚠️  Loss is unusually high: {loss_val}")
        else:
            logger.info(f"✅  Loss looks reasonable: {loss_val}")
    except Exception as e:
        logger.error(f"Forward pass FAILED: {e}")
        diagnostics["forward_error"] = str(e)
        import traceback
        traceback.print_exc()

    # ==================================================================== #
    # 6. Predict action (inference path)
    # ==================================================================== #
    logger.info("=" * 60)
    logger.info("PREDICT ACTION (Inference Path)")
    logger.info("=" * 60)

    try:
        with torch.no_grad():
            pred_output = model.predict_action(examples=batch)
        norm_actions = pred_output["normalized_actions"]
        d = describe_tensor_or_array("predicted_normalized_actions", norm_actions)
        logger.info(f"predicted normalized actions: shape={d['shape']}, "
                    f"range=[{d['min']:.6f}, {d['max']:.6f}], mean={d['mean']:.6f}, std={d['std']:.6f}")
        diagnostics["predicted_actions"] = d

        # Compare with ground truth
        gt_action = np.array(sample["action"], dtype=np.float64)
        pred_action = norm_actions[0]  # first (only) sample in batch

        # The GT action shape is [chunk, action_dim], pred is also [chunk, action_dim]
        # But the model predicts only the last (future_action_window_size+1) steps
        future_window = model.config.framework.action_model.future_action_window_size
        gt_target = gt_action[-(future_window + 1):, :]

        if pred_action.shape == gt_target.shape:
            mse = float(np.mean((pred_action.astype(np.float64) - gt_target.astype(np.float64))**2))
            logger.info(f"MSE between predicted and GT (normalized): {mse:.6f}")
            diagnostics["pred_vs_gt_mse"] = mse

            per_dim_mse = np.mean((pred_action.astype(np.float64) - gt_target.astype(np.float64))**2, axis=0)
            logger.info(f"Per-dimension MSE: {per_dim_mse}")
            diagnostics["pred_vs_gt_per_dim_mse"] = per_dim_mse.tolist()
        else:
            logger.warning(f"Shape mismatch: pred={pred_action.shape} vs gt_target={gt_target.shape}")
            diagnostics["pred_vs_gt_shape_mismatch"] = {
                "pred": list(pred_action.shape),
                "gt_target": list(gt_target.shape),
            }

        # Save predicted and GT actions
        np.save(os.path.join(args.output_dir, "predicted_actions.npy"), norm_actions)
        np.save(os.path.join(args.output_dir, "gt_actions.npy"), gt_action)
        logger.info("Saved predicted_actions.npy and gt_actions.npy")

    except Exception as e:
        logger.error(f"Predict action FAILED: {e}")
        diagnostics["predict_error"] = str(e)
        import traceback
        traceback.print_exc()

    # ==================================================================== #
    # 7. Replay predicted actions in sim env (open-loop)
    # ==================================================================== #
    if args.replay_in_env:
        logger.info("=" * 60)
        logger.info("REPLAY PREDICTED ACTIONS IN SIM ENV")
        logger.info("=" * 60)

        try:
            from vla_align.env.config import get_env_cfg, MAX_EPISODE_STEP_WORKSPACE_EVAL
            from vla_align.utils.env import build_endless_env

            # Retrieve norm stats for unnormalization
            state_norm_stats = norm_stats[unnorm_key]["state"]

            def unnormalize_actions_min_max(normalized_actions, action_norm_stats_dict):
                """Inverse of 2*(x-min)/(max-min)-1 → (x+1)/2*(max-min)+min"""
                a_high = np.array(action_norm_stats_dict["max"])
                a_low = np.array(action_norm_stats_dict["min"])
                normalized_actions = np.clip(normalized_actions, -1, 1)
                return (normalized_actions + 1) / 2 * (a_high - a_low) + a_low

            # Get predicted normed actions from Section 6 (shape [1, 20, 8])
            # norm_actions was set in Section 6's predict_action call
            assert norm_actions is not None, "predict_action must succeed before env replay"
            single_normed = norm_actions[0]  # (20, 8)
            logger.info(f"Normed actions for replay: shape={single_normed.shape}, "
                        f"range=[{single_normed.min():.4f}, {single_normed.max():.4f}]")

            # Unnormalize
            single_unnormed = unnormalize_actions_min_max(single_normed.copy(), action_norm_stats)
            logger.info(f"Unnormed actions for replay: shape={single_unnormed.shape}, "
                        f"range=[{single_unnormed.min():.4f}, {single_unnormed.max():.4f}]")

            # Save unnormed actions for inspection
            np.save(os.path.join(args.output_dir, "replay_unnormed_actions.npy"), single_unnormed)
            logger.info("Saved replay_unnormed_actions.npy")

            num_action_steps = single_unnormed.shape[0]  # 20
            num_envs = args.num_envs

            for cfg_idx in range(args.start_subset, args.end_subset):
                cfg_name = f"config_{cfg_idx}"
                subset_dir = os.path.join(args.video_dir, f"{cfg_name}_{args.split}_replay")
                os.makedirs(subset_dir, exist_ok=True)

                logger.info(f"Building env: {cfg_name} ({args.split}), {num_envs} envs")
                scene_cfg = dict(
                    robot_init_qpos_noise=0.0,
                    cube_size_noise=0.0,
                    cfg_name=cfg_name,
                    mode=args.split,
                )
                env_cfg = get_env_cfg(
                    num_env=num_envs,
                    max_steps=num_action_steps,  # only run for action chunk length
                    obs_mode="rgb+segmentation",
                    scene_cfg_to_overwrite=scene_cfg,
                )
                envs = build_endless_env(
                    env_cfg,
                    record_video=True,
                    data_record_dir=subset_dir,
                )

                obs, _ = envs.reset()
                device = obs["observation.state"].device

                logger.info(f"Stepping {num_action_steps} actions in {num_envs} envs (open-loop) ...")
                for t in range(num_action_steps):
                    # Broadcast the same action to all envs: (8,) → (num_envs, 8)
                    action_t = np.tile(single_unnormed[t], (num_envs, 1))  # (num_envs, 8)
                    action_tensor = torch.tensor(action_t, dtype=torch.float32, device=device)
                    obs, rew, done, info = envs.step(action_tensor)
                    if t == 0 or t == num_action_steps - 1:
                        logger.info(f"  step {t}: action={single_unnormed[t][:4]}... "
                                    f"rew_mean={rew.float().mean():.4f}")

                # Log final info
                if "episode" in info:
                    for k, v in info["episode"].items():
                        val = v.float().mean().item() if hasattr(v, 'float') else v
                        logger.info(f"  final {k}: {val}")
                        diagnostics[f"replay_{cfg_name}_{k}"] = float(val) if isinstance(val, (int, float)) else str(val)

                envs.unwrapped.close()
                logger.info(f"Replay videos saved to {subset_dir}")

            diagnostics["replay_completed"] = True

        except Exception as e:
            logger.error(f"Env replay FAILED: {e}")
            diagnostics["replay_error"] = str(e)
            import traceback
            traceback.print_exc()

    # ==================================================================== #
    # 8. Save diagnostics
    # ==================================================================== #
    diag_path = os.path.join(args.output_dir, "diagnostics.json")
    # Convert any numpy types to Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2, default=convert)
    logger.info(f"Diagnostics saved to {diag_path}")

    logger.info("=" * 60)
    logger.info("VERIFICATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
