"""
Verify checkpoint integrity by reproducing the EXACT training forward pass.

Loads the model exactly like train_starvla.py does during resume:
  1. OmegaConf.load(original training YAML) → cfg
  2. build_framework(cfg)  → fresh model
  3. load_pretrained_backbones(model, checkpoint, reload_modules=None) → load weights (strict=False, same as training)
  4. model.forward(batch) with torch.autocast("cuda", dtype=torch.bfloat16)

This isolates whether the checkpoint produces the correct loss scale when
loaded the training way.  Compare the loss here with your wandb/training logs.

Usage:
    CUDA_VISIBLE_DEVICES=0 python verify_ckpt_training_loss.py \
        --checkpoint /path/to/steps_XXXXX_pytorch_model.pt \
        --config_yaml examples/wiser_panda/starvla_cotrain_wiser.yaml \
        --output_dir /tmp/ckpt_verify
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_STARVLA_ROOT = Path(__file__).resolve().parents[2]  # .../starVLA
if str(_STARVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(_STARVLA_ROOT))

from omegaconf import OmegaConf

from starVLA.model.framework import build_framework
from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Replicate load_pretrained_backbones from TrainerUtils (non-distributed)
# ---------------------------------------------------------------------------
def load_checkpoint_training_style(model, checkpoint_path, reload_modules=None):
    """
    Exact replica of TrainerUtils.load_pretrained_backbones(), but without
    dist.get_rank() calls so it works in a single-GPU non-distributed setting.
    
    When reload_modules is None (default, same as training resume),
    it does model.load_state_dict(checkpoint, strict=False).
    """
    logger.info(f"📦 loading checkpoint (training-style): {checkpoint_path}")
    
    checkpoint_path = str(checkpoint_path)
    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        checkpoint = load_file(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if reload_modules:  # partial load
        module_paths = [p.strip() for p in reload_modules.split(",") if p.strip()]
        for path in module_paths:
            parts = path.split(".")
            module = model
            try:
                for attr in parts:
                    module = getattr(module, attr)
                prefix = path + "."
                sub_state_dict = {k[len(prefix):]: v for k, v in checkpoint.items() if k.startswith(prefix)}
                if sub_state_dict:
                    module.load_state_dict(sub_state_dict, strict=True)
                    logger.info(f"✅ parameters loaded to module '{path}'")
                else:
                    logger.warning(f"⚠️ parameters not found in checkpoint '{path}'")
            except AttributeError:
                logger.error(f"❌ cannot find module path: {path}")
    else:  # full load — THIS IS WHAT TRAINING RESUME DOES
        # NOTE: strict=False, matching train_starvla.py TrainerUtils exactly
        result = model.load_state_dict(checkpoint, strict=False)
        missing = result.missing_keys
        unexpected = result.unexpected_keys
        if missing:
            logger.warning(f"Missing keys ({len(missing)}): {missing[:10]}{'...' if len(missing) > 10 else ''}")
        if unexpected:
            logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")
        if not missing and not unexpected:
            logger.info("✅ loaded <full_model> — all keys matched perfectly")
        else:
            logger.info(f"✅ loaded <full_model> with strict=False (missing={len(missing)}, unexpected={len(unexpected)})")

    return model


def describe(name, x):
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
    parser = argparse.ArgumentParser(
        description="Verify checkpoint by reproducing training forward pass"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to starVLA .pt or .safetensors checkpoint",
    )
    parser.add_argument(
        "--config_yaml", type=str,
        default=str(_STARVLA_ROOT / "examples/wiser_panda/starvla_cotrain_wiser.yaml"),
        help="Path to the ORIGINAL training YAML config",
    )
    parser.add_argument(
        "--output_dir", type=str, default="/tmp/ckpt_verify",
        help="Directory to save diagnostic JSON",
    )
    parser.add_argument(
        "--data_root_dir", type=str, default=None,
        help="Override dataset root dir (default: from YAML)",
    )
    parser.add_argument(
        "--num_samples", type=int, default=5,
        help="Number of training samples to forward",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size for DataLoader",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diagnostics = {}

    # ==================================================================== #
    # 1. Load ORIGINAL training config (same YAML used by submit_starvla_wiser.run)
    # ==================================================================== #
    logger.info("=" * 60)
    logger.info("STEP 1: Loading original training config")
    logger.info("=" * 60)
    logger.info(f"Config YAML: {args.config_yaml}")

    cfg = OmegaConf.load(args.config_yaml)
    if args.data_root_dir:
        cfg.datasets.vla_data.data_root_dir = args.data_root_dir

    # Set output_dir (needed by some internal paths)
    cfg.output_dir = args.output_dir

    diagnostics["config_yaml"] = str(args.config_yaml)
    diagnostics["checkpoint"] = str(args.checkpoint)
    diagnostics["framework_name"] = str(cfg.framework.name)
    logger.info(f"Framework: {cfg.framework.name}")
    logger.info(f"Action dim: {cfg.framework.action_model.action_dim}")
    logger.info(f"State dim: {cfg.framework.action_model.state_dim}")
    logger.info(f"future_action_window_size: {cfg.framework.action_model.future_action_window_size}")

    # ==================================================================== #
    # 2. Build model from config (SAME as train_starvla.py main())
    #    vla = build_framework(cfg)
    # ==================================================================== #
    logger.info("=" * 60)
    logger.info("STEP 2: build_framework(cfg) — same as training")
    logger.info("=" * 60)

    model = build_framework(cfg)
    logger.info("Model built successfully from original training config")

    # ==================================================================== #
    # 3. Load checkpoint (SAME as _init_checkpointing in training)
    #    model = self.load_pretrained_backbones(model, checkpoint, reload_modules=None)
    #    → model.load_state_dict(checkpoint, strict=False)
    # ==================================================================== #
    logger.info("=" * 60)
    logger.info("STEP 3: load_pretrained_backbones(model, ckpt) — same as training resume")
    logger.info("=" * 60)

    model = load_checkpoint_training_style(model, args.checkpoint, reload_modules=None)
    model = model.to(device)
    logger.info("Checkpoint loaded and model moved to device")

    # ==================================================================== #
    # 4. Load training dataset (SAME as train_starvla.py)
    # ==================================================================== #
    logger.info("=" * 60)
    logger.info("STEP 4: Loading training dataset")
    logger.info("=" * 60)

    vla_dataset_cfg = cfg.datasets.vla_data
    logger.info(f"data_mix = {vla_dataset_cfg.data_mix}")
    logger.info(f"data_root_dir = {vla_dataset_cfg.data_root_dir}")

    dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
    logger.info(f"Dataset loaded: {len(dataset)} samples")
    diagnostics["dataset_size"] = len(dataset)

    # Build DataLoader exactly like training (collate_fn just returns the batch list)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=0,
        shuffle=False,
    )

    # ==================================================================== #
    # 5. Forward pass — EXACT training path
    #    with torch.autocast("cuda", dtype=torch.bfloat16):
    #        output_dict = self.model.forward(batch_vla)
    #        action_loss = output_dict["action_loss"]
    # ==================================================================== #
    logger.info("=" * 60)
    logger.info("STEP 5: Training forward pass (model.train() + autocast bfloat16)")
    logger.info("=" * 60)

    # training mode — same as training
    model.train()

    losses = []
    sample_diagnostics = []
    data_iter = iter(dataloader)

    for sample_idx in range(args.num_samples):
        try:
            batch = next(data_iter)
        except StopIteration:
            logger.warning(f"DataLoader exhausted after {sample_idx} samples")
            break

        # Log first sample details
        if sample_idx == 0:
            s0 = batch[0]
            action = s0["action"]
            action_np = np.array(action) if not isinstance(action, np.ndarray) else action
            action_d = describe("action", action_np)
            logger.info(f"Sample 0 — action: shape={action_d['shape']}, dtype={action_d['dtype']}, "
                        f"range=[{action_d['min']:.6f}, {action_d['max']:.6f}], mean={action_d['mean']:.6f}")
            diagnostics["sample_0_action"] = action_d

            images = s0["image"]
            logger.info(f"Sample 0 — images: {len(images)} PIL images")
            for i, img in enumerate(images):
                img_np = np.array(img)
                logger.info(f"  image_{i}: size={img.size}, mode={img.mode}, "
                            f"shape={img_np.shape}, mean={img_np.mean():.1f}")
                img.save(os.path.join(args.output_dir, f"sample0_image_{i}.png"))

            lang = s0.get("lang", s0.get("language", "N/A"))
            logger.info(f"Sample 0 — lang: '{lang}'")
            diagnostics["sample_0_lang"] = str(lang)

            if "state" in s0:
                state_np = np.array(s0["state"])
                state_d = describe("state", state_np)
                logger.info(f"Sample 0 — state: shape={state_d['shape']}, "
                            f"range=[{state_d['min']:.6f}, {state_d['max']:.6f}]")
                diagnostics["sample_0_state"] = state_d
            else:
                logger.info("Sample 0 — state: NOT present")

        # --- EXACT training forward pass ---
        t0 = time.perf_counter()
        try:
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output_dict = model.forward(batch)
                    action_loss = output_dict["action_loss"]

            loss_val = action_loss.item()
            dt = time.perf_counter() - t0
            losses.append(loss_val)
            sample_diagnostics.append({"sample_idx": sample_idx, "loss": loss_val, "time_s": dt})
            logger.info(f"  Sample {sample_idx}: loss = {loss_val:.6f}  ({dt:.2f}s)")

        except Exception as e:
            logger.error(f"  Sample {sample_idx}: FORWARD FAILED: {e}")
            import traceback
            traceback.print_exc()
            sample_diagnostics.append({"sample_idx": sample_idx, "error": str(e)})

    diagnostics["forward_losses"] = sample_diagnostics

    if losses:
        avg_loss = float(np.mean(losses))
        std_loss = float(np.std(losses))
        logger.info("=" * 60)
        logger.info("LOSS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  Samples tested: {len(losses)}")
        logger.info(f"  Average loss:   {avg_loss:.6f}")
        logger.info(f"  Std loss:       {std_loss:.6f}")
        logger.info(f"  Min loss:       {min(losses):.6f}")
        logger.info(f"  Max loss:       {max(losses):.6f}")
        diagnostics["loss_summary"] = {
            "count": len(losses),
            "mean": avg_loss,
            "std": std_loss,
            "min": float(min(losses)),
            "max": float(max(losses)),
        }

        if any(np.isnan(l) for l in losses):
            logger.error("⚠️  Some losses are NaN!")
        elif avg_loss > 10:
            logger.warning(f"⚠️  Average loss very high ({avg_loss:.4f})")
        elif avg_loss > 1.0:
            logger.warning(f"⚠️  Average loss moderately high ({avg_loss:.4f}) — compare with training log")
        else:
            logger.info(f"✅  Average loss ({avg_loss:.4f}) looks reasonable")

    # ==================================================================== #
    # 6. Also test predict_action for comparison
    # ==================================================================== #
    logger.info("=" * 60)
    logger.info("STEP 6: predict_action vs GT (inference path)")
    logger.info("=" * 60)

    model.eval()
    data_iter2 = iter(dataloader)
    try:
        batch = next(data_iter2)
    except StopIteration:
        batch = None

    if batch is not None:
        try:
            with torch.no_grad():
                pred_output = model.predict_action(examples=batch)
            norm_actions = pred_output["normalized_actions"]
            d = describe("predicted_normalized_actions", norm_actions)
            logger.info(f"  pred actions: shape={d['shape']}, range=[{d['min']:.6f}, {d['max']:.6f}], "
                        f"mean={d['mean']:.6f}, std={d['std']:.6f}")
            diagnostics["predicted_actions"] = d

            gt_action = np.array(batch[0]["action"], dtype=np.float64)
            future_window = cfg.framework.action_model.future_action_window_size
            gt_target = gt_action[-(future_window + 1):, :]
            pred_single = norm_actions[0]

            if pred_single.shape == gt_target.shape:
                mse = float(np.mean((pred_single.astype(np.float64) - gt_target) ** 2))
                logger.info(f"  MSE (pred vs GT normalized): {mse:.6f}")
                diagnostics["pred_vs_gt_mse"] = mse
            else:
                logger.warning(f"  Shape mismatch: pred={pred_single.shape} vs gt_target={gt_target.shape}")

            np.save(os.path.join(args.output_dir, "predicted_actions.npy"), norm_actions)
            np.save(os.path.join(args.output_dir, "gt_actions.npy"), gt_action)
        except Exception as e:
            logger.error(f"  predict_action FAILED: {e}")
            import traceback
            traceback.print_exc()

    # ==================================================================== #
    # 7. Save diagnostics
    # ==================================================================== #
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    diag_path = os.path.join(args.output_dir, "ckpt_verify_diagnostics.json")
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2, default=convert)
    logger.info(f"Diagnostics saved to {diag_path}")

    logger.info("=" * 60)
    logger.info("CHECKPOINT VERIFICATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
