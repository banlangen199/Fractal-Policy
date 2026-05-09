#!/usr/bin/env python
"""
Offline validation script.

This script loads a trained checkpoint and evaluates action-sequence prediction
against GT actions on eval/test demonstrations, without simulator rollout.

Example:
    python scripts/val.py \
        snapshot=/path/to/checkpoints/fractal_20000.pt \
        task=close_jar \
        val_batches=10 \
        val_batch_size=128 \
        val_use_generated_actions=true
"""

import os
import sys
import json
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import OmegaConf, open_dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.workspace import Workspace


def to_python_number_or_list(x: Any):
    """Convert torch tensors / numpy-like values to JSON-serializable values."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.numel() == 1:
            return float(x.item())
        return x.tolist()

    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            if x.size == 1:
                return float(x.item())
            return x.tolist()
        if isinstance(x, np.generic):
            return float(x.item())
    except Exception:
        pass

    if isinstance(x, (int, float, str, bool)) or x is None:
        return x

    try:
        return float(x)
    except Exception:
        return str(x)


@hydra.main(config_path="../src/cfgs/", config_name="launch", version_base=None)
def main(cfg):
    if cfg.snapshot is None:
        raise ValueError(
            "Please provide a checkpoint path, e.g. "
            "snapshot=/path/to/checkpoints/fractal_20000.pt"
        )

    checkpoint_path = Path(cfg.snapshot)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint config first, so the model architecture exactly matches training.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "config" not in checkpoint:
        raise RuntimeError("Checkpoint does not contain config. Cannot restore model config.")

    cfg_ckpt = checkpoint["config"]

    # Keep validation-related command-line overrides.
    val_batches = cfg.get("val_batches", 10)
    val_batch_size = cfg.get("val_batch_size", cfg.batch_size)
    val_use_generated_actions = cfg.get("val_use_generated_actions", True)

    # Merge checkpoint model config into current run config.
    # Similar to scripts/eval.py: use ckpt method/method_name/action_sequence,
    # while keeping current command-line dataset/task overrides.
    with open_dict(cfg):
        cfg.env.env_name = cfg_ckpt.env.env_name
        cfg.method = cfg_ckpt.method
        cfg.method_name = cfg_ckpt.method_name
        cfg.action_sequence = cfg_ckpt.action_sequence

        # Disable wandb for standalone validation by default.
        cfg.wandb.use = False
        cfg.log_eval_video = False

        # Restore validation settings from command line / current config.
        cfg.val_batches = val_batches
        cfg.val_batch_size = val_batch_size
        cfg.val_use_generated_actions = val_use_generated_actions

    print("=" * 80)
    print("Offline validation")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Method: {cfg.method_name}")
    print(f"Task: {cfg.env.task_name}")
    print(f"Dataset root eval: {cfg.dataset_root_eval}")
    print(f"val_batches: {cfg.val_batches}")
    print(f"val_batch_size: {cfg.val_batch_size}")
    print(f"val_use_generated_actions: {cfg.val_use_generated_actions}")
    print("=" * 80)

    # train=True is used here because Workspace currently creates dataset_val /
    # dataloader_val only in training mode.
    workspace = Workspace(cfg, train=True)

    # Make sure the checkpoint is loaded.
    # Workspace already loads cfg.snapshot in __init__ if it is not None,
    # but this explicit check makes the script easier to debug.
    if getattr(workspace, "_current_step", 0) == 0 and "step" in checkpoint:
        print(
            "Warning: workspace current step is 0 after loading. "
            "Please check whether checkpoint loading succeeded."
        )

    metrics = workspace.offline_validate()

    metrics_py = {
        k: to_python_number_or_list(v)
        for k, v in metrics.items()
    }

    print("\nValidation metrics:")
    for k in sorted(metrics_py.keys()):
        v = metrics_py[k]
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    output_path = Path(workspace.work_dir) / "offline_val_metrics.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics_py, f, indent=2, ensure_ascii=False)

    print(f"\nSaved metrics to: {output_path}")


if __name__ == "__main__":
    main()