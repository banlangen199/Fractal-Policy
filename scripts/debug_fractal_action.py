#!/usr/bin/env python3
"""Debug script for FractalPolicy using synthetic data only.

This script does not load dataset/env. It composes Hydra config, instantiates the
fractal method, builds dummy tensors, and runs forward loss + inference act.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Iterable

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


def _shape_str(x):
    if torch.is_tensor(x):
        return f"Tensor{tuple(x.shape)}"
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(_shape_str(v) for v in x) + "]"
    if isinstance(x, dict):
        return "{" + ", ".join(f"{k}: {_shape_str(v)}" for k, v in x.items()) + "}"
    return type(x).__name__


def _count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _register_shape_hooks(model: torch.nn.Module) -> Iterable[torch.utils.hooks.RemovableHandle]:
    handles = []

    def hook_fn(module, inputs, output):
        cls = module.__class__.__name__
        print(f"[HOOK] {cls}: in={_shape_str(inputs)} out={_shape_str(output)}")

    for m in model.modules():
        if m is model:
            continue
        if len(list(m.children())) == 0:
            handles.append(m.register_forward_hook(hook_fn))
    return handles


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Debug FractalPolicy with synthetic tensors")
    parser.add_argument("--mode", choices=["forward", "sample", "both"], default="both")
    parser.add_argument("--task", default="turn_tap", help="Placeholder task for Hydra interpolation only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--image-hw",
        type=int,
        default=None,
        help="Optional hint only; actual dummy image size follows cfg.method.encoder_model.input_shape",
    )
    parser.add_argument("--with-hooks", action="store_true", help="Print per-module input/output shapes")
    parser.add_argument("--pdb", action="store_true", help="Enter post-mortem pdb on exception")
    parser.add_argument("--breakpoint", action="store_true", help="Trigger breakpoint() before running")
    return parser


def _build_dummy_batch(
    batch_size: int,
    action_seq: int,
    action_dim: int,
    input_shape: tuple[int, int, int, int],
) -> dict[str, torch.Tensor]:
    _, channels, height, width = input_shape
    rgb_shape = (batch_size, 1, channels, height, width)
    return {
        "left_shoulder_rgb": torch.randint(0, 256, rgb_shape, dtype=torch.uint8),
        "right_shoulder_rgb": torch.randint(0, 256, rgb_shape, dtype=torch.uint8),
        "wrist_rgb": torch.randint(0, 256, rgb_shape, dtype=torch.uint8),
        "front_rgb": torch.randint(0, 256, rgb_shape, dtype=torch.uint8),
        "low_dim_state": torch.randn(batch_size, 8, dtype=torch.float32),
        "action": torch.randn(batch_size, action_seq, action_dim, dtype=torch.float32),
        "is_pad": torch.zeros(batch_size, action_seq, dtype=torch.bool),
    }


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def _build_agent(task: str):
    # Avoid downloading pretrained weights during local debug.
    import src.methods.backbone as backbone_mod

    backbone_mod.is_main_process = lambda: False

    repo_root = Path(__file__).resolve().parents[1]
    cfg_dir = repo_root / "src" / "cfgs"
    with initialize_config_dir(config_dir=str(cfg_dir), version_base=None):
        cfg = compose(
            config_name="launch",
            overrides=[
                "method=fractal_action",
                f"task={task}",
                "wandb.use=false",
                "debug=true",
            ],
        )
    agent = instantiate(cfg.method, accelerator=None)
    return cfg, agent


def main() -> int:
    args = _build_parser().parse_args()
    torch.manual_seed(args.seed)

    print("[Info] Synthetic-data-only debug run. No dataset/env loading.")
    cfg, agent = _build_agent(task=args.task)

    total_params, trainable_params = _count_parameters(agent)
    print(f"[PARAMS] total={total_params:,} trainable={trainable_params:,}")

    action_seq = int(cfg.method.action_sequence)
    action_dim = int(cfg.method.actor_model.action_dim)
    cfg_input_shape = tuple(int(v) for v in cfg.method.encoder_model.input_shape)
    if args.image_hw is not None and args.image_hw != cfg_input_shape[2]:
        print(
            f"[Warn] --image-hw={args.image_hw} mismatches cfg input height={cfg_input_shape[2]}; "
            f"using cfg shape {cfg_input_shape}."
        )
    batch = _build_dummy_batch(
        batch_size=args.batch_size,
        action_seq=action_seq,
        action_dim=action_dim,
        input_shape=cfg_input_shape,
    )
    batch = _to_device(batch, agent.device)

    hooks = []
    if args.with_hooks:
        hooks = list(_register_shape_hooks(agent))

    if args.breakpoint:
        breakpoint()

    try:
        if args.mode in ("forward", "both"):
            agent.training_mode(True)
            total_loss, loss_dict = agent._compute_loss(batch)
            print(f"[FORWARD] total_loss={float(total_loss.detach()):.6f}")
            print(f"[FORWARD] action_loss={float(loss_dict['action_loss'].detach()):.6f}")

        if args.mode in ("sample", "both"):
            agent.training_mode(False)
            with torch.no_grad():
                pred = agent.act(batch)
            print(f"[SAMPLE] pred shape: {tuple(pred.shape)}")

    except Exception:
        traceback.print_exc()
        if args.pdb:
            import pdb

            pdb.post_mortem(sys.exc_info()[2])
        return 1
    finally:
        for h in hooks:
            h.remove()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())