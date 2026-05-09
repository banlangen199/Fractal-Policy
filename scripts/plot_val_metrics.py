#!/usr/bin/env python

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


PREFIX_RE = re.compile(r"^prefix_(\d+)_(action|pos|ori|gripper)_loss$")
TIMESTEP_RE = re.compile(r"^t(\d+)_(action|pos|ori|gripper)_loss$")


def load_metrics(path: Path) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    cleaned = {}
    for k, v in metrics.items():
        try:
            cleaned[k] = float(v)
        except Exception:
            continue

    return cleaned


def parse_metric_arg(arg: str) -> Tuple[Path, str]:
    """
    Accept:
        /path/to/offline_val_metrics.json
        /path/to/offline_val_metrics.json:label
    """
    if ":" in arg:
        path_str, label = arg.split(":", 1)
        return Path(path_str), label
    path = Path(arg)
    return path, path.parent.name


def extract_prefix(metrics: Dict[str, float], component: str) -> Tuple[List[int], List[float]]:
    xs, ys = [], []

    for k, v in metrics.items():
        m = PREFIX_RE.match(k)
        if m is None:
            continue

        horizon = int(m.group(1))
        comp = m.group(2)

        if comp == component:
            xs.append(horizon)
            ys.append(v)

    pairs = sorted(zip(xs, ys), key=lambda p: p[0])
    if not pairs:
        return [], []

    xs, ys = zip(*pairs)
    return list(xs), list(ys)


def extract_timestep(metrics: Dict[str, float], component: str) -> Tuple[List[int], List[float]]:
    xs, ys = [], []

    for k, v in metrics.items():
        m = TIMESTEP_RE.match(k)
        if m is None:
            continue

        timestep = int(m.group(1))
        comp = m.group(2)

        if comp == component:
            xs.append(timestep)
            ys.append(v)

    pairs = sorted(zip(xs, ys), key=lambda p: p[0])
    if not pairs:
        return [], []

    xs, ys = zip(*pairs)
    return list(xs), list(ys)


def maybe_clip(xs: List[int], ys: List[float], max_x: int | None) -> Tuple[List[int], List[float]]:
    if max_x is None:
        return xs, ys

    clipped = [(x, y) for x, y in zip(xs, ys) if x <= max_x]
    if not clipped:
        return [], []

    xs, ys = zip(*clipped)
    return list(xs), list(ys)


def plot_prefix(
    runs: List[Tuple[str, Dict[str, float]]],
    out_dir: Path,
    component: str,
    max_prefix: int | None,
):
    plt.figure(figsize=(8, 5))

    has_data = False
    for label, metrics in runs:
        xs, ys = extract_prefix(metrics, component)
        xs, ys = maybe_clip(xs, ys, max_prefix)

        if xs:
            has_data = True
            plt.plot(xs, ys, marker="o", label=label)

    if not has_data:
        print(f"[skip] No prefix data for component={component}")
        plt.close()
        return

    plt.xlabel("Prefix horizon")
    plt.ylabel(f"{component} loss")
    plt.title(f"Prefix {component} loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = out_dir / f"prefix_{component}_loss.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def plot_timestep(
    runs: List[Tuple[str, Dict[str, float]]],
    out_dir: Path,
    component: str,
    max_timestep: int | None,
):
    plt.figure(figsize=(10, 5))

    has_data = False
    for label, metrics in runs:
        xs, ys = extract_timestep(metrics, component)
        xs, ys = maybe_clip(xs, ys, max_timestep)

        if xs:
            has_data = True
            plt.plot(xs, ys, marker="o", markersize=3, label=label)

    if not has_data:
        print(f"[skip] No timestep data for component={component}")
        plt.close()
        return

    plt.xlabel("Timestep")
    plt.ylabel(f"{component} loss")
    plt.title(f"Per-timestep {component} loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = out_dir / f"timestep_{component}_loss.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def plot_summary_bar(
    runs: List[Tuple[str, Dict[str, float]]],
    out_dir: Path,
):
    keys = [
        "action_loss",
        "first_action_loss",
        "prefix_4_action_loss",
        "prefix_8_action_loss",
        "prefix_16_action_loss",
        "prefix_32_action_loss",
    ]

    labels = []
    values_by_key = {k: [] for k in keys}

    for label, metrics in runs:
        labels.append(label)
        for k in keys:
            values_by_key[k].append(metrics.get(k, None))

    available_keys = [
        k for k in keys
        if any(v is not None for v in values_by_key[k])
    ]

    if not available_keys:
        print("[skip] No summary keys found.")
        return

    x = list(range(len(labels)))
    width = 0.8 / max(1, len(available_keys))

    plt.figure(figsize=(max(8, len(labels) * 1.5), 5))

    for i, k in enumerate(available_keys):
        ys = [
            v if v is not None else 0.0
            for v in values_by_key[k]
        ]
        offsets = [
            xi - 0.4 + width / 2 + i * width
            for xi in x
        ]
        plt.bar(offsets, ys, width=width, label=k)

    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("loss")
    plt.title("Validation summary")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = out_dir / "summary_bar.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        nargs="+",
        required=True,
        help=(
            "Metric JSON paths. Format: path or path:label. "
            "Example: a.json:depth_first b.json:levelwise"
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="val_plots",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        default=["action", "pos", "ori", "gripper"],
        choices=["action", "pos", "ori", "gripper"],
    )
    parser.add_argument(
        "--max_timestep",
        type=int,
        default=None,
        help="Only plot timesteps <= max_timestep.",
    )
    parser.add_argument(
        "--max_prefix",
        type=int,
        default=None,
        help="Only plot prefix horizons <= max_prefix.",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for item in args.metrics:
        path, label = parse_metric_arg(item)
        if not path.exists():
            raise FileNotFoundError(path)

        metrics = load_metrics(path)
        runs.append((label, metrics))

        print(f"Loaded {label}: {path}")
        if "action_loss" in metrics:
            print(f"  action_loss = {metrics['action_loss']:.6f}")

    for component in args.components:
        plot_prefix(
            runs=runs,
            out_dir=out_dir,
            component=component,
            max_prefix=args.max_prefix,
        )
        plot_timestep(
            runs=runs,
            out_dir=out_dir,
            component=component,
            max_timestep=args.max_timestep,
        )

    plot_summary_bar(runs, out_dir)


if __name__ == "__main__":
    main()