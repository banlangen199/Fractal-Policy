# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chain-of-Action (CoA) — Trajectory Autoregressive Modeling for Robotic Manipulation (NeurIPS 2025, ByteDance). A PyTorch framework for training RGB-only visuomotor policies on RLBench tasks. Built on the [robobase](https://github.com/robobase-org/robobase) framework.

## Environment & Installation

```bash
conda create -n coa python=3.9 -y && conda activate coa
bash scripts/init.sh    # installs system deps, CoppeliaSim, and pip install -e .[rlbench]
source ~/.bashrc
```

`setup.py` declares core dependencies (PyTorch not pinned — install separately). Dev extras: `pytest`, `black`, `flake8`, `mypy`.

## Commands

### Data download
```bash
python scripts/download_dataset.py                           # all tasks
python scripts/download_dataset.py --task push_button --train-episodes 100 --eval-episodes 25
python scripts/download_dataset.py --subset                  # recommended 10-task subset
```

### Model download
```bash
python scripts/download_pretrained_model.py --task push_button
```

### Evaluation (one-click)
```bash
bash scripts/eval.sh task=push_button                        # auto-downloads snapshot + dataset
python scripts/eval.py task=task_name snapshot=path_to_snapshot
```

### Training
```bash
python scripts/train.py task=push_button
python scripts/train.py task=push_button batch_size=64       # override any config key
```

Entry points are registered: `coa-train` and `coa-eval` (see `setup.py`).

## Architecture

### Entry point & training loop
`src/workspace.py` — Hydra main (`config_path="cfgs/", config_name="launch"`). `Workspace.__init__` instantiates everything: accelerator (HuggingFace Accelerate), env factory, dataset, dataloader, method (agent via `hydra.utils.instantiate(cfg.method)`), eval env, logger, video recorder. `Workspace.loop()` runs the main training loop, calling `agent.update(batch)` each step.

### Method hierarchy
```
src/methods/base.py          → BaseMethod(nn.Module, ABC)  — defines forward(), prepare_accelerator()
  ├── src/methods/coa/coa.py → CoA(BaseMethod)             — ImageEncoder + ActorModel + Transformer
  └── src/methods/fractal/policy.py → FractalPolicy(BaseMethod) — ImageEncoder + ActorModel + FractalAction
```

Both methods share the same overall pattern: `ImageEncoder` (ResNet-18 backbone + Conv2D projection + sinusoidal position embeddings from `src/methods/backbone.py`) extracts features from 4-view RGB images, then an `ActorModel` with a transformer produces action sequences.

**CoA** (`src/methods/coa/coa.py`): DETR-style encoder-decoder transformer. Encodes image tokens + proprioception as memory, uses learned action queries with sinusoidal position embeddings for decoding. Supports MTP (Multi-Token Prediction) heads. Action order can be FORWARD/REVERSE/HYBRID. Loss: L1 action loss + L1 latent loss.

**FractalPolicy** (`src/methods/fractal/policy.py`): Uses a standard TransformerEncoder to build memory from image+proprio tokens, then a hierarchical fractal decoder (`FractalAction` in `src/methods/fractal/TransformerDecoder/fractal_action.py`) that autoregressively generates actions at multiple scales (e.g., 32→16→4→1).

Action representation: 8-dim = [x, y, z, qx, qy, qz, qw, gripper]. Actions are normalized to [-1, 1] tanh space via `MinMaxNorm`.

### Environment (`src/envs/rlbench/`)
`RLBenchEnvFactory` wraps RLBench into Gymnasium envs. Key wrappers: `RescaleFromTanh` (MinMaxNorm), `ActionSequence`, `TemporalEnsemble`/`ReverseTemporalEnsemble` (receding horizon execution), `FrameStack`. Action modes: `ABS_END_EFFECTOR_POSE` (primary), `ABS_JOINT_POSITION`. Observation extraction filters RGB cameras per config and builds `low_dim_state` (gripper_pose + gripper_open, min-max normalized).

### Dataset (`src/dataset/rlbench_dataset.py`)
`RLBenchDataset` — loads demos from `RLBenchEnvFactory._load_demos()`. For CoA: `keypoint_discovery` splits trajectories at gripper-change/keyframe points, then `get_action_coa` builds reversed action sequences with zero-padding and MTP conversion. For FractalPolicy: uses sliding window sampling (`get_action_window`). Both output dicts with keys: `front_rgb`, `wrist_rgb`, `left_shoulder_rgb`, `right_shoulder_rgb` (CHW, uint8), `low_dim_state` (float32), `action` (float32), `is_pad` (bool).

### Configuration (`src/cfgs/`)
Hydra configs with composition:
```
launch.yaml  →  base/base_config.yaml, env/rlbench.yaml, method/{coa,fractal,act,dp}.yaml, env/episode_length.yaml
```
`launch.yaml` is the top-level config. Method is selected via `defaults: method: coa`. Task-specific episode lengths are in `src/cfgs/env/tasks/*.yaml`. Override any key via CLI: `python scripts/train.py task=push_button batch_size=64`.

Results are saved to `exp_local/{method_name}/rlbench_{task_name}_{timestamp}/` with checkpoints, eval videos, and `.hydra/` config snapshots.

### Key utilities
- `src/utils.py` — `DemoStep` (dict subclass for observations), point cloud helpers (`make_pcd`, `merge_pcds`), distribution classes (`SquashedNormal`, `TruncatedNormal`), schedule parsing
- `src/methods/utils.py` — `extract_many_from_batch`, `flatten_time_dim_into_channel_dim`, `stack_tensor_dictionary`
- `src/logger.py` — WandB + console logging
- `src/video.py` — `VideoRecorder` for eval rollouts

## Important notes
- The HuggingFace dataset omits point cloud observations to reduce storage. Visualization code in `Workspace.vis()` is commented out and non-functional without full point cloud data.
- `RLBenchDataset` requires `_demos` to be set externally by the env factory (not loaded internally).
- CoA uses `num_queries` = `action_sequence` (dynamic, computed as max sub-trajectory length after keypoint splitting).
- FrozenBatchNorm2d is used by default (`use_frozen_bn: true` in method configs) — backbone BN stats are fixed during training.
- The open-source version uses L1 loss (vs L2 in the paper) and 2 MTP heads (vs 5 in the paper), generally yielding better results.
