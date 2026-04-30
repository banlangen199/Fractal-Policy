# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
"""
Training script for Chain-of-Action models.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from src.workspace import Workspace


@hydra.main(config_path="../src/cfgs/", config_name="launch", version_base=None)
def main(cfg):
    workspace = Workspace(cfg, train=True)
    workspace.loop()

if __name__ == "__main__":
    main()


# export LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri
# export MESA_LOADER_DRIVER_OVERRIDE=swrast
# export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7
# export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu

# xvfb-run -a -s "-screen 0 1920x1080x24" \
# python scripts/eval.py \
# task=turn_tap \
# snapshot=./exp_local/coa/train_turn_tap_20260122113135/checkpoints/coa_20000.pt \
# env.renderer=opengl
# xvfb-run -a -s "-screen 0 1920x1080x24" python scripts/train.py task=turn_tap env.renderer=opengl
# xvfb-run -a -s "-screen 0 1920x1080x24" python scripts/eval.py task=turn_tap snapshot=exp_local/coa/train_turn_tap_FOLD_20260309225057/checkpoints/coa_20000.pt env.renderer=opengl
# xvfb-run -a -s "-screen 0 1920x1080x24" python scripts/eval.py task=turn_tap snapshot=exp_local/coa/rlbench_turn_tap_20260122113135/checkpoints/coa_20000.pt env.renderer=opengl

# CUDA_VISIBLE_DEVICES=6 xvfb-run -a -s "-screen 0 1920x1080x24" python scripts/train.py task=turn_tap method=FOLDcoa env.renderer=opengl3
# xvfb-run -a -s "-screen 0 1920x1080x24" python scripts/eval.py task=turn_tap snapshot=exp_local/FOLDcoa/train_turn_tap_20260320122150/checkpoints/FOLDcoa_20000.pt env.renderer=opengl3 method=FOLDcoa

# CUDA_VISIBLE_DEVICES=6 xvfb-run -a -s "-screen 0 1920x1080x24" python scripts/eval.py task=turn_tap env.renderer=opengl3 snapshot=coa/rlbench_turn_tap_20250702080230/checkpoints/rlbench_turn_tap_20250702080230_coa_20000.pt
# CUDA_VISIBLE_DEVICES=5 xvfb-run -a -s "-screen 0 1920x1080x24" python scripts/eval.py task=push_button env.renderer=opengl3 snapshot=coa/rlbench_push_button_20250702092806/checkpoints/rlbench_push_button_20250702092806_coa_20000.pt
# CUDA_VISIBLE_DEVICES=4 xvfb-run -a -s "-screen 0 1920x1080x24" python scripts/eval.py task=pick_up_cup env.renderer=opengl3 snapshot=coa/rlbench_pick_up_cup_20250702080155/checkpoints/rlbench_pick_up_cup_20250702080155_coa_20000.pt
# CUDA_VISIBLE_DEVICES=6 xvfb-run -a -s "-screen 0 1920x1080x24" python scripts/train.py task=turn_tap method=fractal env.renderer=opengl3