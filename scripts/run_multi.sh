#!/usr/bin/env bash

set -uo pipefail

# 用法：
#   bash scripts/run_multi.sh train
#   bash scripts/run_multi.sh val
#   bash scripts/run_multi.sh env_eval

GPUS=(2 3 4)

MODE="${1:-train}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
SESSION="multi_${MODE}_${RUN_TAG}"
LOG_ROOT="logs/${RUN_TAG}_${MODE}"

mkdir -p "${LOG_ROOT}"
mkdir -p "${LOG_ROOT}/job_scripts"

echo "Project root: ${PROJECT_ROOT}"
echo "Mode: ${MODE}"
echo "Session: ${SESSION}"
echo "GPUs: ${GPUS[*]}"
echo "Log root: ${LOG_ROOT}"
echo "========================================"


# =========================
# 训练任务
# 每一项格式：
#   "任务名|完整命令"
# 注意：Hydra list override 用单引号包起来
# =========================

TRAIN_JOBS=(

"fractal_long_allmemory|python scripts/train.py method=fractal task=turn_tap val_sample_mode=depth_first num_train_steps=40000 val_every_steps=1000 val_batches=10 eval_every_steps=99999 'method.actor_model.transformer_decoder.action_size_list=[60,30,10,1]' 'method.actor_model.transformer_decoder.hidden_dim_list=[512,256,256,128]' 'method.actor_model.transformer_decoder.memory_access_list=[true,true,true]' 'method.actor_model.transformer_decoder.num_blocks_list=[4,4,4,4]' 'method.actor_model.transformer_decoder.num_heads_list=[8,8,8,8]' 'method.actor_model.transformer_decoder.generator_type_list=[ar,ar,ar]'"

"fractal_long_onememory|python scripts/train.py method=fractal task=turn_tap val_sample_mode=depth_first num_train_steps=40000 val_every_steps=1000 val_batches=10 eval_every_steps=99999 'method.actor_model.transformer_decoder.action_size_list=[60,30,10,1]' 'method.actor_model.transformer_decoder.hidden_dim_list=[512,256,256,128]' 'method.actor_model.transformer_decoder.memory_access_list=[true,false,false]' 'method.actor_model.transformer_decoder.num_blocks_list=[4,4,4,4]' 'method.actor_model.transformer_decoder.num_heads_list=[8,8,8,8]' 'method.actor_model.transformer_decoder.generator_type_list=[ar,ar,ar]'"

"fractal_long_topbottom|python scripts/train.py method=fractal task=turn_tap val_sample_mode=depth_first num_train_steps=40000 val_every_steps=1000 val_batches=10 eval_every_steps=99999 'method.actor_model.transformer_decoder.action_size_list=[60,30,10,1]' 'method.actor_model.transformer_decoder.hidden_dim_list=[512,256,256,128]' 'method.actor_model.transformer_decoder.memory_access_list=[true,false,true]' 'method.actor_model.transformer_decoder.num_blocks_list=[4,4,4,4]' 'method.actor_model.transformer_decoder.num_heads_list=[8,8,8,8]' 'method.actor_model.transformer_decoder.generator_type_list=[ar,ar,ar]'"

)

# =========================
# Offline validation 任务
# 修改 snapshot 为你的真实路径
# =========================

VAL_JOBS=(

"fractal_40k_depth_val|python scripts/val.py snapshot=exp_local/fractal/rlbench_turn_tap_20260512122144/checkpoints/fractal_40000.pt task=turn_tap method=fractal 
val_use_generated_actions=true val_sample_mode=depth_first val_batches=10 val_batch_size=128 val_output_dir=exp_local/val_compare/fractal_40000_turn_tap val_run_name=depth_first"

"fractal_40k_levelwise_val|python scripts/val.py snapshot=exp_local/fractal/rlbench_turn_tap_20260512122144/checkpoints/fractal_40000.pt task=turn_tap method=fractal 
val_use_generated_actions=true val_sample_mode=levelwise val_batches=10 val_batch_size=128 val_output_dir=exp_local/val_compare/fractal_40000_turn_tap val_run_name=levelwise"

)


# =========================
# 模拟器 rollout eval 任务
# 注意：RLBench / CoppeliaSim 不建议并行太多，容易 core dumped
# 如果不稳定，就只保留一个任务，一个一个跑
# =========================

ENV_EVAL_JOBS=(

"fractal_40k_depth_eval_exe2|python scripts/eval.py snapshot=exp_local/fractal/rlbench_turn_tap_20260512122144/checkpoints/fractal_40000.pt task=turn_tap method=fractal 
eval_sample_mode=depth_first execution_length=2"

"fractal_40k_depth_eval_exe4|python scripts/eval.py snapshot=exp_local/fractal/rlbench_turn_tap_20260512122144/checkpoints/fractal_40000.pt task=turn_tap method=fractal 
eval_sample_mode=depth_first execution_length=4"

"fractal_40k_levelwise_eval_exe1|python scripts/eval.py snapshot=exp_local/fractal/rlbench_turn_tap_20260512122144/checkpoints/fractal_40000.pt task=turn_tap method=fractal 
eval_sample_mode=levelwise execution_length=1"

"fractal_40k_levelwise_eval_exe2|python scripts/eval.py snapshot=exp_local/fractal/rlbench_turn_tap_20260512122144/checkpoints/fractal_40000.pt task=turn_tap method=fractal 
eval_sample_mode=levelwise execution_length=2"

"fractal_40k_levelwise_eval_exe4|python scripts/eval.py snapshot=exp_local/fractal/rlbench_turn_tap_20260512122144/checkpoints/fractal_40000.pt task=turn_tap method=fractal 
eval_sample_mode=levelwise execution_length=4"

)


create_job_window() {
  local gpu="$1"
  local name="$2"
  local cmd="$3"

  local safe_name
  safe_name="$(echo "${name}" | sed 's/[^a-zA-Z0-9_.=-]/_/g')"

  local log_file="${LOG_ROOT}/${safe_name}.log"
  local job_script="${LOG_ROOT}/job_scripts/${safe_name}.sh"

  cat > "${job_script}" <<EOF
#!/usr/bin/env bash
set -uo pipefail

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${gpu}"

echo "========================================"
echo "Job name: ${name}"
echo "GPU: ${gpu}"
echo "Log file: ${log_file}"
echo "Start time: \$(date)"
echo "Command:"
echo "${cmd}"
echo "========================================"

${cmd} 2>&1 | tee "${log_file}"

echo "========================================"
echo "Job finished: ${name}"
echo "End time: \$(date)"
echo "Log saved to: ${log_file}"
echo "========================================"

exec \${SHELL:-/bin/bash}
EOF

  chmod +x "${job_script}"

  echo "Creating tmux window:"
  echo "  name: ${name}"
  echo "  gpu:  ${gpu}"
  echo "  log:  ${log_file}"
  echo "----------------------------------------"

  tmux new-window -t "${SESSION}" -n "${safe_name:0:30}" "bash '${job_script}'"
}


launch_jobs_in_tmux() {
  local jobs=("$@")
  local num_gpus="${#GPUS[@]}"
  local job_idx=0

  if [[ "${#jobs[@]}" -eq 0 ]]; then
    echo "No jobs found for mode=${MODE}"
    exit 1
  fi

  tmux new-session -d -s "${SESSION}" -n "manager"

  tmux send-keys -t "${SESSION}:manager" "cd '${PROJECT_ROOT}'" C-m
  tmux send-keys -t "${SESSION}:manager" "echo 'Session: ${SESSION}'" C-m
  tmux send-keys -t "${SESSION}:manager" "echo 'Logs: ${LOG_ROOT}'" C-m
  tmux send-keys -t "${SESSION}:manager" "echo 'Use Ctrl+b w to switch windows; Ctrl+b d to detach.'" C-m

  for job in "${jobs[@]}"; do
    if [[ "${job}" != *"|"* ]]; then
      echo "[ERROR] Invalid job entry:"
      echo "${job}"
      echo "Each job must be formatted as: name|command"
      exit 1
    fi

    local name="${job%%|*}"
    local cmd="${job#*|}"

    if [[ -z "${cmd}" || "${cmd}" == "${name}" ]]; then
      echo "[ERROR] Empty command for job: ${name}"
      exit 1
    fi

    local gpu="${GPUS[$((job_idx % num_gpus))]}"
    create_job_window "${gpu}" "${name}" "${cmd}"

    job_idx=$((job_idx + 1))
  done

  echo "========================================"
  echo "Created tmux session: ${SESSION}"
  echo "Attach with:"
  echo "  tmux attach -t ${SESSION}"
  echo ""
  echo "Useful keys inside tmux:"
  echo "  Ctrl+b w    choose window"
  echo "  Ctrl+b n    next window"
  echo "  Ctrl+b p    previous window"
  echo "  Ctrl+b d    detach"
  echo ""
  echo "Kill all jobs/session:"
  echo "  tmux kill-session -t ${SESSION}"
  echo "========================================"

  tmux attach -t "${SESSION}"
}


case "${MODE}" in
  train)
    launch_jobs_in_tmux "${TRAIN_JOBS[@]}"
    ;;

  val)
    launch_jobs_in_tmux "${VAL_JOBS[@]}"
    ;;

  env_eval)
    launch_jobs_in_tmux "${ENV_EVAL_JOBS[@]}"
    ;;

  *)
    echo "Unknown mode: ${MODE}"
    echo "Use one of: train | val | env_eval"
    exit 1
    ;;
esac