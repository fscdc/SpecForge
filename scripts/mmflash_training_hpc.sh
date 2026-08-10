
export PYTHONPATH=/home/svu/fengsicheng/Projects/SpecForge
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # fix

# for deep100
# export PYTHONPATH=/home/fengsicheng/Projects/SpecForge
# export LD_LIBRARY_PATH="/home/svu/fengsicheng/miniconda3/envs/specforge/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH}"
# export FLASHINFER_USE_CUDA_NORM=1
# export NVCC_PREPEND_FLAGS="-ccbin g++-11"

# for hopper
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export FLASHINFER_USE_CUDA_NORM=1

rm -rf /scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge/outputs/qwen3.5-4b-mmflash/control
rm -rf /scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge/outputs/qwen3.5-4b-mmflash/consumer-state

specforge train --config scripts/mmtraining_configs/qwen3.5-4b-mmflash_hpc.yaml


# 跑之前check一下
# ps -ef | grep -E "specforge|sglang.launch_server|mooncake_master" | grep -v grep

# 如果上面check有残留，则要全部清除
# pkill -f "specforge train"
# pkill -f sglang.launch_server
# pkill -f mooncake_master