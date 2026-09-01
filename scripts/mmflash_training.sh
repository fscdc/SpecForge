# TODO@song: 松哥可以不用export这些设置
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # fix

# for deep100
export PYTHONPATH=/home/fengsicheng/Projects/SpecForge
export LD_LIBRARY_PATH="/home/fengsicheng/miniconda3/envs/specforge/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH}"
export FLASHINFER_USE_CUDA_NORM=1
export NVCC_PREPEND_FLAGS="-ccbin g++-11"

# for hopper
# export PYTHONPATH=/home/svu/fengsicheng/Projects/SpecForge
# export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
# export FLASHINFER_USE_CUDA_NORM=1

# export TORCHINDUCTOR_COMPILE_THREADS=1

export TRITON_CACHE_DIR="/tmp/${USER}_triton"
export TORCHINDUCTOR_CACHE_DIR="/tmp/${USER}_inductor"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# rm -rf /scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge/outputs/qwen3.5-4b-mmflash-sharegpt4v/control
# rm -rf /scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge/outputs/qwen3.5-4b-mmflash-sharegpt4v/consumer-state

rm -rf /local_home2/fengsicheng/specforge/outputs/qwen3.5-4b-dflash-baseline-llava-ov15-1M

specforge train --config scripts/mmtraining_configs/qwen3.5-4b-mmflash.yaml

# 跑之前check一下
# ps -ef | grep -E "specforge|sglang.launch_server|mooncake_master" | grep -v grep

# pkill -f "specforge train"
# pkill -f sglang.launch_server
# pkill -f mooncake_master