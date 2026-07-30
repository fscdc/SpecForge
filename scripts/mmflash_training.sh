# NOTE: do not pin CUDA_VISIBLE_DEVICES here. managed_local assigns GPUs itself
# (trainer 0,1 + capture server 2); pinning 0,1 would hide GPU 2 from the server.
export PYTHONPATH=/home/fengsicheng/Projects/SpecForge

export LD_LIBRARY_PATH="/home/fengsicheng/miniconda3/envs/specforge/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH}"
export FLASHINFER_USE_CUDA_NORM=1
export NVCC_PREPEND_FLAGS="-ccbin g++-11"

rm -rf /local_home1/fengsicheng/specforge/outputs/qwen3.5-4b-mmflash/control
rm -rf /local_home1/fengsicheng/specforge/outputs/qwen3.5-4b-mmflash/consumer-state

specforge train --config scripts/mmtraining_configs/qwen3.5-4b-mmflash.yaml