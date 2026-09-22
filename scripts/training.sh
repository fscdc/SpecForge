# TODO@song: 松哥可以不用export这些设置
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # fix

# for deep100
export LD_LIBRARY_PATH="/home/fengsicheng/miniconda3/envs/specforge/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH}"
export FLASHINFER_USE_CUDA_NORM=1
export NVCC_PREPEND_FLAGS="-ccbin g++-11"

# for hopper
# export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
# export FLASHINFER_USE_CUDA_NORM=1
# export SGLANG_NUMA_BIND_V2=0
# CACHE_ROOT="${SPECFORGE_CACHE_ROOT:-/scratch/${USER}/tmp}"
# export TRITON_CACHE_DIR="${CACHE_ROOT}/triton-$(hostname -s)"
# export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/inductor-$(hostname -s)"
# mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" || {
#     echo "cannot create compile caches under ${CACHE_ROOT}; set SPECFORGE_CACHE_ROOT" >&2
#     exit 1
# }

# export TORCHINDUCTOR_COMPILE_THREADS=1


# rm -rf /scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge/outputs/qwen3.5-4b-mmflash-sharegpt4v/control
# rm -rf /scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge/outputs/qwen3.5-4b-mmflash-sharegpt4v/consumer-state

# rm -rf /local_home2/fengsicheng/specforge/outputs/qwen3.5-4b-dflash-baseline-llava-ov15-1M


DFLASH_CONFIG=scripts/mmtraining_configs/qwen3.5-9b-dflash.yaml
# MMFLASH_CONFIG=scripts/mmtraining_configs/qwen3.5-4b-mmflash.yaml
# EAGLE3_CONFIG=scripts/mmtraining_configs/qwen3.5-4b-eagle3_hpc.yaml


# bash scripts/score_visual_kl_hpc.sh ${MMFLASH_CONFIG} # only need for mmflash

specforge train --config ${EAGLE3_CONFIG}

