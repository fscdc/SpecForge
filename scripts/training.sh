# TODO@song: 松哥可以不用export这些设置
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # fix

# for deep100
# export LD_LIBRARY_PATH="/home/fengsicheng/miniconda3/envs/specforge/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH}"
# export FLASHINFER_USE_CUDA_NORM=1
# export NVCC_PREPEND_FLAGS="-ccbin g++-11"

# for hopper
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export FLASHINFER_USE_CUDA_NORM=1
export SGLANG_NUMA_BIND_V2=0

# export TORCHINDUCTOR_COMPILE_THREADS=1

# Compile caches. /tmp on these nodes is a ~5 GB node-local ext4 shared by every
# job on the box, and it runs full -- the login node's is at 100% right now with
# nothing of ours in it. `attention_backend: flex_attention` means torch.compile
# runs for every new shape, so the caches are written throughout training, and a
# full /tmp kills the run partway with "OSError: [Errno 28] No space left on
# device". Keyed by host so jobs landing on the same node still share a warm
# cache, while two nodes never write the same NFS files.
CACHE_ROOT="${SPECFORGE_CACHE_ROOT:-/scratch/${USER}/tmp}"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton-$(hostname -s)"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/inductor-$(hostname -s)"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" || {
    echo "cannot create compile caches under ${CACHE_ROOT}; set SPECFORGE_CACHE_ROOT" >&2
    exit 1
}

# rm -rf /scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge/outputs/qwen3.5-4b-mmflash-sharegpt4v/control
# rm -rf /scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge/outputs/qwen3.5-4b-mmflash-sharegpt4v/consumer-state

# rm -rf /local_home2/fengsicheng/specforge/outputs/qwen3.5-4b-dflash-baseline-llava-ov15-1M

# dflash baseline
# specforge train --config scripts/mmtraining_configs/qwen3.5-4b-dflash.yaml

# ours: mmflash
MMFLASH_CONFIG=scripts/mmtraining_configs/qwen3.5-4b-mmflash.yaml

# qsub -l select=1:ngpus=4 -v CONFIG=${MMFLASH_CONFIG} scripts/score_visual_kl_hpc.sh


bash scripts/score_visual_kl_hpc.sh ${MMFLASH_CONFIG} 

specforge train --config ${MMFLASH_CONFIG}

