
# Same benchmark as benchmark_mmflash.sh, plus the target model's entropy at
# DFlash verify time.
#
# What is recorded: for every verify step, the entropy of the target's
# distribution at each of the BLOCK_SIZE positions it just checked, split by
# what happened to that position (accepted / the rejection point / discarded
# because it sat behind a rejected draft token). The rejection point is the
# interesting one: it says whether the draft loses on tokens the target itself
# was unsure about, or on tokens the target considered obvious.
#
# Requires the DFlash verify-entropy patch in the INSTALLED sglang:
#   SGL_PARENT=$(python -c 'import sglang, os; print(os.path.dirname(os.path.dirname(sglang.__file__)))')
#   patch -p2 -d "$SGL_PARENT" < patches/sglang/v0.5.14/dflash-verify-entropy.patch
# (reverse it with the same command plus --reverse). The preflight check below
# refuses to run without it, rather than producing an empty dump an hour later.

# for draft model, we need to convert it to a sglang loadable format first
# specforge export --to hf \
#   --checkpoint /scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge/outputs/qwen3.5-4b-mmflash-sharegpt4v/qwen3.5-4b-mmflash-latest \
#   --draft-config configs/qwen3.5-4b-dflash.json \
#   --output-dir /scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge/draft_models/qwen3.5-4b-mmflash-sharegpt4v

export CUDA_VISIBLE_DEVICES=0

GPU_IDS=(0)

# regen dataset: sharegpt4v

# for deep100
# export LD_LIBRARY_PATH="/home/svu/fengsicheng/miniconda3/envs/specforge/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH}"
# export FLASHINFER_USE_CUDA_NORM=1
# export NVCC_PREPEND_FLAGS="-ccbin g++-11"

# for hopper
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export FLASHINFER_USE_CUDA_NORM=1


# z-lab/Qwen3.5-4B-DFlash
# /scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge/draft_models/qwen3.5-4b-mmflash-sharegpt4v
# /scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge/draft_models/qwen3.5-4b-mmflash-hf 这个是一个只有1000step的test版本

BLOCK_SIZE=16

# --- entropy recording -------------------------------------------------------
# Absolute, so the server and the benchmark agree on it no matter where either
# is started from. The recorder appends and adds a .pid<PID> suffix per process,
# so stale shards from an earlier run are cleared first.
ENTROPY_DUMP="$(cd "$(dirname "$0")/.." && pwd)/results/mmflash_verify_entropy.jsonl"
mkdir -p "$(dirname "${ENTROPY_DUMP}")"
rm -f "${ENTROPY_DUMP%.jsonl}".pid*.jsonl "${ENTROPY_DUMP}"

# Fail now if the installed sglang has no recorder: without it the server runs
# fine but writes nothing, and that only shows up after the whole benchmark.
python3 - <<'PY' || exit 1
import os, sys, sglang
worker = os.path.join(
    os.path.dirname(sglang.__file__), "srt", "speculative", "dflash_worker_v2.py"
)
if "DFlashVerifyEntropyRecorder" not in open(worker).read():
    sys.exit(
        "ERROR: installed sglang has no DFlash verify-entropy recorder.\n"
        "Apply it first:\n"
        "  SGL_PARENT=$(python -c 'import sglang, os; "
        "print(os.path.dirname(os.path.dirname(sglang.__file__)))')\n"
        "  patch -p2 -d \"$SGL_PARENT\" < "
        "patches/sglang/v0.5.14/dflash-verify-entropy.patch"
    )
print(f"verify-entropy recorder present in {worker}")
PY
# -----------------------------------------------------------------------------

# Patch the installed SGLang so every response carries its prefill/decode split
# (first_token_latency / decode_latency). Must run before launch_server imports
# tokenizer_manager.py. `bash scripts/benchmark_helper.sh --unpatch` undoes it.
bash scripts/benchmark_helper.sh || exit 1


IFS=',' read -ra VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

SERVER_ADDRESSES=()
PORTS=()
BASE_URLS=()
for idx in "${!GPU_IDS[@]}"; do
    gpu_id="${VISIBLE_GPUS[${GPU_IDS[$idx]}]:-}"
    if [ -z "${gpu_id}" ]; then
        echo "GPU ${GPU_IDS[$idx]} is not among the ${#VISIBLE_GPUS[@]} GPU(s) of this job" >&2
        exit 1
    fi
    port=$((30000 + idx * 10))
    SERVER_ADDRESSES+=("localhost:${port}")
    PORTS+=("${port}")
    BASE_URLS+=("http://localhost:${port}")
    # SGLANG_DFLASH_ENTROPY_DUMP has to be set HERE: the benchmark runs with
    # --skip-launch-server, so it never gets to set it on the server itself.
    CUDA_VISIBLE_DEVICES=${gpu_id} \
    SGLANG_DFLASH_ENTROPY_DUMP="${ENTROPY_DUMP}" \
    python3 -m sglang.launch_server \
        --model Qwen/Qwen3.5-4B \
        --speculative-algorithm DFLASH \
        --speculative-draft-model-path /scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge/draft_models/qwen3.5-4b-mmflash-sharegpt4v \
        --speculative-dflash-block-size ${BLOCK_SIZE} \
        --mem-fraction-static 0.7 \
        --tp 1 \
        --trust-remote-code \
        --cuda-graph-max-bs 128 \
        --attention-backend triton \
        --mm-attention-backend sdpa \
        --host 0.0.0.0 \
        --port ${port} \
        --dtype bfloat16 \
        --reasoning-parser qwen3 &
done


# The servers load in the background, so wait for all of them before benchmarking.
# Polling is done with python rather than curl, which is not available everywhere,
# and it costs nothing but the standard library. Replace the whole block with
# `sleep 300` if you would rather wait blind.
python3 - "${SERVER_ADDRESSES[@]}" <<'PY'
import sys
import time
import urllib.error
import urllib.request

TIMEOUT = 1800  # a cold start pulling weights from disk can take a while
deadline = time.time() + TIMEOUT

for address in sys.argv[1:]:
    while True:
        try:
            urllib.request.urlopen(f"http://{address}/health", timeout=5).read()
            print(f"server {address} is ready", flush=True)
            break
        except (urllib.error.URLError, OSError) as error:
            if time.time() > deadline:
                sys.exit(f"server {address} was not ready within {TIMEOUT}s: {error}")
            time.sleep(5)
PY
# stop here if a server never came up, benchmarking would only fail later anyway
if [ $? -ne 0 ]; then
    exit 1
fi


CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_mm.py \
    --model Qwen/Qwen3.5-4B \
    --base-url "${BASE_URLS[@]}" \
    --concurrency 1 \
    --block-size ${BLOCK_SIZE} \
    --benchmark-list chartqa:200 \
    --reasoning off \
    --temperature 0.0 \
    --top-p 0.95 \
    --top-k 20 \
    --max-tokens 8192 \
    --save-generations \
    --verify-entropy-dump "${ENTROPY_DUMP}" \
    --name mmflash_qwen35-4B_concurrency1_entropy


# Kill the servers only after the benchmark has read the dump. The recorder
# writes line-buffered, so nothing in flight is lost here.
pkill -f "sglang.launch_server"

echo
echo "verify-entropy shards:"
ls -la "${ENTROPY_DUMP%.jsonl}".pid*.jsonl 2> /dev/null || echo "  (none written)"
