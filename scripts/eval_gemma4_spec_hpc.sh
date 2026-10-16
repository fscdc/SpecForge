#!/bin/bash
# Accept length of an exported Gemma 4 E4B draft (EAGLE3, DFlash or MMFlash) on
# a small mixed image + text set, one GPU, with the target served the way its
# training capture ran it. Submit from the repo root:
#
#   qsub -v ALGO=origin scripts/eval_gemma4_spec_hpc.sh                     # the target alone
#   qsub -v ALGO=dflash,DRAFT=/path/to/export scripts/eval_gemma4_spec_hpc.sh
#   qsub -v ALGO=eagle3,DRAFT=/path/to/export scripts/eval_gemma4_spec_hpc.sh
#
# Exports (CPU, from a training checkpoint <ckpt>):
#   EAGLE3          specforge export --to sglang --checkpoint <ckpt> --draft-config configs/gemma4-e4b-eagle3.json \
#                       --vocab-mapping <vocab_mapping/gemma4-e4b-eagle3.pt> --output-dir <export>
#   DFlash/MMFlash  specforge export --to hf --checkpoint <ckpt> --draft-config configs/gemma4-e4b-<dflash|mmflash>.json --output-dir <export>
#                   python scripts/gates/normalize_dflash_export.py --config <export>/config.json --block-size 16
# An MMFlash export is served like a DFlash one (ALGO=dflash), as SGLang's DFLASH
# worker serves both.
#
# The server matches capture: the pinned target snapshot the recipes train on,
# --attention-backend triton (the only Gemma4 backend on Hopper), --mm-attention-backend
# sdpa (the regen's and capture's vision attention), --reasoning-parser gemma4,
# and the gemma4-spec SGLang patch (applied by scripts/benchmark_helper.sh,
# checked below), since the drafts are trained on its features.
# SGLANG_GEMMA_OUT_OF_PLACE_POSITION_MUTATION=1 keeps Gemma's positions += 1 off
# the DFLASH worker's own positions buffer, whose RoPE it applies to the
# committed draft K/V.
#
# bench_mm.py runs image (and video) benchmarks only, text ones live in
# bench_text.py, so the mixed set is two runs against the same server, greedy,
# thinking off, at most MAX_TOKENS new tokens, generations saved next to the
# results (an ALGO=origin run gives the reference outputs):
#   results/image_<name>_results.jsonl   IMAGE_BENCHES (default "chartqa:10 mmstar:10")
#   results/text_<name>_results.jsonl    TEXT_BENCHES  (default "gsm8k:10 math500:10")
# with <name> = gemma4-e4b_<algo>_concurrency1_temp0_<MAX_TOKENS>[_<export dir name>]<RUN_TAG>.
# Both skip a benchmark their results file already holds, so a rerun finishes a
# partial one.
#
# Optional (-v NAME=value):
#   MODEL              the target (default: the pinned snapshot below)
#   MAX_TOKENS         default 512
#   BLOCK_SIZE         DFLASH block size, default 16 (the drafts' block_size)
#   EAGLE3_STEPS       EAGLE3 draft steps, default 15 (16 draft tokens, top-1,
#                      as the Qwen3.5 EAGLE3 runs)
#   IMAGE_BENCHES / TEXT_BENCHES   "" skips that half
#   EXTRA_SERVER_ARGS  more server flags, e.g. --disable-cuda-graph
#   RUN_TAG            appended to the results name
#   SPECFORGE_ALLOW_UNPATCHED_GEMMA4=1   neither apply nor require gemma4-spec:
#                      serve whatever the installed tree carries (its state is
#                      logged). To serve stock SGLang, run
#                      scripts/apply_sglang_gemma4_patch.sh --reverse first,
#                      which changes the shared env for every job
#PBS -P CFP04-CF-054
#PBS -j oe
#PBS -k oed
#PBS -N gemma4-eval
#PBS -q auto
#PBS -l select=1:ngpus=1
#PBS -l walltime=02:00:00

set -uo pipefail
export PYTHONUNBUFFERED=1
echo "[eval] start $(date '+%F %T') host=$(hostname) job=${PBS_JOBID:-local} gpus=${CUDA_VISIBLE_DEVICES:-unset}"

# ------------------------------------------------------------- environment
CONDA_ENV="${CONDA_ENV:-specforge}"
if ! python3 -c "import torch" > /dev/null 2>&1; then
    # /etc/bashrc and conda's activate scripts read unset variables
    set +u
    # shellcheck disable=SC1091
    [ -f "${HOME}/.bashrc" ] && source "${HOME}/.bashrc"
    conda activate "${CONDA_ENV}" > /dev/null 2>&1 || true
    set -u
fi
if ! python3 -c "import torch" > /dev/null 2>&1; then
    echo "[eval] ERROR: torch is not importable after activating '${CONDA_ENV}'" >&2
    exit 1
fi
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." 2> /dev/null && pwd)"
if [ ! -f "${PROJECT_DIR:-}/benchmarks/bench_mm.py" ] && [ -n "${PBS_O_WORKDIR:-}" ]; then
    PROJECT_DIR="${PBS_O_WORKDIR}"
fi
cd "${PROJECT_DIR}" || exit 1

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export FLASHINFER_USE_CUDA_NORM=1
export SGLANG_NUMA_BIND_V2=0
export SGLANG_FORCE_STREAM_INTERVAL=1
export SGLANG_GEMMA_OUT_OF_PLACE_POSITION_MUTATION="${SGLANG_GEMMA_OUT_OF_PLACE_POSITION_MUTATION:-1}"
CACHE_TAG="${PBS_JOBID:-local-$$}"; CACHE_TAG="${CACHE_TAG%%.*}"
export TRITON_CACHE_DIR="${SPECFORGE_CACHE_ROOT:-/scratch/${USER}/tmp}/triton-eval-${CACHE_TAG}"
mkdir -p "${TRITON_CACHE_DIR}"

# ------------------------------------------------------------- settings
MODEL="${MODEL:-/scratch/fengsicheng/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2}"
# the name the server reports and the benchmarks send and record
MODEL_NAME="${MODEL_NAME:-google/gemma-4-E4B-it}"
MODEL_TAG="${MODEL_TAG:-gemma4-e4b}"
ALGO="${ALGO:-origin}"
DRAFT="${DRAFT:-}"
MAX_TOKENS="${MAX_TOKENS:-512}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
EAGLE3_STEPS="${EAGLE3_STEPS:-15}"
IMAGE_BENCHES="${IMAGE_BENCHES-chartqa:10 mmstar:10}"
TEXT_BENCHES="${TEXT_BENCHES-gsm8k:10 math500:10}"
read -r -a extra_server_args <<< "${EXTRA_SERVER_ARGS:-}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-1800}"
for value in "${MAX_TOKENS}" "${BLOCK_SIZE}" "${EAGLE3_STEPS}"; do
    case "${value}" in ''|*[!0-9]*) echo "[eval] ERROR: MAX_TOKENS, BLOCK_SIZE and EAGLE3_STEPS must be numbers" >&2; exit 1 ;; esac
done
case "${ALGO}" in
    origin)
        [ -z "${DRAFT}" ] || { echo "[eval] ERROR: ALGO=origin serves the target alone; drop DRAFT" >&2; exit 1; }
        NAME_SUFFIX=""
        ;;
    eagle3|dflash)
        [ -n "${DRAFT}" ] || { echo "[eval] ERROR: ALGO=${ALGO} needs DRAFT=<export dir>" >&2; exit 1; }
        DRAFT="$(cd "${DRAFT}" 2> /dev/null && pwd)" || { echo "[eval] ERROR: DRAFT is not a directory" >&2; exit 1; }
        NAME_SUFFIX="_$(basename "${DRAFT}")"
        ;;
    *)
        echo "[eval] ERROR: ALGO must be origin, eagle3 or dflash (an MMFlash export is served as dflash), got '${ALGO}'" >&2
        exit 1
        ;;
esac
NAME="${MODEL_TAG}_${ALGO}_concurrency1_temp0_${MAX_TOKENS}${NAME_SUFFIX}${RUN_TAG:-}"
if [ -z "${IMAGE_BENCHES}" ] && [ -z "${TEXT_BENCHES}" ]; then
    echo "[eval] ERROR: IMAGE_BENCHES and TEXT_BENCHES are both empty" >&2
    exit 1
fi

# an export SGLang would refuse, or serve as something else, fails here
# instead of after the server's start-up
if [ "${ALGO}" != origin ]; then
    python3 - "${ALGO}" "${DRAFT}" "${BLOCK_SIZE}" <<'PY' || exit 1
import json, os, sys

algo, export, block_size = sys.argv[1], sys.argv[2], int(sys.argv[3])
path = os.path.join(export, "config.json")
if not os.path.isfile(path):
    sys.exit(f"[eval] ERROR: {path} not found")
with open(path, encoding="utf-8") as handle:
    config = json.load(handle)
arch = config.get("architectures")
problems = []
if algo == "eagle3":
    layers = (config.get("eagle_config") or {}).get("eagle_aux_hidden_state_layer_ids")
    if arch != ["LlamaForCausalLMEagle3"]:
        problems.append(f"architectures {arch}, not an EAGLE3 export (specforge export --to sglang)")
    if not layers:
        problems.append("no eagle_config.eagle_aux_hidden_state_layer_ids")
else:
    method = config.get("dflash_config") or {}
    layers = method.get("target_layer_ids")
    if arch != ["DFlashDraftModel"]:
        problems.append(f"architectures {arch}: run scripts/gates/normalize_dflash_export.py on it")
    if config.get("block_size") != block_size:
        problems.append(f"block_size {config.get('block_size')}, served with {block_size}")
    if method.get("mask_token_id") is None:
        problems.append("no dflash_config.mask_token_id (SGLang would add a mask token past Gemma's vocab)")
    if not layers:
        problems.append("no dflash_config.target_layer_ids")
    if method.get("draft_sparse"):
        problems.append("trained on a sparse context, which this script does not serve")
for problem in problems:
    print(f"[eval] ERROR: {path}: {problem}", file=sys.stderr)
if problems:
    sys.exit(1)
print(f"[eval] draft {export}: {arch[0]}, target layers {layers}")
PY
fi

IFS=',' read -ra VISIBLE <<< "${CUDA_VISIBLE_DEVICES:-0}"
GPU_ID="${VISIBLE[0]}"
JOBNUM="${PBS_JOBID:-0}"; JOBNUM="${JOBNUM%%.*}"
case "${JOBNUM}" in ''|*[!0-9]*) JOBNUM=0 ;; esac
# ends in 5: never one of train_mmflash_video_hpc.sh's capture ports (31000 + 10k)
PORT="${PORT:-$((35005 + (10#${JOBNUM} % 400) * 10))}"
LOG_DIR="logs/gemma4-eval_${PBS_JOBID:-local}"
SERVER_LOG="${LOG_DIR}/server.log"

bash scripts/benchmark_helper.sh || exit 1
# the drafts are trained on the gemma4-spec patched SGLang (copied aux hidden
# states, causal E4B image tokens, PIL-decoded JPEGs), so a server without it
# measures another target.
# With SPECFORGE_ALLOW_UNPATCHED_GEMMA4=1 the helper leaves gemma4-spec as the
# tree has it, and this only logs which state the server will run.
python3 - "${SPECFORGE_ALLOW_UNPATCHED_GEMMA4:-0}" <<'PY' || exit 1
import importlib.util, os, sys

parent = os.path.dirname(list(importlib.util.find_spec("sglang").submodule_search_locations)[0])
states = []
for target, sentinel in (
    ("sglang/srt/models/gemma4_causal.py", "aux_hidden_states.append(hidden_states.clone())"),
    ("sglang/srt/models/gemma4_mm.py", "and vision_bidirectional"),
    ("sglang/srt/multimodal/processors/gemma4.py", "gpu_image_decode = False"),
):
    path = os.path.join(parent, target)
    if not os.path.isfile(path):
        states.append((target, "missing"))
    else:
        states.append((target, "applied" if sentinel in open(path, encoding="utf-8").read() else "clean"))
missing = [target for target, state in states if state != "applied"]
if sys.argv[1] == "1":
    # one word, as benchmark_helper.sh reports it: applied | clean | partial | missing
    found = {state for _, state in states}
    state = "missing" if "missing" in found else found.pop() if len(found) == 1 else "partial"
    print(f"[eval] SPECFORGE_ALLOW_UNPATCHED_GEMMA4=1: serving the SGLang under {parent} as it is, gemma4-spec {state}")
elif missing:
    sys.exit(
        f"[eval] ERROR: the SGLang under {parent} lacks the gemma4-spec patch in {missing}; "
        "run bash scripts/benchmark_helper.sh, or set SPECFORGE_ALLOW_UNPATCHED_GEMMA4=1 to serve it unpatched"
    )
PY
mkdir -p "${LOG_DIR}" results
echo "[eval] target=${MODEL} algo=${ALGO} draft=${DRAFT:-<none: target alone>}"
echo "[eval] SGLANG_GEMMA_OUT_OF_PLACE_POSITION_MUTATION=${SGLANG_GEMMA_OUT_OF_PLACE_POSITION_MUTATION} extra server args: ${extra_server_args[*]:-<none>}"
echo "[eval] image=${IMAGE_BENCHES:-<none>} text=${TEXT_BENCHES:-<none>} max_tokens=${MAX_TOKENS} -> results/{image,text}_${NAME}_results.jsonl"
echo "[eval] gpu=${GPU_ID} port=${PORT} server log ${SERVER_LOG}"

SERVER_PID=""
stop_server() {
    if [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" 2> /dev/null; then
        kill "${SERVER_PID}" 2> /dev/null
        for _ in $(seq 1 30); do kill -0 "${SERVER_PID}" 2> /dev/null || break; sleep 1; done
        kill -9 "${SERVER_PID}" 2> /dev/null
    fi
    pkill -f "sglang.launch_server.*--port ${PORT}" 2> /dev/null
    SERVER_PID=""
}
trap stop_server EXIT

spec_args=()
bench_block_args=(--block-size 0)
if [ "${ALGO}" = eagle3 ]; then
    spec_args=(--speculative-algorithm EAGLE3 --speculative-draft-model-path "${DRAFT}"
               --speculative-num-steps "${EAGLE3_STEPS}" --speculative-eagle-topk 1
               --speculative-num-draft-tokens "$((EAGLE3_STEPS + 1))")
    bench_block_args=()   # as the Qwen3.5 EAGLE3 runs: bench's default
elif [ "${ALGO}" = dflash ]; then
    spec_args=(--speculative-algorithm DFLASH --speculative-draft-model-path "${DRAFT}"
               --speculative-dflash-block-size "${BLOCK_SIZE}")
    bench_block_args=(--block-size "${BLOCK_SIZE}")
fi
CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 -m sglang.launch_server \
    --model "${MODEL}" \
    --served-model-name "${MODEL_NAME}" \
    "${spec_args[@]}" \
    --mem-fraction-static 0.7 \
    --tp 1 \
    --trust-remote-code \
    --cuda-graph-max-bs 128 \
    --attention-backend triton \
    --mm-attention-backend sdpa \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --dtype bfloat16 \
    --reasoning-parser gemma4 \
    "${extra_server_args[@]}" > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

if ! python3 - "${PORT}" "${SERVER_TIMEOUT}" "${SERVER_PID}" <<'PY'
import os, sys, time, urllib.request
port, timeout, pid = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
deadline = time.time() + timeout
while time.time() < deadline:
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5).read()
        print(f"[eval] server ready on port {port}", flush=True)
        sys.exit(0)
    except Exception:
        pass
    try:
        os.kill(pid, 0)
    except OSError:
        print("[eval] server exited before becoming ready", flush=True)
        sys.exit(1)
    time.sleep(5)
print(f"[eval] server not ready within {timeout:.0f}s", flush=True)
sys.exit(1)
PY
then
    tail -n 40 "${SERVER_LOG}" >&2
    exit 1
fi

# The benches are only HTTP clients, but importing them runs
# sglang/test/test_utils.py, which does int(CUDA_VISIBLE_DEVICES[0]) and dies on
# the GPU UUIDs PBS hands out
status=0
if [ -n "${IMAGE_BENCHES}" ]; then
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=0 python3 benchmarks/bench_mm.py \
        --model "${MODEL_NAME}" \
        --base-url "http://127.0.0.1:${PORT}" \
        --concurrency 1 \
        "${bench_block_args[@]}" \
        --benchmark-list ${IMAGE_BENCHES} \
        --reasoning off \
        --temperature 0.0 \
        --top-p 0.95 \
        --top-k 20 \
        --max-tokens "${MAX_TOKENS}" \
        --save-generations \
        --name "image_${NAME}" || status=1
fi
if [ -n "${TEXT_BENCHES}" ]; then
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=0 python3 benchmarks/bench_text.py \
        --model "${MODEL_NAME}" \
        --base-url "http://127.0.0.1:${PORT}" \
        --concurrency 1 \
        "${bench_block_args[@]}" \
        --benchmark-list ${TEXT_BENCHES} \
        --reasoning off \
        --temperature 0.0 \
        --top-p 0.95 \
        --top-k 20 \
        --max-tokens "${MAX_TOKENS}" \
        --save-generations \
        --name "text_${NAME}" || status=1
fi
echo "[eval] $(date '+%F %T') benchmarks exited ${status}"
exit "${status}"
