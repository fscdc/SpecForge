#!/bin/bash
# Gemma 4 GPU probes G1 and G2 in one 1-GPU PBS job. Submit from the repo root:
#
#   qsub scripts/gates/gemma4_probe_hpc.sh
#   qsub -v RUNS=stock:sdpa+patched:sdpa,NUM_PROMPTS=40 scripts/gates/gemma4_probe_hpc.sh
#   bash scripts/gates/gemma4_probe_hpc.sh          # interactively, on a GPU node
#
# 1. G1, scripts/gates/gemma4_capture_parity.py: captures 8 image + 8 text regen
#    rows on the real checkpoint with a capture server per RUNS entry (default:
#    stock SGLang, the gemma4-spec patch, and the patch with the vision tower on
#    fa3), then compares them with HF bf16. "stock" and "patched" are emulated
#    at import time when the installed tree differs, so both are measured
#    whatever this environment carries.
# 2. G2, scripts/gates/gemma4_serving_probe.py, against a plain serving server
#    (triton, mm sdpa, no reasoning parser) with the gemma4-spec patch: sends the
#    grayscale-JPEG rows of the regen's _error file (a supervisor relaunches the
#    server when one kills it, so every crash is counted) and NUM_PROMPTS regen
#    prompts to count answers that open an empty thought channel.
#    The serving server needs the patch installed (bash
#    scripts/apply_sglang_gemma4_patch.sh); EMULATE_PATCH=1 runs it on an
#    unpatched tree through the same import-time shim instead.
#
# Everything lands in results/gemma4_probe/<jobid>/: parity/{report.json,summary.md},
# serving/{serving_probe.json,summary.md}, logs/ (servers, probes, GPU memory
# every minute) and patch_state.txt. Ports are derived from the job number.
# qsub does not inherit the submitting shell's environment: pass settings with
# -v NAME=value. -v splits its list at every unquoted comma, so separate RUNS
# and TAPS entries with '+' there (TAPS=1+10+20); commas still work under bash.
#PBS -P CFP04-CF-054
#PBS -j oe
#PBS -k oed
#PBS -N gemma4-probe
#PBS -q auto
#PBS -l select=1:ngpus=1
#PBS -l walltime=04:00:00

set -uo pipefail
export PYTHONUNBUFFERED=1
echo "[probe] start $(date '+%F %T') host=$(hostname) job=${PBS_JOBID:-local} gpus=${CUDA_VISIBLE_DEVICES:-unset}"

# ------------------------------------------------------------- environment
CONDA_ENV="${CONDA_ENV:-specforge}"
if ! python3 -c "import torch" > /dev/null 2>&1; then
    # /etc/bashrc and conda's activate scripts read unset variables, which
    # set -u turns into an immediate exit
    set +u
    # shellcheck disable=SC1091
    [ -f "${HOME}/.bashrc" ] && source "${HOME}/.bashrc"
    conda activate "${CONDA_ENV}" > /dev/null 2>&1 || true
    set -u
fi
if ! python3 -c "import torch" > /dev/null 2>&1; then
    echo "[probe] ERROR: torch is not importable after activating '${CONDA_ENV}'" >&2
    exit 1
fi
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." 2> /dev/null && pwd)"
if [ ! -f "${PROJECT_DIR:-}/scripts/gates/gemma4_capture_parity.py" ] && [ -n "${PBS_O_WORKDIR:-}" ]; then
    PROJECT_DIR="${PBS_O_WORKDIR}"
fi
cd "${PROJECT_DIR}" || exit 1
[ -f scripts/gates/gemma4_capture_parity.py ] || {
    echo "[probe] ERROR: cannot locate the SpecForge checkout (looked in ${PROJECT_DIR})" >&2
    exit 1
}

export LD_LIBRARY_PATH="${CONDA_PREFIX:+${CONDA_PREFIX}/lib:}$(python3 -c 'import os,torch;print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):${LD_LIBRARY_PATH:-}"
export FLASHINFER_USE_CUDA_NORM=1
export SGLANG_NUMA_BIND_V2=0
# the target is a local snapshot; compute nodes need not reach the Hub
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
JOBNUM="${PBS_JOBID:-0}"; JOBNUM="${JOBNUM%%.*}"
case "${JOBNUM}" in ''|*[!0-9]*) JOBNUM=0 ;; esac
CACHE_ROOT="${SPECFORGE_CACHE_ROOT:-/scratch/${USER}/tmp}"
CACHE_TAG="${PBS_JOBID:-local-$$}"; CACHE_TAG="${CACHE_TAG%%.*}"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton-gemma4-probe-${CACHE_TAG}"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/inductor-gemma4-probe-${CACHE_TAG}"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

# ------------------------------------------------------------- settings
TARGET="${TARGET:-/scratch/fengsicheng/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2}"
DATA="${DATA:-/scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge/regen_data/gemma4-E4B-it_llava-ov15-1M-prompted_regen_first_turn.jsonl}"
RUNS="${RUNS:-stock:sdpa,patched:sdpa,patched:fa3}"
TAPS="${TAPS:-1,10,20,30,38,39}"
# '+' separates entries under qsub -v, which splits values at unquoted commas
RUNS="${RUNS//+/,}"
TAPS="${TAPS//+/,}"
NUM_IMAGE_ROWS="${NUM_IMAGE_ROWS:-8}"
NUM_TEXT_ROWS="${NUM_TEXT_ROWS:-8}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
CAPTURE_MEM="${CAPTURE_MEM:-0.40}"
SERVING_MEM="${SERVING_MEM:-0.80}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-1200}"
EMULATE_PATCH="${EMULATE_PATCH:-0}"
SKIP_PARITY="${SKIP_PARITY:-0}"
SKIP_SERVING="${SKIP_SERVING:-0}"
OUT="${OUT_DIR:-${PROJECT_DIR}/results/gemma4_probe/${JOBNUM}}"
OFFSET=$(( (10#${JOBNUM} % 400) * 10 ))
PARITY_PORT=$(( 32000 + OFFSET ))
SERVING_PORT=$(( 33000 + OFFSET ))
MC_RPC=$(( 47000 + OFFSET ))
MC_METADATA=$(( 47001 + OFFSET ))
MC_METRICS=$(( 47002 + OFFSET ))

if [ -e "${OUT}/parity" ] || [ -e "${OUT}/serving" ]; then
    OUT="${OUT}-$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "${OUT}/logs" || { echo "[probe] ERROR: cannot create ${OUT}" >&2; exit 1; }
echo "[probe] target=${TARGET}"
echo "[probe] data=${DATA}"
echo "[probe] runs=${RUNS} taps=${TAPS} rows=${NUM_IMAGE_ROWS}+${NUM_TEXT_ROWS} prompts=${NUM_PROMPTS}"
echo "[probe] ports: capture ${PARITY_PORT}, serving ${SERVING_PORT}, mooncake ${MC_RPC}/${MC_METADATA}/${MC_METRICS}"
echo "[probe] output ${OUT}"
nvidia-smi --query-gpu=index,name,memory.total,compute_mode --format=csv,noheader
(
    while true; do
        echo "$(date '+%F %T') $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';') | $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')"
        sleep 60
    done
) > "${OUT}/logs/gpu.log" 2>&1 &
GPU_LOG_PID=$!

# -------------------------------------------------- the installed gemma4-spec patch
# prints applied | clean | partial
patch_state() {
    python3 - << 'PY'
import importlib.util, os
spec = importlib.util.find_spec("sglang")
root = os.path.dirname(os.path.dirname(spec.origin)) if spec and spec.origin else ""
found = []
for relative, sentinel in (
    ("sglang/srt/models/gemma4_causal.py", "aux_hidden_states.append(hidden_states.clone())"),
    ("sglang/srt/models/gemma4_mm.py", "and vision_bidirectional"),
    ("sglang/srt/multimodal/processors/gemma4.py", "gpu_image_decode = False"),
):
    try:
        with open(os.path.join(root, relative), encoding="utf-8") as handle:
            found.append(sentinel in handle.read())
    except OSError:
        found.append(False)
print("applied" if all(found) else "clean" if not any(found) else "partial")
PY
}
PATCH_STATE="$(patch_state)"
{
    echo "sentinel state: ${PATCH_STATE}"
    if [ -f scripts/apply_sglang_gemma4_patch.sh ]; then
        bash scripts/apply_sglang_gemma4_patch.sh --check 2>&1
    fi
} > "${OUT}/patch_state.txt" 2>&1
echo "[probe] installed gemma4-spec patch: ${PATCH_STATE}"

SUPERVISOR_PID=""
STOP_FILE="${OUT}/logs/stop-serving-server"
SERVER_PID_FILE="${OUT}/logs/serving-server.pid"
# Stops the serving server's whole process group (it was started with setsid)
# and its supervisor; also runs when the job is killed.
cleanup() {
    touch "${STOP_FILE}"
    local pid
    pid="$(cat "${SERVER_PID_FILE}" 2> /dev/null)"
    if [ -n "${pid}" ]; then
        kill -TERM -- "-${pid}" 2> /dev/null || kill -TERM "${pid}" 2> /dev/null
        for _ in $(seq 1 60); do kill -0 "${pid}" 2> /dev/null || break; sleep 1; done
        kill -KILL -- "-${pid}" 2> /dev/null || kill -KILL "${pid}" 2> /dev/null
    fi
    [ -n "${SUPERVISOR_PID}" ] && wait "${SUPERVISOR_PID}" 2> /dev/null
    kill "${GPU_LOG_PID}" 2> /dev/null
}
trap cleanup EXIT

# ------------------------------------------------------------------- G1
PARITY_STATUS=skipped
if [ "${SKIP_PARITY}" != 1 ]; then
    echo "[probe] $(date '+%F %T') G1 capture parity -> ${OUT}/parity (log logs/parity.log)"
    python3 scripts/gates/gemma4_capture_parity.py \
        --target-model-path "${TARGET}" \
        --data-path "${DATA}" \
        --output-dir "${OUT}/parity" \
        --runs "${RUNS}" \
        --taps "${TAPS}" \
        --num-image-rows "${NUM_IMAGE_ROWS}" \
        --num-text-rows "${NUM_TEXT_ROWS}" \
        --port "${PARITY_PORT}" \
        --mooncake-rpc-port "${MC_RPC}" \
        --mooncake-http-port "${MC_METADATA}" \
        --mooncake-metrics-port "${MC_METRICS}" \
        --mem-fraction-static "${CAPTURE_MEM}" \
        --server-timeout "${SERVER_TIMEOUT}" \
        > "${OUT}/logs/parity.log" 2>&1
    rc=$?
    PARITY_STATUS="exit ${rc}"
    echo "[probe] $(date '+%F %T') G1 finished: ${PARITY_STATUS}"
fi

# ------------------------------------------------------------------- G2
# One serving server under a supervisor that relaunches it until STOP_FILE
# exists, so a crash costs one relaunch instead of the rest of the probe.
SERVING_STATUS=skipped
SHIM_ENV=()
if [ "${SKIP_SERVING}" != 1 ]; then
    if [ "${PATCH_STATE}" != applied ]; then
        if [ "${EMULATE_PATCH}" = 1 ]; then
            python3 scripts/gates/gemma4_capture_parity.py --write-sglang-shim "${OUT}/sglang_shim" > /dev/null
            SHIM_ENV=("PYTHONPATH=${OUT}/sglang_shim${PYTHONPATH:+:${PYTHONPATH}}" "SPECFORGE_GEMMA4_SGLANG_MODE=patched")
            echo "[probe] serving on the ${PATCH_STATE} tree with the patch emulated (EMULATE_PATCH=1)"
        else
            SERVING_STATUS="not run: the installed sglang is ${PATCH_STATE}; run scripts/apply_sglang_gemma4_patch.sh or pass EMULATE_PATCH=1"
            echo "[probe] ERROR: ${SERVING_STATUS}" >&2
        fi
    fi
fi
if [ "${SKIP_SERVING}" != 1 ] && [ "${SERVING_STATUS}" = skipped ]; then
    rm -f "${STOP_FILE}"
    supervise_server() {
        local attempt=0
        while [ ! -f "${STOP_FILE}" ]; do
            attempt=$((attempt + 1))
            echo "[server] launch #${attempt} $(date '+%F %T')" >> "${OUT}/logs/serving-server.log"
            env "${SHIM_ENV[@]}" SGLANG_GEMMA_OUT_OF_PLACE_POSITION_MUTATION=1 \
                setsid python3 -m sglang.launch_server \
                --model-path "${TARGET}" \
                --dtype bfloat16 \
                --trust-remote-code \
                --attention-backend triton \
                --mm-attention-backend sdpa \
                --mem-fraction-static "${SERVING_MEM}" \
                --context-length 8192 \
                --host 127.0.0.1 \
                --port "${SERVING_PORT}" \
                >> "${OUT}/logs/serving-server.log" 2>&1 &
            echo "$!" > "${SERVER_PID_FILE}"
            wait "$!"
            echo "[server] exited with $? $(date '+%F %T')" >> "${OUT}/logs/serving-server.log"
            [ -f "${STOP_FILE}" ] || sleep 10
        done
    }
    supervise_server &
    SUPERVISOR_PID=$!
    echo "[probe] $(date '+%F %T') G2 waiting for the serving server on port ${SERVING_PORT}"
    ready=0
    for _ in $(seq 1 $(( SERVER_TIMEOUT / 10 ))); do
        if python3 -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:${SERVING_PORT}/health_generate', timeout=30).status == 200 else 1)" 2> /dev/null; then
            ready=1
            break
        fi
        sleep 10
    done
    if [ "${ready}" = 1 ]; then
        echo "[probe] $(date '+%F %T') G2 serving probe -> ${OUT}/serving (log logs/serving_probe.log)"
        python3 scripts/gates/gemma4_serving_probe.py \
            --server-url "http://127.0.0.1:${SERVING_PORT}" \
            --model-path "${TARGET}" \
            --data-path "${DATA}" \
            --output-dir "${OUT}/serving" \
            --num-prompts "${NUM_PROMPTS}" \
            --restart-wait-s "${SERVER_TIMEOUT}" \
            > "${OUT}/logs/serving_probe.log" 2>&1
        rc=$?
        SERVING_STATUS="exit ${rc}"
    else
        SERVING_STATUS="not run: the serving server did not come up in ${SERVER_TIMEOUT}s (logs/serving-server.log)"
    fi
    LAUNCHES=$(grep -c '^\[server\] launch #' "${OUT}/logs/serving-server.log" 2> /dev/null) || LAUNCHES=0
    echo "[probe] $(date '+%F %T') G2 finished: ${SERVING_STATUS}; server launches ${LAUNCHES} (1 = no crash)"
fi

# ---------------------------------------------------------------- summary
{
    echo "job ${PBS_JOBID:-local} on $(hostname), $(date '+%F %T')"
    echo "installed gemma4-spec patch: ${PATCH_STATE}"
    echo "G1 capture parity: ${PARITY_STATUS}"
    echo "G2 serving probe: ${SERVING_STATUS}${LAUNCHES:+ (server launches ${LAUNCHES})}"
} | tee "${OUT}/status.txt"
for summary in "${OUT}/parity/summary.md" "${OUT}/serving/summary.md"; do
    if [ -f "${summary}" ]; then
        echo
        cat "${summary}"
    fi
done
case "${PARITY_STATUS}${SERVING_STATUS}" in
    *"not run"* | *"exit "[1-9]*) exit 1 ;;
esac
exit 0
