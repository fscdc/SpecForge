#!/bin/bash
# Sliding-window sweep for the DFlash/MMFlash draft on the video benchmark.
#
# Question: does restricting the draft's attention to the most recent W target
# tokens (SGLang --speculative-draft-window-size, training-free) change the
# acceptance length and the decode throughput on VDC, where a prompt is ~43k
# tokens of which almost all are video frames?
#
# Each window is one full run of scripts/benchmark_mmflash.sh: its own server
# start, the vdc:20 pass, server kill. Results land in results/ as
#   video_mmflash_qwen35-4B_concurrency1_temp0_4096_win<W>_results.jsonl
# and "full" is re-measured in the same job under the name ..._winfull so the
# throughput numbers of the sweep come from the same GPU at the same time.
#
#   qsub -l select=1:ngpus=1 -l walltime=06:00:00 scripts/sweep_draft_window.sh
#   WINDOWS="1024 4096" bash scripts/sweep_draft_window.sh
#
# ~12 min per window on one H100 (20 videos x ~25 s prefill + decode, plus the
# server start).
#PBS -P CFP04-CF-054
#PBS -j oe
#PBS -k oed
#PBS -N draft-window
#PBS -q auto
#PBS -l select=1:ngpus=1
#PBS -l walltime=06:00:00

# Locate the checkout. The script's own directory is tried first and
# PBS_O_WORKDIR only as a fallback: under `qsub` PBS spools the script to a
# private directory, so only PBS_O_WORKDIR knows the checkout; run by hand
# inside an interactive job the reverse holds, PBS_O_WORKDIR is still exported
# from wherever that job was submitted (usually $HOME) and is wrong.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." 2>/dev/null && pwd)"
if [ ! -f "${PROJECT_DIR:-}/scripts/benchmark_mmflash.sh" ] && [ -n "${PBS_O_WORKDIR:-}" ]; then
    PROJECT_DIR="${PBS_O_WORKDIR}"
fi
if [ ! -f "${PROJECT_DIR:-}/scripts/benchmark_mmflash.sh" ]; then
    echo "ERROR: cannot locate the SpecForge checkout (scripts/benchmark_mmflash.sh); run from the repo or submit with qsub from it" >&2
    exit 1
fi
cd "${PROJECT_DIR}" || exit 1
if ! python3 -c "import torch" > /dev/null 2>&1; then
    source ~/.bashrc
    conda activate specforge
fi

WINDOWS="${WINDOWS:-512 2048 8192 full}"
# seconds to let the previous run's server free the port and the GPU
COOLDOWN="${COOLDOWN:-30}"

for w in ${WINDOWS}; do
    echo "==================== draft window: ${w}  $(date '+%F %T') ===================="
    if [ "${w}" = "full" ]; then
        DRAFT_WINDOW="" NAME_SUFFIX="_winfull" bash scripts/benchmark_mmflash.sh
    else
        DRAFT_WINDOW="${w}" NAME_SUFFIX="" bash scripts/benchmark_mmflash.sh
    fi
    status=$?
    if [ "${status}" -ne 0 ]; then
        echo "[sweep] window ${w} failed with status ${status}" >&2
        exit "${status}"
    fi
    pkill -f "sglang.launch_server" 2>/dev/null
    sleep "${COOLDOWN}"
done

echo "==================== summary $(date '+%F %T') ===================="
python3 scripts/summarize_video_results.py
