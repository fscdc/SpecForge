#!/bin/bash
# Sparse-context sweep for the DFlash/MMFlash draft on the video benchmark,
# inference only (no retraining). Builds on the window sweep: window 2048 gave
# accept 2.55 vs 2.23 for the full context. Each variant below keeps that
# recent window and adds one ingredient of a Slow-Fast style sparse pattern,
# so the table reads as an ablation:
#
#   sink4                  + the first 4 prompt positions (attention sinks)
#   sink4-text             + every text token (instruction, generated answer)
#   sink4-text-stride32    + 1/32 of every frame's tokens (overview, ~1.3k)
#   sink4-text-stride8     + 1/8  of every frame's tokens (~5k)
#   win512-sink4-text-stride32   small window, does the overview replace it?
#
# Needs the patched worker (patches/sglang/v0.5.14/dflash-draft-sparse-context.patch,
# already applied in the specforge env). Results:
#   results/video_mmflash_qwen35-4B_concurrency1_temp0_4096_win2048_sparse-<spec>_results.jsonl
#
#   qsub scripts/sweep_draft_sparse.sh
#
# Which video benchmarks run is VIDEO_BENCHES (default "vdc:20 longvideobench:20
# moviechat:20 videomme:20 mvbench:20", see scripts/benchmark_mmflash.sh); a result file that already
# holds a benchmark only gets the missing ones on a re-run.
#   VARIANTS="sink=4,text=1,stride=16,window=2048" bash scripts/sweep_draft_sparse.sh
#PBS -P CFP04-CF-054
#PBS -j oe
#PBS -k oed
#PBS -N draft-sparse
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

# the CLI window that enables the compact draft cache; the per-variant
# window=<n> inside the spec is the one the draft actually uses
CLI_WINDOW="${CLI_WINDOW:-2048}"
VARIANTS="${VARIANTS:-\
sink=4,text=0,stride=0,window=2048 \
sink=4,text=1,stride=0,window=2048 \
sink=4,text=1,stride=32,window=2048 \
sink=4,text=1,stride=8,window=2048 \
sink=4,text=1,stride=32,window=512}"
COOLDOWN="${COOLDOWN:-30}"

for spec in ${VARIANTS}; do
    echo "==================== draft sparse: ${spec}  $(date '+%F %T') ===================="
    DRAFT_WINDOW="${CLI_WINDOW}" DRAFT_SPARSE="${spec}" NAME_SUFFIX="" bash scripts/benchmark_mmflash.sh
    status=$?
    if [ "${status}" -ne 0 ]; then
        echo "[sweep] variant ${spec} failed with status ${status}" >&2
        exit "${status}"
    fi
    pkill -f "sglang.launch_server" 2>/dev/null
    sleep "${COOLDOWN}"
done

echo "==================== summary $(date '+%F %T') ===================="
# one table per video benchmark in VIDEO_BENCHES
python3 scripts/summarize_video_results.py --benchmark "$(for b in ${VIDEO_BENCHES:-vdc:20 longvideobench:20 moviechat:20 videomme:20 mvbench:20}; do printf '%s,' "${b%%:*}"; done)"
