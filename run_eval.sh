#!/bin/bash
# One evaluation pass over already generated code.
#
# The script evaluates every eval dir with the same pipeline version, archives
# the per-sample verdicts of this pass into OUT_DIR and writes a README next to
# them. Run it twice with two different OUT_DIR values and you get two runs over
# the same generated code, which is what Section 4.4 of the thesis compares to
# obtain the reporting thresholds.
#
# Runs INSIDE the co1lin/cweval container, against the repo mounted at /host/CWEval:
#   docker run -d --name cweval_eval \
#     -v /path/to/CWEval:/host/CWEval \
#     co1lin/cweval bash /host/CWEval/run_eval.sh
#
# Optional: pass eval dir names to evaluate only those, e.g.
#   ... bash /host/CWEval/run_eval.sh eval_gpt54 eval_haiku45

set -u

# ===========================================================================
#  ADJUST THESE TWO FOR EACH PASS
# ===========================================================================

# Where this pass stores its results. Use a different directory for every pass.
#   first pass:   evals/_run_A_2026-08-06
#   second pass:  evals/_run_B_2026-08-06
OUT_DIR=evals/_run_A_$(date +%F)

# Leave EMPTY for the first pass. Then the live files under evals/eval_<model>/
# hold this pass, and the report scripts in tools/ read it.
#
# For the second pass put the OUT_DIR of the first pass here. After each model
# the live files are restored from it, so the working tree stays on the first
# pass and the reports keep reading it. The second pass then exists only in its
# own OUT_DIR, which is exactly what data_basis_report.py compares against.
RESTORE_FROM=

# ===========================================================================
#  Everything below stays as it is
# ===========================================================================

# repo root inside the container; override with REPO=... if mounted elsewhere
REPO=${REPO:-/host/CWEval}

# Hard time limit per model. Run A needed at most 53 minutes for a model. The
# image ships an amd64 Go toolchain, and a single cgo build has been observed to
# hang indefinitely, which once blocked a pass for nine hours.
LIMIT=${LIMIT:-7200}

source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate cweval

# same environment as CWEval's .env, but pointing at the mounted repo
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
export NODE_PATH=$(npm root -g)
export C_INCLUDE_PATH="$CONDA_PREFIX/include"
export LIBRARY_PATH="$CONDA_PREFIX/lib"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
export PATH=$PATH:/usr/local/go/bin:$HOME/go/bin

# the image ships an amd64 go toolchain; on an Apple Silicon host, Docker
# emulates amd64 and cgo tasks need to target arm64 instead to compile/run
# natively. On a native amd64 host (e.g. tower-pc) the toolchain's own
# architecture already matches - forcing arm64 unconditionally broke that case.
HOST_ARCH=$(uname -m)
if [ "$HOST_ARCH" = "arm64" ] || [ "$HOST_ARCH" = "aarch64" ]; then
    export GOARCH=arm64
fi
export CGO_ENABLED=1

cd "$REPO"
export PYTHONPATH="$REPO"

LOGDIR=evals/eval_logs
mkdir -p "$LOGDIR"

# run all models by default, or only the ones passed as arguments
if [ $# -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=(
        eval_glm45
        eval_glm52
        eval_kimik27
        eval_kimik2think
        eval_minimaxm2
        eval_minimaxm3
        eval_gpt54
        eval_gpt54mini
        eval_sonnet46
        eval_haiku45
        eval_gemini3flash
        eval_gemini31pro
    )
fi

# --- guards ----------------------------------------------------------------

if [ -n "$RESTORE_FROM" ] && [ "$RESTORE_FROM" = "$OUT_DIR" ]; then
    echo "ABORT: RESTORE_FROM and OUT_DIR are the same directory" | tee -a "$LOGDIR/master.log"
    exit 1
fi

# restoring is only possible from a pass that actually holds every model
if [ -n "$RESTORE_FROM" ]; then
    for m in "${MODELS[@]}"; do
        if [ ! -f "$RESTORE_FROM/$m/res_all.json" ]; then
            echo "ABORT: $RESTORE_FROM/$m/res_all.json missing" | tee -a "$LOGDIR/master.log"
            exit 1
        fi
    done
fi

# never silently overwrite a finished pass
for m in "${MODELS[@]}"; do
    if [ -f "$OUT_DIR/$m/res_all.json" ]; then
        echo "ABORT: $OUT_DIR/$m/res_all.json already exists." | tee -a "$LOGDIR/master.log"
        echo "       Change OUT_DIR, or delete that file to redo this model." | tee -a "$LOGDIR/master.log"
        exit 1
    fi
done

# --- run -------------------------------------------------------------------

mkdir -p "$OUT_DIR"
echo "pass start: $(date) OUT_DIR=$OUT_DIR RESTORE_FROM=${RESTORE_FROM:-<none>} [${MODELS[*]}]" \
    | tee -a "$LOGDIR/master.log"

for m in "${MODELS[@]}"; do
    echo "=== $m start $(date) ===" | tee -a "$LOGDIR/master.log"

    # stale build artifacts from a previous pass would be reused otherwise
    find "evals/$m" -type d \( -name compiled -o -name __pycache__ \) -exec rm -rf {} + 2>/dev/null

    timeout --signal=KILL "$LIMIT" \
        python cweval/evaluate.py pipeline \
        --eval_path "evals/$m" \
        --num_proc 8 \
        --docker False \
        > "$LOGDIR/$(basename "$OUT_DIR")_$m.log" 2>&1
    rc=$?

    if [ $rc -eq 0 ]; then
        mkdir -p "$OUT_DIR/$m"
        cp "evals/$m/res_all.json" "$OUT_DIR/$m/res_all.json"
        echo "=== $m OK $(date) ===" | tee -a "$LOGDIR/master.log"
    elif [ $rc -eq 137 ]; then
        echo "=== $m TIMEOUT after ${LIMIT}s $(date) ===" | tee -a "$LOGDIR/master.log"
    else
        echo "=== $m FAILED rc=$rc $(date) ===" | tee -a "$LOGDIR/master.log"
    fi

    # put the working tree back on the reference pass before the next model, so
    # an abort can never leave a mixture of two passes behind
    if [ -n "$RESTORE_FROM" ]; then
        cp "$RESTORE_FROM/$m/res_all.json" "evals/$m/res_all.json"
        find "evals/$m" -type d \( -name compiled -o -name __pycache__ \) -exec rm -rf {} + 2>/dev/null
    fi
done

# --- finish ----------------------------------------------------------------

if [ -z "$RESTORE_FROM" ]; then
    # the live files hold this pass, so the reports belong to it
    echo "=== regenerating CSV reports ===" | tee -a "$LOGDIR/master.log"
    python tools/passk_report.py      >> "$LOGDIR/master.log" 2>&1
    python tools/flip_report.py       >> "$LOGDIR/master.log" 2>&1
    python tools/breakdown_report.py  >> "$LOGDIR/master.log" 2>&1
    REPORTS="Regenerated from this pass: the CSV reports directly under evals/."
else
    REPORTS="The CSV reports under evals/ belong to $RESTORE_FROM and were deliberately not regenerated."
fi

cat > "$OUT_DIR/README.md" <<EOT
# $(basename "$OUT_DIR")

Evaluation pass over the generated code under evals/eval_<model>/, produced by
run_eval.sh with OUT_DIR=$OUT_DIR and RESTORE_FROM=${RESTORE_FROM:-<none>}.

Holds res_all.json per model, that is the per-sample verdicts of this pass.

$REPORTS
EOT

echo "pass done: $(date)" | tee -a "$LOGDIR/master.log"
