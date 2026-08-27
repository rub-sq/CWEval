#!/usr/bin/env bash
# CWEval generation for the six proprietary models (OpenAI, Anthropic, Google).
#
# Usage:
#   bash run_models.sh                     # all 6 models; skips already-complete ones
#   bash run_models.sh <name> [<name>...]  # one or more specific models
#
# Model names: gpt54  gpt54mini  sonnet46  haiku45  gemini31pro  gemini3flash
#
# Prerequisites:
#   export OPENROUTER_API_KEY="sk-or-..."
#   source .env

set -uo pipefail
: "${OPENROUTER_API_KEY:?Please export OPENROUTER_API_KEY before running}"
export OPENROUTER_API_KEY

# make cweval importable without requiring `source .env` manually
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

# ---------------------------------------------------------------------------
# Generation parameters
# n=20 supports pass@k for every k up to 20; the thesis reports k in {1, 10}
# max_completion_tokens=32768 prevents reasoning models from being cut off
# num_proc=8 parallelises across tasks; lower to 1 if hitting rate limits
# ---------------------------------------------------------------------------
N=20
TEMPERATURE=0.8
MAX_TOKENS=32768
NUM_PROC=8

# ---------------------------------------------------------------------------
# Model registry  (name -> OpenRouter slug)
# ---------------------------------------------------------------------------
model_slug() {
    case "$1" in
        gpt54)       echo "openrouter/openai/gpt-5.4" ;;
        gpt54mini)   echo "openrouter/openai/gpt-5.4-mini" ;;
        sonnet46)    echo "openrouter/anthropic/claude-sonnet-4.6" ;;
        haiku45)     echo "openrouter/anthropic/claude-haiku-4.5" ;;
        gemini31pro) echo "openrouter/google/gemini-3.1-pro-preview" ;;
        gemini3flash) echo "openrouter/google/gemini-3-flash-preview" ;;
        *) return 1 ;;
    esac
}

# run order when no argument is given
ALL_MODELS=(gpt54 gpt54mini sonnet46 haiku45 gemini31pro gemini3flash)

# ---------------------------------------------------------------------------
# Skip check: a model is considered complete when generated_{N-1} exists
# and contains at least one _raw.* file (generate.py skips per-file anyway,
# but this avoids launching Python at all for a fully finished model).
# ---------------------------------------------------------------------------
is_complete() {
    local eval_dir="$1"
    local last_gen="${eval_dir}/generated_$((N - 1))"
    [[ -d "$last_gen" ]] && \
        [[ -n "$(find "$last_gen" -name '*_raw.*' -print -quit 2>/dev/null)" ]]
}

# ---------------------------------------------------------------------------
# Generation helper
# ---------------------------------------------------------------------------
run_model() {
    local name="$1"

    local model
    model=$(model_slug "$name") || {
        echo "Unknown model: '$name'. Valid names: ${ALL_MODELS[*]}" >&2
        return 1
    }
    local eval_dir="evals/eval_${name}"

    echo ""
    echo "======================================================"
    echo "  Model : $name"
    echo "  Slug  : $model"
    echo "  Output: $eval_dir"
    echo "======================================================"

    if is_complete "$eval_dir"; then
        echo "  -> already complete (generated_$((N-1)) exists), skipping"
        return 0
    fi

    if python cweval/generate.py gen \
        --model    "$model" \
        --n        "$N" \
        --temperature "$TEMPERATURE" \
        --max_completion_tokens "$MAX_TOKENS" \
        --num_proc "$NUM_PROC" \
        --ppt      direct \
        --eval_path "$eval_dir"; then
        echo "  -> $name done"
    else
        echo "  -> $name FAILED" >&2
        return 1
    fi
}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if [[ $# -gt 0 ]]; then
    for name in "$@"; do
        run_model "$name"
    done
else
    for name in "${ALL_MODELS[@]}"; do
        run_model "$name" || true   # continue even if one model fails
    done
fi

echo ""
echo "======================================================"
echo "  Generation complete. Results: evals/eval_<name>/"
echo "  Token report: python tools/token_report.py --eval_path evals/eval_<name>"
echo "======================================================"
