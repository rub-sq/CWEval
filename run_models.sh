#!/usr/bin/env bash
# CWEval generation for proprietary models (OpenAI, Anthropic, Google) via OpenRouter.
#
# Usage:
#   bash run_models.sh                     # all models in ALL_MODELS; skips already-complete ones
#   bash run_models.sh <name> [<name>...]  # one or more specific models
#   N=10 bash run_models.sh <name>...      # override n (also: TEMPERATURE, MAX_TOKENS, NUM_PROC)
#
# Model names: gpt56sol  gpt56luna  sonnet5  haiku45  gemini31pro  gemini37flash
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
# Generation parameters (override via env, e.g. N=10 bash run_models.sh ...)
# n=20 supports pass@k for every k up to 20; the thesis reports k in {1, 10}
# max_completion_tokens=32768 prevents reasoning models from being cut off
# num_proc=8 parallelises across tasks; lower to 1 if hitting rate limits
# ---------------------------------------------------------------------------
N="${N:-20}"
TEMPERATURE="${TEMPERATURE:-0.8}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
NUM_PROC="${NUM_PROC:-8}"

# ---------------------------------------------------------------------------
# Model registry  (name -> OpenRouter slug; verified against
# https://openrouter.ai/api/v1/models on 2026-08-27)
# ---------------------------------------------------------------------------
model_slug() {
    case "$1" in
        gpt56sol)      echo "openrouter/openai/gpt-5.6-sol" ;;
        gpt56luna)     echo "openrouter/openai/gpt-5.6-luna" ;;
        sonnet5)       echo "openrouter/anthropic/claude-sonnet-5" ;;
        haiku45)       echo "openrouter/anthropic/claude-haiku-4.5" ;;
        gemini31pro)   echo "openrouter/google/gemini-3.1-pro-preview" ;;
        gemini37flash) echo "openrouter/google/gemini-3.7-flash" ;;
        *) return 1 ;;
    esac
}

# run order when no argument is given: the new proprietary comparison set
ALL_MODELS=(gpt56sol gpt56luna sonnet5 haiku45 gemini31pro gemini37flash)

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
