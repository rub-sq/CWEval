#!/usr/bin/env bash
# CWEval generation for proprietary models (OpenAI, Anthropic, Google) via OpenRouter.
#
# Usage:
#   bash run_models.sh                     # all models in ALL_MODELS; skips already-complete ones
#   bash run_models.sh <name> [<name>...]  # one or more specific models
#   N=10 bash run_models.sh <name>...      # override n (also: TEMPERATURE, MAX_TOKENS, NUM_PROC)
#   BATCH_MODE=1 bash run_models.sh <name>...   # submit the model's whole shortfall (up to 119*N
#                                           # requests) as ONE OpenRouter batch at ~half price,
#                                           # instead of NUM_PROC parallel synchronous calls.
#                                           # Same generated_N/*_raw.*/*_meta.*.json output either
#                                           # way. Can take up to 24h to complete (polls until
#                                           # done); safe to Ctrl-C and rerun the same command -
#                                           # it resumes the same submitted batch, doesn't resubmit.
#                                           # NOTE: with multiple model names in one invocation,
#                                           # each model's batch is submitted+polled to completion
#                                           # before the next model starts - they do NOT run
#                                           # concurrently, so N models each near the 24h window
#                                           # run sequentially, not in parallel.
#
# Model names: gpt56sol  gpt56luna  haiku45  gemini31pro  gemini37flash
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
# max_completion_tokens=65536 matches the open-weight arm's completion cap
# (hpc/gen_part_0X.slurm) for a fair comparison - actual usage in the N=1
# pilot never exceeded 16% of the cap, so this is effectively free;
# REASONING_MAX_TOKENS below is what actually bounds cost
# num_proc=8 parallelises across tasks; lower to 1 if hitting rate limits
# ---------------------------------------------------------------------------
N="${N:-20}"
TEMPERATURE="${TEMPERATURE:-0.8}"
MAX_TOKENS="${MAX_TOKENS:-65536}"
NUM_PROC="${NUM_PROC:-8}"
# Caps reasoning-token spend (OpenRouter `reasoning.max_tokens`, forwarded via
# litellm's extra_body). Applies to ALL models for methodological consistency,
# not just the ones known to reason heavily - set to "" to disable entirely.
# See openrouter.ai/docs/guides/best-practices/reasoning-tokens.
REASONING_MAX_TOKENS="${REASONING_MAX_TOKENS:-2048}"
BATCH_MODE="${BATCH_MODE:-0}"

# ---------------------------------------------------------------------------
# Model registry  (name -> OpenRouter slug; verified against
# https://openrouter.ai/api/v1/models on 2026-08-27)
# ---------------------------------------------------------------------------
model_slug() {
    case "$1" in
        gpt56sol)      echo "openrouter/openai/gpt-5.6-sol" ;;
        gpt56luna)     echo "openrouter/openai/gpt-5.6-luna" ;;
        haiku45)       echo "openrouter/anthropic/claude-haiku-4.5" ;;
        gemini31pro)   echo "openrouter/google/gemini-3.1-pro-preview" ;;
        gemini37flash) echo "openrouter/google/gemini-3.7-flash" ;;
        *) return 1 ;;
    esac
}

# run order when no argument is given: the five proprietary models
ALL_MODELS=(gpt56sol gpt56luna haiku45 gemini31pro gemini37flash)

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

    # Existing samples are kept: only the shortfall is generated, into a temp
    # path, then merged in at the next free indices (same trick as
    # hpc/gen_part_0X.slurm). Without this, generate.py's per-task skip check
    # only fires when ALL N samples for a task already exist - bumping N and
    # rerunning would otherwise silently regenerate (overwrite) every sample.
    local existing_n=0
    [[ -d "$eval_dir" ]] && existing_n=$(find "$eval_dir" -maxdepth 1 -type d -name 'generated_*' 2>/dev/null | wc -l | tr -d ' ')

    local need_n=$(( N - existing_n ))
    local gen_dir="$eval_dir"
    if [[ "$existing_n" -gt 0 ]]; then
        gen_dir="${eval_dir}_topup"
        echo "  -> have $existing_n, need $need_n more -> generating into $gen_dir"
    else
        echo "  -> generating $need_n samples"
    fi

    local extra_body_args=()
    if [[ -n "$REASONING_MAX_TOKENS" ]]; then
        extra_body_args=(--extra_body "{\"reasoning\": {\"max_tokens\": $REASONING_MAX_TOKENS}}")
    fi

    local batch_args=()
    if [[ "$BATCH_MODE" = "1" ]]; then
        echo "  -> batch mode: submitting as one OpenRouter batch (can take up to 24h)"
        batch_args=(--batch True)
    fi

    # --assume_yes skips generate.py's "already exists, continue?" prompt
    # (e.g. a leftover _topup dir from a prior crashed run) instead of
    # blindly piping 'y' into stdin - nothing is ever deleted either way,
    # only missing samples get filled in.
    if ! python cweval/generate.py gen \
        --model    "$model" \
        --n        "$need_n" \
        --temperature "$TEMPERATURE" \
        --max_completion_tokens "$MAX_TOKENS" \
        --num_proc "$NUM_PROC" \
        --ppt      direct \
        --assume_yes True \
        "${extra_body_args[@]}" \
        "${batch_args[@]}" \
        --eval_path "$gen_dir"; then
        echo "  -> $name FAILED" >&2
        return 1
    fi

    if [[ "$gen_dir" != "$eval_dir" ]]; then
        echo "  -> merging $gen_dir into $eval_dir (indices $existing_n..$((N-1)))"
        for i in $(seq 0 $(( need_n - 1 ))); do
            src="$gen_dir/generated_$i"
            dst="$eval_dir/generated_$(( existing_n + i ))"
            [[ -d "$src" ]] && mv "$src" "$dst"
        done
        rmdir "$gen_dir" 2>/dev/null || true

        # generate.py numbers sample_index locally to each run (always
        # starting at 0); rewrite it to match the final generated_N index
        # now that the mv above has settled where each sample actually lives.
        python - "$eval_dir" "$existing_n" "$need_n" <<'PYEOF'
import glob, json, os, sys

eval_dir, existing_n, need_n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
for offset in range(need_n):
    final_index = existing_n + offset
    for meta_path in glob.glob(
        os.path.join(eval_dir, f'generated_{final_index}', '**', '*_meta.*.json'),
        recursive=True,
    ):
        with open(meta_path) as f:
            meta = json.load(f)
        meta['sample_index'] = final_index
        with open(meta_path, 'w') as f:
            json.dump(meta, f)
PYEOF
    fi

    echo "  -> $name done"
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
