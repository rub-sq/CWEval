#!/bin/bash
# Re-evaluates only the three cwe_1333_0 variants, with the fixed ReDoS oracle
# (recheck called without a shell), for all 32 models of the study, and
# writes the new verdicts into the study repository's results/ - the parent
# of this CWEval checkout (../results, with the generated code read from
# ../backups/<source>/<eval>.zip).
#
# Same three-pass procedure as the original evaluation (run_eval.sh), in a
# scratch tree that is mounted into the container as evals/:
#   pass A  RESTORE_FROM empty: updates the scratch eval_<model>/ trees in
#           place - these are the final per-sample res.json and res_all.json
#   pass B  RESTORE_FROM = pass A: re-executes the same tests, archived only,
#           then the scratch tree is put back on pass A
#   pass C  as pass B
# Then it verifies against the untouched results/ and, only if every check
# passes, writes back:
#   results/<source>/eval_<model>/res_all.json and generated_*/res.json
#       (pass A; only the three cwe_1333_0 entries differ)
#   results/<source>/noise_passes/run_{B,C}/eval_<model>/res_all.json
#       (where such a file exists: only its three cwe_1333_0 entries replaced)
# Only the tests are re-executed, on the existing generated code: no
# generation, no API or vLLM calls. Undo with git checkout -- results/.
#
# Run on the linux/amd64 tower from this CWEval checkout (the cweval/
# submodule of the study repository), e.g.
#   tmux new -s cwe1333 'bash run_cwe1333_rerun.sh'
#   nohup bash run_cwe1333_rerun.sh > cwe1333_rerun.nohup 2>&1 &
#
# Optional environment overrides:
#   MODELS="eval_a eval_b"   only these models (default: all 32)
#   WORK_ROOT=/some/dir      scratch tree + log (default: ~/cwe1333_rerun)
#   IMAGE=co1lin/cweval      container image (must be the linux/amd64 build)
#   NO_WRITEBACK=1           run and verify, but leave results/ untouched

set -uo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
STUDY="$(cd "$REPO/.." && pwd)"
cd "$REPO"

IMAGE=${IMAGE:-co1lin/cweval}
TASKS=core/py/cwe_1333_0,core/js/cwe_1333_0_js,core/cpp/cwe_1333_0_cpp
TS=$(date +%Y%m%d_%H%M%S)
WORK_ROOT=${WORK_ROOT:-$HOME/cwe1333_rerun}
WORK="$WORK_ROOT/$TS"            # mounted as /host/CWEval/evals
LOG="$WORK/rerun.log"
LOCK="$WORK_ROOT/.lock"

if [ -n "${MODELS:-}" ]; then
    read -ra MODELS <<< "$MODELS"
else
    MODELS=(
        eval_gpro_t8 eval_gemini25pro eval_gemini31pro
        eval_gflash_t8 eval_gemini25flash eval_gemini37flash
        eval_4o_t8 eval_gpt5 eval_gpt56sol
        eval_4omini_t8 eval_gpt5mini eval_gpt56luna
        eval_qwen3235b eval_qwen3coder480b eval_qwen35397b
        eval_qwen330b eval_qwen3coder30b eval_qwen3527b
        eval_deepseekv3 eval_deepseekv32 eval_deepseekv4pro eval_deepseekv4flash
        eval_glm45 eval_glm47 eval_glm52 eval_glm47flash
        eval_kimik2think eval_kimik25 eval_kimik27
        eval_minimaxm21 eval_minimaxm25 eval_minimaxm3
    )
fi

die() { echo "REFUSING TO START: $*" >&2; exit 1; }
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# --- nothing else may be evaluating ------------------------------------------

running=$(docker ps --format '{{.ID}} {{.Image}} {{.Names}}' 2>/dev/null | grep -i cweval)
[ -z "$running" ] || die "a cweval container is running:
$running"
procs=$(pgrep -af 'cweval/evaluate\.py|run_eval\.sh' | grep -v pgrep)
[ -z "$procs" ] || die "an evaluation process is running:
$procs"
mkdir -p "$WORK_ROOT"
mkdir "$LOCK" 2>/dev/null || die "$LOCK exists - another run_cwe1333_rerun.sh is running, or a crashed one left it behind (remove it if so)"
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

# --- environment checks ------------------------------------------------------

case "$(uname -m)" in
    x86_64|amd64) ;;
    *) [ "${ALLOW_NON_AMD64:-0}" = 1 ] || die "host is $(uname -m), not amd64: run this on the tower" ;;
esac
command -v python3 >/dev/null || die "python3 not found on the host"
[ -d "$STUDY/results" ] && [ -d "$STUDY/backups" ] || die "$STUDY has no results/ and backups/ - run this from the cweval/ submodule of the study repository"
arch=$(docker image inspect "$IMAGE" --format '{{.Architecture}}' 2>/dev/null) || die "image $IMAGE not found"
[ "$arch" = amd64 ] || die "image $IMAGE is $arch; recheck-linux-x64 only runs in the linux/amd64 build"
for t in benchmark/core/py/cwe_1333_0_test.py benchmark/core/js/cwe_1333_0_js_test.py \
         benchmark/core/cpp/cwe_1333_0_cpp_test.py; do
    grep -q "os.popen" "$t" && die "$t still uses os.popen - the fixed tests are missing"
    grep -q "subprocess.run(" "$t" || die "$t does not contain the fixed oracle"
done
grep -q "tasks: str = ''" cweval/evaluate.py || die "cweval/evaluate.py has no --tasks option"
grep -q 'TASKS=${TASKS:-}' run_eval.sh || die "run_eval.sh has no TASKS option"

mkdir -p "$WORK"
log "cwe_1333_0 rerun $TS: ${#MODELS[@]} models, TASKS=$TASKS, image $IMAGE ($arch)"
log "study repo: $STUDY at $(git -C "$STUDY" rev-parse --short HEAD 2>/dev/null || echo '?'), cweval at $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
log "scratch tree and log: $WORK"

# the container sees $WORK as /host/CWEval/evals; the checkout's own evals/
# (if any) is hidden from it and never touched
mkdir -p "$REPO/evals"
DOCKER_RUN=(docker run --rm --platform linux/amd64 -v "$REPO:/host/CWEval" -v "$WORK:/host/CWEval/evals")

smoke=$("${DOCKER_RUN[@]}" "$IMAGE" bash -c 'cd /host/CWEval && ./third_party/recheck-linux-x64 "/^a+$/"' 2>&1)
echo "$smoke" | grep -q '^Status *: *safe' || { log "recheck smoke test failed inside $IMAGE:"; log "$smoke"; exit 1; }
log "recheck runs inside the container"

log "building the scratch tree from results/ and backups/:"
python3 tools/cwe1333_rerun_check.py prepare "$STUDY" "$WORK" "${MODELS[@]}" 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { log "preparation failed, results/ untouched"; exit 1; }

# --- the three passes ----------------------------------------------------------

run_pass() {  # name out_dir restore_from (paths as the container sees them)
    local name=$1 out=$2 restore=$3
    log "=== pass $name start: OUT_DIR=$out RESTORE_FROM=${restore:-<none>}"
    "${DOCKER_RUN[@]}" --name "cweval_cwe1333_${name}_$TS" \
        -e OUT_DIR="$out" -e RESTORE_FROM="$restore" -e TASKS="$TASKS" \
        "$IMAGE" bash /host/CWEval/run_eval.sh "${MODELS[@]}" 2>&1 \
        | while IFS= read -r line; do
            # per-model progress and anything alarming to the console too,
            # the rest (pass_at_k tables, report output) only to the log
            case "$line" in
                "=== "*|"pass "*|ABORT*|"CHECK FAILED"*) log "pass $name: ${line#=== }" ;;
                *) echo "$line" >> "$LOG" ;;
            esac
          done
    local rc=${PIPESTATUS[0]} done=0
    for m in "${MODELS[@]}"; do
        [ -f "$WORK/${out#evals/}/$m/res_all.json" ] && done=$((done + 1))
    done
    log "=== pass $name end: exit $rc, $done/${#MODELS[@]} models archived"
    if [ "$rc" -ne 0 ] || [ "$done" -ne "${#MODELS[@]}" ]; then
        log "pass $name incomplete - stopping, results/ untouched. Per-model logs: $WORK/eval_logs/"
        exit 1
    fi
}

A=_run_A_cwe1333_$TS; B=_run_B_cwe1333_$TS; C=_run_C_cwe1333_$TS
run_pass A "evals/$A" ""
run_pass B "evals/$B" "evals/$A"
run_pass C "evals/$C" "evals/$A"

# --- verification and write-back ---------------------------------------------

log "verification against results/:"
python3 tools/cwe1333_rerun_check.py verify "$STUDY" "$WORK" "$WORK/$A" "$WORK/$B" "$WORK/$C" "${MODELS[@]}" 2>&1 | tee -a "$LOG"
vrc=${PIPESTATUS[0]}
WRITEBACK=(python3 tools/cwe1333_rerun_check.py writeback "$STUDY" "$WORK" "$WORK/$B" "$WORK/$C" "${MODELS[@]}")
if [ "$vrc" -ne 0 ]; then
    log "verification failed: results/ left untouched. To write back anyway, from $REPO:"
    log "  ${WRITEBACK[*]}"
    exit "$vrc"
fi
if [ "${NO_WRITEBACK:-0}" = 1 ]; then
    log "NO_WRITEBACK=1: results/ left untouched. To write back, from $REPO:"
    log "  ${WRITEBACK[*]}"
    exit 0
fi
log "writing back into results/:"
"${WRITEBACK[@]}" 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { log "write-back failed (git checkout -- results/ undoes a partial one)"; exit 1; }
log "done: $(git -C "$STUDY" status --short -- results | wc -l | tr -d ' ') changed files under results/ (git diff to review, git checkout -- results/ to undo)"
