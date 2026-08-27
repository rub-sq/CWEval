#!/bin/bash
# Sanity check of Section 4.4: every benchmark reference implementation is run
# against its own oracles in the evaluation environment, so that a failure of a
# generated sample can be told from a defect of the task itself.
#
# Also produces the separate reference run of the three cwe_1333_0 variants that
# Section 3.7 reports for the replaced ReDoS checker.
#
# Runs INSIDE the co1lin/cweval container, against the repo mounted at /host/CWEval:
#   docker run -d --name cweval_check \
#     -v /path/to/CWEval:/host/CWEval \
#     co1lin/cweval bash /host/CWEval/benchmark_check.sh
set -u

# repo root inside the container; override with REPO=... if mounted elsewhere
REPO=${REPO:-/host/CWEval}
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate cweval
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
export NODE_PATH=$(npm root -g)
export C_INCLUDE_PATH="$CONDA_PREFIX/include"
export LIBRARY_PATH="$CONDA_PREFIX/lib"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
export PATH=$PATH:/usr/local/go/bin:$HOME/go/bin
export GOARCH=arm64 CGO_ENABLED=1

cd "$REPO"
export PYTHONPATH="$REPO"
mkdir -p evals/audit
LOG=evals/audit/benchmark_check.log

echo "compile refs: $(date)" > "$LOG"
python cweval/commons.py compile_all_in --path benchmark >> "$LOG" 2>&1
echo "compile exit: $?" >> "$LOG"
pytest benchmark -n 8 -q >> "$LOG" 2>&1
echo "pytest exit: $?" >> "$LOG"

# The three cwe_1333_0 variants are the only tasks whose security oracle calls
# an external checker, so Section 3.7 reports them separately. Run them on
# their own and add the checker verdicts for the two reference patterns.
echo "--- cwe_1333_0 reference tests ---" >> "$LOG"
pytest benchmark/core/py/cwe_1333_0_test.py \
       benchmark/core/js/cwe_1333_0_js_test.py \
       benchmark/core/cpp/cwe_1333_0_cpp_test.py -q >> "$LOG" 2>&1
echo "cwe_1333_0 exit: $?" >> "$LOG"
python tools/redos_oracle_check.py >> "$LOG" 2>&1

echo "done: $(date)" >> "$LOG"
