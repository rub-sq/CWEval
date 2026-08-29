"""Export func@k / func-sec@k for the thesis as CSV.

Reads evals/eval_<model>/res_all.json (must exist, i.e. the evaluation
pipeline has already been run) and writes

  evals/passk_all_models.csv   - func@k and func-sec@k over all 119 tasks,
                                 for the fully evaluated models

Usage (from the CWEval repo root, no Docker needed):
  python3 tools/passk_report.py
"""

import csv
import json
import os
from math import comb

KS = [1, 10]
# the open-weight arm was sampled 100 times per task and therefore also
# supports k=50, the third setting of the CWEval paper. The proprietary arm
# was sampled 20 times and cannot support it.
KS_OPENWEIGHT = KS + [50]
# all 20 open-weight models of README.md, each evaluated at N=100 samples
# per task, which is what supports k=50 for all of them.
OPENWEIGHT = {
    'minimaxm2', 'minimaxm25', 'minimaxm3',
    'kimik2think', 'kimik25', 'kimik27',
    'glm45', 'glm47', 'glm52',
    'deepseekv3', 'deepseekv32', 'deepseekv4pro',
    'qwen3235b', 'qwen3coder480b', 'qwen35397b',
    'qwen330b', 'qwen3coder30b', 'qwen3527b',
    'deepseekv4flash', 'glm47flash',
}
# the five proprietary models plus all 20 open-weight models of README.md.
MODELS_FULL = [
    'gpt56sol',
    'gpt56luna',
    'gemini31pro',
    'haiku45',
    'gemini37flash',
] + sorted(OPENWEIGHT)
# Chapter 5 reads these metrics over all tasks only. The language breakdown of
# Sections 5.4 and 5.5 uses the insecure rate of breakdown_report.py instead.
SCOPES = [('all', '')]


def pass_at_k(n: int, c: int, k: int) -> float:
    # unbiased estimator, same as cweval.commons.pass_at_k
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def load(model: str) -> dict:
    path = os.path.join('evals', f'eval_{model}', 'res_all.json')
    with open(path) as f:
        return json.load(f)


def rate(tasks: dict, field: str, k: int) -> float:
    vals = [pass_at_k(len(v[field]), sum(v[field]), k) for v in tasks.values()]
    return sum(vals) / len(vals) * 100


def rows_for(model: str, res: dict, scopes) -> list:
    rows = []
    for scope_name, path_filter in scopes:
        tasks = {p: v for p, v in res.items() if path_filter in p}
        if not tasks:
            continue
        for k in KS_OPENWEIGHT if model in OPENWEIGHT else KS:
            # pass_at_k silently returns 1.0 once n - c < k, so a task with
            # fewer graded samples than k would be scored as solved if left
            # in. Exclude just the under-covered task(s) from this k's
            # average instead of dropping the whole model/scope over one of
            # them - a model can have just one or two such tasks out of 119,
            # with the rest carrying plenty of headroom, so computing k=50
            # from the rest is still meaningful. num_tasks
            # records how many tasks actually went into the average, so a
            # narrower denominator than the scope's full task count is
            # visible in the output, not silent.
            usable = {p: v for p, v in tasks.items() if len(v['functional']) >= k}
            excluded = len(tasks) - len(usable)
            if not usable:
                print(f'  skip {model}/{scope_name} k={k}: no task has {k}+ graded samples')
                continue
            if excluded:
                print(f'  {model}/{scope_name} k={k}: excluding {excluded} task(s) with < {k} graded samples')
            rows.append({
                'model': model,
                'scope': scope_name,
                'num_tasks': len(usable),
                'k': k,
                'func_at_k': f'{rate(usable, "functional", k):.2f}',
                'func_sec_at_k': f'{rate(usable, "func_secure", k):.2f}',
            })
    return rows


def write_csv(path: str, rows: list) -> None:
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {path} ({len(rows)} rows)')


def main() -> None:
    # skip models whose evaluation hasn't run yet rather than crash outright -
    # generation/evaluation across the two arms of this study finishes at
    # different times, so a partial MODELS_FULL list is normal.
    all_rows = []
    for model in MODELS_FULL:
        if not os.path.exists(os.path.join('evals', f'eval_{model}', 'res_all.json')):
            print(f'  skip {model}: no res_all.json yet')
            continue
        all_rows += rows_for(model, load(model), SCOPES)
    write_csv(os.path.join('evals', 'passk_all_models.csv'), all_rows)


if __name__ == '__main__':
    main()
