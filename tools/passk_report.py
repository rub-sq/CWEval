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
OPENWEIGHT = {'glm45', 'glm52', 'kimik27', 'kimik2think', 'minimaxm2', 'minimaxm3'}
# renamed 2026-08-27 for the current proprietary registry: gpt54->gpt56sol,
# gpt54mini->gpt56luna (sol/luna inferred from pricing tier - luna is the
# cheap one, matching "mini" - confirm if wrong), sonnet46->sonnet5,
# gemini3flash->gemini37flash. haiku45/gemini31pro unchanged. gemini31pro
# intentionally excluded here, matching the original list - see audit_report.py's
# TOKENS_ONLY for why (partial/aborted run).
MODELS_FULL = [
    'gpt56sol',
    'gpt56luna',
    'sonnet5',
    'haiku45',
    'gemini37flash',
    'glm45',
    'glm52',
    'kimik27',
    'kimik2think',
    'minimaxm2',
    'minimaxm3',
]
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
            # fewer graded samples than k would be scored as solved. Skip the
            # whole scope in that case rather than report a wrong value.
            short = min(len(v['functional']) for v in tasks.values())
            if short < k:
                print(f'  skip {model}/{scope_name} k={k}: min graded n = {short}')
                continue
            rows.append({
                'model': model,
                'scope': scope_name,
                'num_tasks': len(tasks),
                'k': k,
                'func_at_k': f'{rate(tasks, "functional", k):.2f}',
                'func_sec_at_k': f'{rate(tasks, "func_secure", k):.2f}',
            })
    return rows


def write_csv(path: str, rows: list) -> None:
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {path} ({len(rows)} rows)')


def main() -> None:
    all_rows = []
    for model in MODELS_FULL:
        all_rows += rows_for(model, load(model), SCOPES)
    write_csv(os.path.join('evals', 'passk_all_models.csv'), all_rows)


if __name__ == '__main__':
    main()
