"""Measured evaluation noise floor across N independent passes.

data_basis_report.py's flakiness_* compares exactly two hardcoded passes
(run A vs run B). This generalizes to any number of passes over the same
already-generated code, run via run_eval.sh with different OUT_DIR values,
and reports what fraction of graded samples get a different verdict on at
least one pass - the noise floor itself, not a single fixed threshold.

A sample is "unanimous" only if every pass agrees on all three verdict
fields (functional, secure, func_secure). Non-unanimous does not mean
"wrong" on any particular pass - it means the pipeline's own repeatability
is the limiting factor for that sample, independent of which model or code
produced it.

Produces
  evals/noise_floor.csv           non-unanimous rate per model (all samples)
  evals/noise_floor_samples.csv   the individual non-unanimous samples, for
                                  inspecting root cause (timeout, unstable
                                  parsing, environment-dependent test, ...)

Usage (from the CWEval repo root, no Docker needed):
  python3 tools/noise_floor_report.py evals/eval_ evals/_run_B_2026-09-08 evals/_run_C_2026-09-08 <model>...

  First argument group before the model names: one or more pass roots, in
  the same shape run_eval.sh's OUT_DIR produces - a directory holding
  eval_<model>/res_all.json per model.'evals/eval_' (i.e. the live tree)
  is a valid pass root too, expanded as 'evals/eval_<model>'.
  Then one or more model names to check.
"""

import csv
import json
import os
import sys

FIELDS = ('functional', 'secure', 'func_secure')


def res_path(pass_root: str, model: str) -> str:
    if pass_root.endswith('eval_'):
        return f'{pass_root}{model}/res_all.json'
    return os.path.join(pass_root, f'eval_{model}', 'res_all.json')


def load(pass_root: str, model: str) -> dict:
    with open(res_path(pass_root, model)) as f:
        return json.load(f)


def write_csv(path: str, rows: list) -> None:
    if not rows:
        print(f'  (no rows for {path}, not writing)')
        return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'Wrote {path} ({len(rows)} rows)')


def compare(pass_roots: list, model: str) -> tuple:
    per_pass = []
    for root in pass_roots:
        p = res_path(root, model)
        if not os.path.exists(p):
            print(f'  skip {model}: no res_all.json under {root}')
            return None
        per_pass.append(load(root, model))

    tasks = set(per_pass[0])
    for r in per_pass[1:]:
        tasks &= set(r)

    samples_rows = []
    total, non_unanimous = 0, 0
    for task in sorted(tasks):
        n = len(per_pass[0][task]['functional'])
        for i in range(n):
            verdicts = []
            for r in per_pass:
                v = tuple(r[task][f][i] for f in FIELDS if i < len(r[task][f]))
                verdicts.append(v)
            if len(set(len(v) for v in verdicts)) > 1 or len(verdicts[0]) < len(FIELDS):
                continue  # sample missing on some pass - not comparable
            total += 1
            if len(set(verdicts)) > 1:
                non_unanimous += 1
                samples_rows.append({
                    'model': model,
                    'task': task,
                    'sample_index': i,
                    **{f'pass_{j}_{f}': v[k] for j, v in enumerate(verdicts)
                       for k, f in enumerate(FIELDS)},
                })
    return total, non_unanimous, samples_rows


def main() -> None:
    args = sys.argv[1:]
    pass_roots = [a for a in args if '/' in a or a.endswith('eval_')]
    models = [a for a in args if a not in pass_roots]
    if len(pass_roots) < 2 or not models:
        print(__doc__)
        sys.exit(1)

    print(f'Comparing {len(pass_roots)} passes: {pass_roots}')
    summary_rows, all_samples = [], []
    for model in models:
        result = compare(pass_roots, model)
        if result is None:
            continue
        total, non_unanimous, samples = result
        pct = 100 * non_unanimous / total if total else 0.0
        summary_rows.append({
            'model': model, 'passes': len(pass_roots),
            'graded_samples': total, 'non_unanimous': non_unanimous,
            'non_unanimous_pct': f'{pct:.2f}',
        })
        all_samples += samples
        print(f'  {model}: {non_unanimous}/{total} non-unanimous ({pct:.2f}%)')

    write_csv('evals/noise_floor.csv', summary_rows)
    write_csv('evals/noise_floor_samples.csv', all_samples)


if __name__ == '__main__':
    main()
