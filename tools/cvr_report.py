"""Conditional Vulnerability Rate (CVR): canonical per-model, per-CWE and
per-language security metric, alongside func/func-sec/flip.

CVR(cwe) = Count(not secure AND functional) / Count(functional)

Unlike the plain secure rate, CVR is conditioned on functional correctness:
a sample that never runs correctly can't be exploited, but it also can't be
shipped, so crediting a "secure" verdict on it the same as a genuinely safe
working solution conflates capability with safety. CVR isolates the pure
security-competence question: given the model solves the task, how often is
the solution dangerous?

Produces
  evals/breakdowns/cwe_cvr.csv        CVR per model, all 31 CWEs (unfiltered -
                                      figure generation applies its own
                                      value-based cutoff for readability)
  evals/breakdowns/language_cvr.csv   CVR per model, all 6 language groups

Supersedes the old breakdown_report.py (language_rates.csv/cwe_rates.csv),
which reported the unconditioned insecure rate instead.

Reads evals/eval_*/res_all.json. Read-only. Usage from the repo root:
  python3 tools/cvr_report.py
"""

import csv
import json
import os
import re
from collections import defaultdict

# this study's own proprietary generations - the intermediate stage, then
# the current frontier stage, for OpenAI and Google.
FRONTIER_MID = ['gpt5', 'gpt5mini', 'gemini25pro', 'gemini25flash']
FRONTIER_NEW = ['gpt56sol', 'gpt56luna', 'gemini31pro', 'gemini37flash']
# old-generation proprietary baselines (CWEval paper Table I), kept outside
# evals/ since they're historical reference data, not this study's own runs.
OLD_BASELINE = {
    'gpt4o': '../results/original_paper/eval_4o_t8',
    'gpt4omini': '../results/original_paper/eval_4omini_t8',
    'gemini15pro': '../results/original_paper/eval_gpro_t8',
    'gemini15flash': '../results/original_paper/eval_gflash_t8',
}
# every open-weight model of README.md.
OPENWEIGHT_ALL = [
    'minimaxm21', 'minimaxm25', 'minimaxm3', 'kimik2think', 'kimik25', 'kimik27',
    'glm45', 'glm47', 'glm47flash', 'glm52', 'deepseekv3', 'deepseekv32', 'deepseekv4pro',
    'deepseekv4flash', 'qwen3235b', 'qwen330b', 'qwen3coder480b', 'qwen3coder30b',
    'qwen35397b', 'qwen3527b',
]
MODELS = FRONTIER_MID + FRONTIER_NEW + OPENWEIGHT_ALL + sorted(OLD_BASELINE)
LANG_ORDER = ['py', 'c', 'cpp', 'go', 'js', 'lang-c']
OUT_DIR = 'evals/breakdowns'


def res_path(model: str) -> str:
    if model in OLD_BASELINE:
        return os.path.join(OLD_BASELINE[model], 'res_all.json')
    return f'evals/eval_{model}/res_all.json'


def load(model: str) -> dict:
    return json.load(open(res_path(model)))


def write_csv(path: str, rows: list) -> None:
    if not rows:
        print(f'  skip {path}: nothing to write (0 rows)')
        return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'Wrote {path} ({len(rows)} rows)')


def task_meta(task: str):
    cwe = int(re.search(r'cwe_(\d+)_', task).group(1))
    parts = task.split('/')
    lang = 'lang-c' if parts[0] == 'lang' else parts[1]
    return cwe, lang


def cvr_by(res: dict, key_fn) -> dict:
    """key_fn(task) -> group key (a CWE int, or a language string). Returns
    {key: (cvr, n_functional_samples, n_tasks_in_group)}. A group with zero
    functional samples anywhere gets cvr forced to 0.0 - the ratio is
    undefined there, not a claim of security, matching this project's
    established convention for that case."""
    num, den = defaultdict(int), defaultdict(int)
    tasks_seen = defaultdict(set)
    for key, v in res.items():
        task = key.split('generated_X/')[-1]
        group = key_fn(task)
        tasks_seen[group].add(task)
        for func, sec in zip(v['functional'], v['secure']):
            if func:
                den[group] += 1
                if not sec:
                    num[group] += 1
    return {
        group: (num[group] / den[group] if den[group] else 0.0, den[group], len(tasks))
        for group, tasks in tasks_seen.items()
    }


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    # skip models whose evaluation hasn't run yet rather than crash outright.
    res_by_model = {}
    for m in MODELS:
        if not os.path.exists(res_path(m)):
            print(f'  skip {m}: no res_all.json yet')
            continue
        res_by_model[m] = load(m)

    all_cwes = sorted({
        task_meta(k.split('generated_X/')[-1])[0]
        for res in res_by_model.values() for k in res
    })

    cwe_rows, lang_rows = [], []
    for m, res in res_by_model.items():
        by_cwe = cvr_by(res, lambda t: task_meta(t)[0])
        for c in all_cwes:
            val, n_func, n_tasks = by_cwe.get(c, (0.0, 0, 0))
            cwe_rows.append({
                'model': m, 'cwe': c, 'tasks': n_tasks,
                'functional_samples': n_func, 'cvr': f'{val:.4f}',
            })
        by_lang = cvr_by(res, lambda t: task_meta(t)[1])
        for lg in LANG_ORDER:
            val, n_func, n_tasks = by_lang.get(lg, (0.0, 0, 0))
            lang_rows.append({
                'model': m, 'language': lg, 'tasks': n_tasks,
                'functional_samples': n_func, 'cvr': f'{val:.4f}',
            })

    write_csv(f'{OUT_DIR}/cwe_cvr.csv', cwe_rows)
    write_csv(f'{OUT_DIR}/language_cvr.csv', lang_rows)


if __name__ == '__main__':
    main()
