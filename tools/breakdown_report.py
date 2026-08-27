"""Breakdowns of the security verdict: what Sections 5.4 and 5.5 report.

The insecure rate of a model on a group of tasks is the mean over the tasks
of the group of the share of its graded samples that fail at least one
security oracle (Section 4.7 of the thesis).

Produces
  language_rates.csv   insecure rate per model and language group
  cwe_rates.csv        insecure rate per model and weakness type
  dead_variants.csv    task variants on which no sample of any current model
                       is plausible. Their insecure rate of 100 percent is
                       produced by the evaluation rather than by the generated
                       code, which Section 5.5 has to qualify before it reads
                       such a rate as a property of a model.

Reads evals/eval_*/res_all.json. Read-only. Usage from the repo root:
  python3 tools/breakdown_report.py
"""

import csv
import json
import os
import re
from collections import defaultdict

# each family's stage-1 -> stage-3 pair, so CURRENT_GEN below covers the
# latest stage of all 5 open-weight families (was missing DeepSeek/Qwen
# entirely - fixed 2026-08-27, see also tools/flip_report.py's fuller
# per-stage PAIRS for the PFR/NFR computation itself).
PAIRS = [
    ('glm45', 'glm52'), ('kimik2think', 'kimik27'), ('minimaxm2', 'minimaxm3'),
    ('deepseekv2', 'deepseekv4pro'), ('qwen3235b', 'qwen35397b'),
]
# renamed 2026-08-27 for the current proprietary registry: gpt54->gpt56sol,
# gpt54mini->gpt56luna (inferred from pricing - confirm if wrong),
# sonnet46->sonnet5, gemini3flash->gemini37flash. haiku45 unchanged.
FRONTIER_NEW = ['gpt56sol', 'gpt56luna', 'sonnet5', 'haiku45', 'gemini37flash']
MODELS = FRONTIER_NEW + [m for pair in PAIRS for m in pair]
CURRENT_GEN = FRONTIER_NEW + [new for _, new in PAIRS]
LANG_ORDER = ['all', 'py', 'c', 'cpp', 'go', 'js', 'lang-c']
MIN_TASKS = 3
OUT_DIR = 'evals/breakdowns'


def load(model):
    return json.load(open(f'evals/eval_{model}/res_all.json'))


def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'Wrote {path} ({len(rows)} rows)')


def groups(res):
    """the two partitions of the benchmark, as task-key sets"""
    langs, cwes = defaultdict(set), defaultdict(set)
    for key in res:
        task = key.split('generated_X/')[-1]
        parts = task.split('/')
        langs['all'].add(task)
        langs['lang-c' if parts[0] == 'lang' else parts[1]].add(task)
        cwes[int(re.search(r'cwe_(\d+)_', task).group(1))].add(task)
    return langs, {n: t for n, t in cwes.items() if len(t) >= MIN_TASKS}


def insecure_rate(res, tasks):
    per_task = []
    for key, v in res.items():
        if key.split('generated_X/')[-1] not in tasks:
            continue
        sec = v['secure']
        per_task.append(100 * sum(1 for x in sec if not x) / len(sec))
    return sum(per_task) / len(per_task)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    res_by_model = {m: load(m) for m in MODELS}
    langs, cwes = groups(res_by_model['gpt56sol'])

    lang_rows, cwe_rows = [], []
    for m in MODELS:
        res = res_by_model[m]
        for scope in LANG_ORDER:
            lang_rows.append({
                'model': m, 'scope': scope, 'tasks': len(langs[scope]),
                'insecure_pct': f'{insecure_rate(res, langs[scope]):.2f}',
            })
        for num in sorted(cwes):
            cwe_rows.append({
                'model': m, 'cwe': num, 'tasks': len(cwes[num]),
                'insecure_pct': f'{insecure_rate(res, cwes[num]):.2f}',
            })
    write_csv(f'{OUT_DIR}/language_rates.csv', lang_rows)
    write_csv(f'{OUT_DIR}/cwe_rates.csv', cwe_rows)
    write_csv(f'{OUT_DIR}/dead_variants.csv', dead_variants(res_by_model))


def dead_variants(res_by_model):
    """Task variants that never yield a plausible sample.

    A variant on which not one sample of any current model passes the
    plausibility oracles cannot be distinguishing between models. The usual
    cause is a test script that fails before it reaches a verdict, which marks
    the sample as not plausible and as insecure at once.
    """
    agg = defaultdict(lambda: [0, 0])
    for model in CURRENT_GEN:
        for key, v in res_by_model[model].items():
            task = key.split('generated_X/')[-1]
            agg[task][0] += sum(v['functional'])
            agg[task][1] += len(v['functional'])
    rows = []
    for task in sorted(agg):
        plausible, graded = agg[task]
        if plausible:
            continue
        cwe = int(re.search(r'cwe_(\d+)_', task).group(1))
        rows.append({
            'task': task,
            'cwe': cwe,
            'models': len(CURRENT_GEN),
            'graded_samples': graded,
            'plausible_samples': plausible,
        })
    return rows


if __name__ == '__main__':
    main()
