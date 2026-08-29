"""Data basis of the evaluation: everything Section 5.1 of the thesis reports.

Produces
  coverage.csv        raw, graded and dropped samples per model, the share
                      of graded samples that pass the security oracles without
                      passing the plausibility oracles, and the smallest number
                      of graded samples any single task of that model carries
  dropped_samples.csv one row per dropped sample with its category
  flakiness_models.csv changed verdicts per model between run A and run B
  flakiness_tasks.csv  changed verdicts per task, across models
  thresholds.csv       reporting threshold per metric

Reads evals/eval_*/ (run A) and evals/_run_B_*/ (run B). Read-only regarding
all inputs. Usage from the repo root, no docker needed:
  python3 tools/data_basis_report.py
"""

import ast
import csv
import json
import os
import re
import sys
import types
from collections import Counter, defaultdict
from glob import glob
from math import comb

# make cweval.commons importable on the host without its heavy deps
sys.path.insert(0, os.getcwd())
for _mod in ['fire', 'numpy', 'natsort', 'psutil']:
    try:
        __import__(_mod)
    except ImportError:
        _stub = types.ModuleType(_mod)
        if _mod == 'natsort':
            _stub.natsorted = sorted
        sys.modules[_mod] = _stub

from cweval.commons import get_code_blocks, strip_reasoning, select_code_block

# all 20 open-weight models of README.md - matches passk_report.py and
# breakdown_report.py, all fully evaluated and included in the
# coverage/flakiness/threshold reports.
OPENWEIGHT = {
    'minimaxm2', 'minimaxm25', 'minimaxm3',
    'kimik2think', 'kimik25', 'kimik27',
    'glm45', 'glm47', 'glm52',
    'deepseekv3', 'deepseekv32', 'deepseekv4pro',
    'qwen3235b', 'qwen3coder480b', 'qwen35397b',
    'qwen330b', 'qwen3coder30b', 'qwen3527b',
    'deepseekv4flash', 'glm47flash',
}
MODELS = sorted(OPENWEIGHT) + [
    'gpt56sol', 'gpt56luna', 'gemini31pro', 'haiku45', 'gemini37flash',
]
CURRENT_GEN = ['gpt56sol', 'gpt56luna', 'gemini31pro', 'haiku45', 'gemini37flash',
               'glm52', 'kimik27', 'minimaxm3']
RUN_B = 'evals/_run_B_2026-08-06'
OUT_DIR = 'evals/data_basis'

# expected python entrypoints from the benchmark
PY_ENTRYPOINTS = {}
for _f in glob('benchmark/core/py/*_task.py'):
    _m = re.search(r'^def (\w+)', open(_f).read(), re.M)
    PY_ENTRYPOINTS[os.path.basename(_f).replace('_task.py', '')] = _m.group(1)


def rel_task(raw_path: str) -> str:
    return re.search(r'generated_\d+/(.+)_raw\.\w+$', raw_path).group(1)


def lang_of(task: str) -> str:
    return 'lang_c' if task.startswith('lang/') else task.split('/')[1]


def write_csv(path: str, rows: list) -> None:
    if not rows:
        rows = [{'empty': 'no rows'}]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {path} ({len(rows)} rows)')


def load_run(model, run_b=False):
    base = f'{RUN_B}/eval_{model}' if run_b else f'evals/eval_{model}'
    return json.load(open(f'{base}/res_all.json'))


# ------------------------------------------------------- coverage + drops ---
def classify_drop(model: str, raw_path: str, task: str) -> dict:
    raw = open(raw_path, encoding='utf-8', errors='replace').read()
    stripped = strip_reasoning(raw)
    blocks = get_code_blocks(stripped)
    entry = PY_ENTRYPOINTS.get(os.path.basename(task), '')
    extracted = select_code_block(raw, entry) or raw

    category, note = '', ''
    if not raw.strip():
        category = 'empty_response'
    elif not blocks:
        category = 'no_code_block'
        note = stripped.strip()[:120].replace('\n', ' ')
    else:
        try:
            ast.parse(extracted)
            if entry and not re.search(rf'^\s*def {entry}\b', extracted, re.M):
                category = 'missing_entrypoint'
                note = f'expected def {entry}'
            else:
                category = 'unclear_graded_elsewhere'
        except SyntaxError as e:
            category = 'invalid_python'
            note = str(e)[:100]
    return {
        'model': model,
        'raw_path': raw_path,
        'lang': lang_of(task),
        'category': category,
        'note': note,
    }


def coverage():
    coverage_rows, dropped_rows = [], []
    for model in MODELS:
        gen_paths = sorted(
            glob(f'evals/eval_{model}/generated_*'),
            key=lambda p: int(p.rsplit('_', 1)[1]),
        )
        # needs each generation's own res.json (per-task verdicts, written by
        # evaluation - not the same file as the aggregate res_all.json).
        # Coverage can be partial within one model (a model mid-generation
        # can have res.json for its earlier generations but not for samples
        # added since) - filter to just the generations that have it, rather
        # than an all-or-nothing check on the model.
        gen_paths = [gp for gp in gen_paths if os.path.exists(os.path.join(gp, 'res.json'))]
        # also needs the aggregate res_all.json below (load_run) - a model
        # whose res_all.json predates its later generations is stale rather
        # than wrong, so exclude it here too rather than report on it
        # half-fresh.
        if not gen_paths or not os.path.exists(f'evals/eval_{model}/res_all.json'):
            print(f'  skip {model}: no per-generation res.json, or stale/missing res_all.json')
            continue
        n_raw = n_graded = 0
        by_lang, by_cat = Counter(), Counter()
        for gp in gen_paths:
            with open(os.path.join(gp, 'res.json')) as f:
                res_keys = set(json.load(f).keys())
            raws = [p for p in glob(f'{gp}/**/*_raw.*', recursive=True)
                    if '__pycache__' not in p]
            n_raw += len(raws)
            for raw in raws:
                if re.sub(r'_raw\.\w+$', '_test.py', raw) in res_keys:
                    n_graded += 1
                    continue
                task = rel_task(raw)
                by_lang[lang_of(task)] += 1
                row = classify_drop(model, raw, task)
                by_cat[row['category']] += 1
                dropped_rows.append(row)
        res = load_run(model)
        n_si = sum(1 for v in res.values()
                   for f, s in zip(v['functional'], v['secure'])
                   if s and not f)
        # the pass@k estimator is defined only where a task carries at least k
        # graded samples, and the implementation does not check it. The
        # smallest count per model is therefore the margin that a reported k
        # still has, which Chapter 7 reads under conclusion validity.
        min_n, min_task = min(
            (len(v['functional']), key.split('generated_X/')[-1])
            for key, v in res.items()
        )
        coverage_rows.append({
            'model': model,
            'samples_per_task': len(gen_paths),
            'raw_samples': n_raw,
            'graded': n_graded,
            'dropped': n_raw - n_graded,
            'dropped_pct': f'{(n_raw - n_graded) / n_raw * 100:.2f}',
            'secure_implausible_pct': f'{n_si / n_graded * 100:.2f}',
            'min_graded_per_task': min_n,
            'min_graded_task': min_task,
            'dropped_by_lang': dict(by_lang) or '',
            'dropped_by_category': dict(by_cat) or '',
        })
    write_csv(f'{OUT_DIR}/coverage.csv', coverage_rows)
    write_csv(f'{OUT_DIR}/dropped_samples.csv', dropped_rows)


# --------------------------------------------------- run-to-run variation ---
def flakiness():
    per_task = Counter()
    model_rows = []
    for model in MODELS:
        if not os.path.exists(f'{RUN_B}/eval_{model}/res_all.json'):
            continue
        a, b = load_run(model), load_run(model, run_b=True)
        flips = total = 0
        for key in a:
            x, y = a[key], b[key]
            if len(x['secure']) != len(y['secure']):
                continue
            n = sum(1 for p, q in zip(x['func_secure'], y['func_secure'])
                    if p != q)
            flips += n
            total += len(x['func_secure'])
            if n:
                per_task[key.split('generated_X/')[-1]] += n
        model_rows.append({
            'model': model,
            'samples_compared': total,
            'func_secure_flips': flips,
            'flip_rate_pct': f'{flips / total * 100:.2f}',
        })
    write_csv(f'{OUT_DIR}/flakiness_models.csv', model_rows)
    write_csv(f'{OUT_DIR}/flakiness_tasks.csv',
              [{'task': t, 'flipped_samples_across_models': c}
               for t, c in per_task.most_common()])


# ------------------------------------------------------------ thresholds ---
def passk(res, k, which):
    vals = []
    for v in res.values():
        f, sec = v['functional'], v['secure']
        n = len(f)
        if n < k:
            continue
        c = sum(f) if which == 'func' else sum(
            1 for a, b in zip(f, sec) if a and b)
        vals.append(1.0 if n - c < k else 1 - comb(n - c, k) / comb(n, k))
    return 100 * sum(vals) / len(vals)


def insecure_rate(res, tasks):
    """mean of the per-task share of samples failing a security oracle"""
    vals = []
    for key, v in res.items():
        if key.split('generated_X/')[-1] not in tasks:
            continue
        vals.append(100 * sum(1 for x in v['secure'] if not x)
                    / len(v['secure']))
    return sum(vals) / len(vals)


def thresholds():
    best = defaultdict(lambda: (0.0, '', ''))

    def note(metric, shift, model, where):
        if abs(shift) > best[metric][0]:
            best[metric] = (abs(shift), model, where)

    for model in MODELS:
        if not os.path.exists(f'{RUN_B}/eval_{model}/res_all.json'):
            continue
        a, b = load_run(model), load_run(model, run_b=True)
        for k in ([1, 10, 50] if model in OPENWEIGHT else [1, 10]):
            fa, sa = passk(a, k, 'func'), passk(a, k, 'fs')
            fb, sb = passk(b, k, 'func'), passk(b, k, 'fs')
            note('func@k', fa - fb, model, f'k={k}')
            note('func-sec@k', sa - sb, model, f'k={k}')
            note('Gap@k', (fa - sa) - (fb - sb), model, f'k={k}')
            note('SecRatio@k', (sa / fa - sb / fb) * 100, model, f'k={k}')

    # any model's task-key set works here - just enumerating the benchmark's
    # own tasks, not reading anything model-specific. glm45 is guaranteed to
    # have res_all.json (all 20 open-weight models do); gpt56sol may not yet.
    tasks = sorted({k.split('generated_X/')[-1]
                    for k in load_run('glm45')})
    langs, cwes = defaultdict(set), defaultdict(set)
    for t in tasks:
        parts = t.split('/')
        langs['lang-c' if parts[0] == 'lang' else parts[1]].add(t)
        cwes[re.search(r'cwe_(\d+)_', t).group(1)].add(t)
    cwes = {n: v for n, v in cwes.items() if len(v) >= 3}

    for model in CURRENT_GEN:
        if not os.path.exists(f'{RUN_B}/eval_{model}/res_all.json'):
            continue
        a, b = load_run(model), load_run(model, run_b=True)
        for name, group in langs.items():
            note('insecure rate (language)',
                 insecure_rate(a, group) - insecure_rate(b, group), model, name)
        for num, group in cwes.items():
            note('insecure rate (weakness type)',
                 insecure_rate(a, group) - insecure_rate(b, group),
                 model, f'CWE-{num}')

    write_csv(f'{OUT_DIR}/thresholds.csv',
              [{'metric': m, 'threshold_pp': f'{v[0]:.2f}',
                'largest_shift_model': v[1], 'largest_shift_at': v[2]}
               for m, v in best.items()])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    coverage()
    flakiness()
    thresholds()


if __name__ == '__main__':
    main()
