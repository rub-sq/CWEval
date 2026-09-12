"""Export soft positive/negative flip rates between model generations as CSV.

For each (old, new) model pair the per-task rate r = c/n is computed from the
`func_secure` field of evals/eval_<model>/res_all.json (functional AND
secure - a sample only counts as a "success" here if it actually works, not
just if it happens to dodge the security oracle by failing to run at all).
CHANGED 2026-09-12: this used to be computed from `secure` alone, independent
of functionality. Switched to `func_secure` because a "secure" sample that
never runs correctly is not a meaningful security outcome - it can't be
exploited, but it also can't be shipped, so crediting it identically to a
genuinely safe working solution conflates capability with safety. Verified
empirically before switching: on all 5 proprietary pairs the aggregate
PFR/NFR/net changed by at most ~1.6 points versus the old `secure`-only
version, so this is a definitional cleanup, not a result-changing swap.

  repair contribution     = (1 - r_old) * r_new
  regression contribution = r_old * (1 - r_new)

Averaged over tasks these give the soft positive flip rate (PFR) and soft
negative flip rate (NFR); adapted from the negative flip rate of
Yan et al. (CVPR 2021) to the sampling setting. Identity: PFR - NFR equals
the change of the mean func-sec rate (func-sec@1).

The self-comparison noise floor mean(r * (1 - r)) is reported per model:
even comparing a model against itself yields flip rates of this size, so
absolute PFR/NFR values must be read against it.

Writes:
  evals/flip_report.csv         - PFR, NFR and the two noise floors, one row
                                  per pair and scope
  evals/flip_concentration.csv  - the share of each rate that the largest
                                  TOP_N tasks of a pair carry

Usage (from the CWEval repo root, no Docker needed):
  python3 tools/flip_report.py
"""

import csv
import json
import os

TOP_N = 5      # Section 5.3 reports the share the five largest tasks carry

# Flagship stage progressions, per README.md's "Large" table: each family's
# gradual (1->2, 2->3) transitions plus the direct 1->3 jump. A transition
# whose endpoint is a coding-specialized checkpoint (Kimi K2.7 Code at
# Moonshot stage 3, Qwen3-Coder-480B at Qwen stage 2) is a specialization
# comparison, not a version step: it is handled in SIBLING_PAIRS and omitted
# here, so those families contribute fewer than three version pairs.
STAGE_PAIRS = [
    ('minimaxm21', 'minimaxm25'), ('minimaxm25', 'minimaxm3'), ('minimaxm21', 'minimaxm3'),
    ('kimik2think', 'kimik25'),
    ('glm45', 'glm47'), ('glm47', 'glm52'), ('glm45', 'glm52'),
    ('deepseekv3', 'deepseekv32'), ('deepseekv32', 'deepseekv4pro'), ('deepseekv3', 'deepseekv4pro'),
    ('qwen3235b', 'qwen35397b'),
]
# Same-stage flagship-vs-compact size pairs, per README.md's "Small" table
# ("every small model sibling against its own big brother"). The coding
# checkpoints pair with the general-purpose checkpoint of the same size and
# organization closest to them, testing whether a code model is at least as
# secure as the general one beside it.
SIBLING_PAIRS = [
    ('qwen330b', 'qwen3235b'),
    ('qwen3coder30b', 'qwen3coder480b'),
    ('qwen3527b', 'qwen35397b'),
    ('deepseekv4flash', 'deepseekv4pro'),
    ('glm47flash', 'glm47'),
]
# old-generation proprietary baselines (CWEval paper Table I) against this
# study's frontier proprietary models, same generational-stage idea as
# STAGE_PAIRS above, one old->new pair per family (flagship->flagship,
# mini/fast->luna). See tools/passk_report.py's OLD_BASELINE for the same
# historical reference data.
PROPRIETARY_PAIRS = [
    ('gpt4o', 'gpt56sol'), ('gpt4omini', 'gpt56luna'),
    ('haiku35', 'haiku45'),
    ('gemini15pro', 'gemini31pro'), ('gemini15flash', 'gemini37flash'),
]
OLD_BASELINE = {
    'gpt4o': '../results/original_paper/eval_4o_t8',
    'gpt4omini': '../results/original_paper/eval_4omini_t8',
    'haiku35': '../results/original_paper/eval_haiku_t8',
    'gemini15pro': '../results/original_paper/eval_gpro_t8',
    'gemini15flash': '../results/original_paper/eval_gflash_t8',
}
_REAL_PAIRS = STAGE_PAIRS + SIBLING_PAIRS + PROPRIETARY_PAIRS
# Literal self-comparison (the same model's res_all.json on both sides, not a
# split) for every model appearing in a real pair above. Mathematically
# identical to reading noise_floor_old/new alone (r_old == r_new, so
# repair == regression == r*(1-r)), but reported as its own row.
_SELF_MODELS = sorted({m for pair in _REAL_PAIRS for m in pair})
PAIRS = _REAL_PAIRS + [(m, m) for m in _SELF_MODELS]
SCOPES = [
    ('all', ''),
    ('py', 'core/py/'),
    ('c', 'core/c/'),
    ('cpp', 'core/cpp/'),
    ('go', 'core/go/'),
    ('js', 'core/js/'),
    ('lang_c', 'lang/c'),
]


def res_path(model: str) -> str:
    if model in OLD_BASELINE:
        return os.path.join(OLD_BASELINE[model], 'res_all.json')
    return os.path.join('evals', f'eval_{model}', 'res_all.json')


def load_func_secure_rates(model: str) -> dict:
    with open(res_path(model)) as f:
        res = json.load(f)
    rates = {}
    for key, fields in res.items():
        task = key.split('generated_X/')[-1]
        func_secure = fields['func_secure']
        rates[task] = (sum(func_secure), len(func_secure))
    return rates


def pair_rows(old: str, new: str) -> tuple:
    rates_old = load_func_secure_rates(old)
    rates_new = load_func_secure_rates(new)
    # Old-generation baselines can be missing a task or two (e.g. haiku_t8 has
    # 118 of 119) - a pre-existing data characteristic, not a bug. Compute over
    # the intersection instead of crashing the whole report on one pair.
    tasks = set(rates_old) & set(rates_new)
    missing = (set(rates_old) | set(rates_new)) - tasks
    if missing:
        print(f'  {old}->{new}: {len(missing)} task(s) missing on one side, excluded')

    detail = []
    for task in sorted(tasks):
        c_o, n_o = rates_old[task]
        c_n, n_n = rates_new[task]
        r_o, r_n = c_o / n_o, c_n / n_n
        detail.append({
            'task': task,
            'n_old': n_o,
            'c_old': c_o,
            'r_old': r_o,
            'n_new': n_n,
            'c_new': c_n,
            'r_new': r_n,
            'repair': (1 - r_o) * r_n,
            'regression': r_o * (1 - r_n),
            'delta': r_n - r_o,
        })

    rows = []
    for scope_name, path_filter in SCOPES:
        tasks = [d for d in detail if path_filter in d['task']]
        if not tasks:
            continue
        num = len(tasks)
        mean_r_old = sum(d['r_old'] for d in tasks) / num
        mean_r_new = sum(d['r_new'] for d in tasks) / num
        pfr = sum(d['repair'] for d in tasks) / num
        nfr = sum(d['regression'] for d in tasks) / num
        noise_old = sum(d['r_old'] * (1 - d['r_old']) for d in tasks) / num
        noise_new = sum(d['r_new'] * (1 - d['r_new']) for d in tasks) / num
        # identity check: net flip rate == change of mean func-sec rate
        assert abs((pfr - nfr) - (mean_r_new - mean_r_old)) < 1e-6
        rows.append({
            'old': old,
            'new': new,
            'scope': scope_name,
            'soft_pfr': f'{pfr * 100:.2f}',
            'soft_nfr': f'{nfr * 100:.2f}',
            'noise_floor_old': f'{noise_old * 100:.2f}',
            'noise_floor_new': f'{noise_new * 100:.2f}',
        })

    # how the two rates spread over the single tasks. A rate is a mean over
    # the tasks, so the share one task contributes is its own term divided by
    # the sum of all terms.
    top = TOP_N
    def share(field):
        vals = sorted((d[field] for d in detail), reverse=True)
        return 100 * sum(vals[:top]) / sum(vals)

    conc = {
        'old': old,
        'new': new,
        'tasks': len(detail),
        'top_n': top,
        'even_share_pct': f'{100 * top / len(detail):.2f}',
        'top_n_of_nfr_pct': f'{share("regression"):.2f}',
        'top_n_of_pfr_pct': f'{share("repair"):.2f}',
    }
    return rows, conc


def write_csv(path: str, rows: list) -> None:
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {path} ({len(rows)} rows)')


def _has_data(model: str) -> bool:
    return os.path.exists(res_path(model))


def main() -> None:
    all_rows, conc_rows = [], []
    skipped = []
    for old, new in PAIRS:
        if not (_has_data(old) and _has_data(new)):
            skipped.append(f'{old}->{new}')
            continue
        rows, conc = pair_rows(old, new)
        all_rows += rows
        conc_rows.append(conc)
    if skipped:
        print(f'  skipping {len(skipped)} pair(s), no res_all.json yet for '
              f'one side: {", ".join(skipped)}')
    write_csv(os.path.join('evals', 'flip_report.csv'), all_rows)
    write_csv(os.path.join('evals', 'flip_concentration.csv'), conc_rows)


if __name__ == '__main__':
    main()
