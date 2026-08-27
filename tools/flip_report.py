"""Export soft positive/negative flip rates between model generations as CSV.

For each (old, new) model pair the per-task secure rate r = c/n is computed
from the `secure` field of evals/eval_<model>/res_all.json (security oracles
only, independent of functionality). Per task:

  repair contribution     = (1 - r_old) * r_new
  regression contribution = r_old * (1 - r_new)

Averaged over tasks these give the soft positive flip rate (PFR) and soft
negative flip rate (NFR); adapted from the negative flip rate of
Yan et al. (CVPR 2021) to the sampling setting. Identity: PFR - NFR equals
the change of the mean secure rate (secure@1).

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

# Large-model stage progressions, per README.md's "Large" table: each
# family's gradual (1->2, 2->3) transitions plus the original direct 1->3
# jump, added 2026-08-27 at the user's request ("stage 1 to 2. 2 to 3").
STAGE_PAIRS = [
    ('minimaxm2', 'minimaxm25'), ('minimaxm25', 'minimaxm3'), ('minimaxm2', 'minimaxm3'),
    ('kimik2think', 'kimik25'), ('kimik25', 'kimik27'), ('kimik2think', 'kimik27'),
    ('glm45', 'glm47'), ('glm47', 'glm52'), ('glm45', 'glm52'),
    ('deepseekv2', 'deepseekv32'), ('deepseekv32', 'deepseekv4pro'), ('deepseekv2', 'deepseekv4pro'),
    ('qwen3235b', 'qwen3coder480b'), ('qwen3coder480b', 'qwen35397b'), ('qwen3235b', 'qwen35397b'),
]
# Small-vs-large size-matched sibling pairs at the same stage, per README.md's
# "Small" table ("every small model sibling against its own big brother").
SIBLING_PAIRS = [
    ('qwen330b', 'qwen3235b'),
    ('qwen3coder30b', 'qwen3coder480b'),
    ('qwen3527b', 'qwen35397b'),
    ('deepseekv2lite', 'deepseekv2'),
    ('glm47flash', 'glm47'),
]
_REAL_PAIRS = STAGE_PAIRS + SIBLING_PAIRS
# Literal self-comparison (the same model's res_all.json on both sides, not a
# split) for every model appearing in a real pair above - covers all 20
# open-weight models. Requested explicitly instead of relying on the
# noise_floor_old/new columns alone: mathematically identical (r_old == r_new
# exactly, so repair == regression == r*(1-r)) but reported as its own
# first-class row per pair/scope rather than a side column on someone else's.
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


def load_secure_rates(model: str) -> dict:
    path = os.path.join('evals', f'eval_{model}', 'res_all.json')
    with open(path) as f:
        res = json.load(f)
    rates = {}
    for key, fields in res.items():
        task = key.split('generated_X/')[-1]
        secure = fields['secure']
        rates[task] = (sum(secure), len(secure))
    return rates


def pair_rows(old: str, new: str) -> tuple:
    rates_old = load_secure_rates(old)
    rates_new = load_secure_rates(new)
    assert set(rates_old) == set(rates_new), f'task sets differ: {old} vs {new}'

    detail = []
    for task in sorted(rates_old):
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
        # identity check: net flip rate == change of mean secure rate
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


def main() -> None:
    all_rows, conc_rows = [], []
    for old, new in PAIRS:
        rows, conc = pair_rows(old, new)
        all_rows += rows
        conc_rows.append(conc)
    write_csv(os.path.join('evals', 'flip_report.csv'), all_rows)
    write_csv(os.path.join('evals', 'flip_concentration.csv'), conc_rows)


if __name__ == '__main__':
    main()
