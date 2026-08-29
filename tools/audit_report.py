"""Findings that no other script of this thesis produces.

  format_compliance.csv  per model: visible reasoning sections and multiple
                         code blocks (Section 3.7), and the responses whose
                         code did not survive the token budget together with
                         their length in characters (Chapter 7, internal
                         validity).
  token_stats.csv        completion-token statistics per model, including the
                         longest response and the reasoning-token outlier
                         (Section 5.7).

Sample accounting, dropped samples, run-to-run variation and the reporting
thresholds are produced by data_basis_report.py and are not repeated here.

Reads evals/eval_*/. Read-only regarding all inputs. Usage from the repo
root, no docker needed:
  python3 tools/audit_report.py
"""

import csv
import json
import os
import re
import sys
import types
from collections import defaultdict
from glob import glob

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

from cweval.commons import REASONING_END_RE, get_code_blocks, strip_reasoning

# the five proprietary models plus all 20 open-weight models of README.md.
MODELS = [
    'minimaxm2', 'minimaxm25', 'minimaxm3',
    'kimik2think', 'kimik25', 'kimik27',
    'glm45', 'glm47', 'glm52',
    'deepseekv2', 'deepseekv32', 'deepseekv4pro',
    'qwen3235b', 'qwen3coder480b', 'qwen35397b',
    'qwen330b', 'qwen3coder30b', 'qwen3527b',
    'deepseekv2lite', 'glm47flash',
    'gpt56sol', 'gpt56luna', 'gemini31pro', 'haiku45', 'gemini37flash',
]
# Models to include in token_stats() but not format_compliance() (e.g. a
# model whose meta files exist but whose raw responses cannot be audited
# the same way). Empty when every evaluated model gets the full audit.
TOKENS_ONLY: list = []
AUDIT_DIR = 'evals/audit'


def write_csv(path: str, rows: list) -> None:
    if not rows:
        rows = [{'empty': 'no rows'}]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {path} ({len(rows)} rows)')


def batch_reported(meta_paths: list) -> bool:
    """True if the provider reported one usage record per request instead of
    one per response. cweval/ai.py then splits the batch total evenly over the
    samples of the request, so the per-sample counts within a task collapse
    onto base and base+1. The mean survives that split exactly, the median and
    the maximum do not, and are therefore left empty for such a model.
    """
    per_task = defaultdict(list)
    for mp in meta_paths:
        try:
            d = json.load(open(mp))
        except (json.JSONDecodeError, OSError):
            continue
        per_task[d.get('task_file_path')].append(d.get('completion_tokens') or 0)
    clean = 0
    for vals in per_task.values():
        if len(vals) < 3:
            continue
        uniq = sorted(set(vals))
        runs, cur = [], [uniq[0]]
        for a, b in zip(uniq, uniq[1:]):
            if b - a <= 1:
                cur.append(b)
            else:
                runs.append(cur)
                cur = [b]
        runs.append(cur)
        # a split batch shows at most two adjacent values per request
        if all(len(r) <= 2 for r in runs) and len(uniq) < len(vals) / 4:
            clean += 1
    return bool(per_task) and clean > 0.9 * len(per_task)


# ------------------------------------------------------- format compliance ---
def format_compliance():
    """Per model: how the responses are shaped, and which of them lost their
    code to the token budget.

    `truncated` is the union of the two failure modes, not their sum: a
    response can both lose every code block and end on an unterminated fence,
    and Chapter 7 counts such a response once. The character means separate
    the truncated responses from the rest, which is the basis for the
    statement that the affected responses are the longest of the run.
    """
    rows = []
    for model in MODELS:
        raws = [
            p for p in glob(f'evals/eval_{model}/generated_*/**/*_raw.*',
                            recursive=True)
            if '__pycache__' not in p
        ]
        n = len(raws)
        think = multi = noblock = unclosed = 0
        chars_trunc, chars_other = [], []
        for p in raws:
            raw = open(p, encoding='utf-8', errors='replace').read()
            if re.search(r'</think>|</mm:think>', raw, re.I):
                think += 1
            if len(re.findall(r'^\s*```', raw, re.M)) >= 4:
                multi += 1
            stripped = strip_reasoning(raw)
            lost = not get_code_blocks(stripped)
            open_fence = stripped.count('```') % 2 == 1
            noblock += lost
            unclosed += open_fence
            (chars_trunc if (lost or open_fence) else chars_other).append(len(raw))
        trunc = len(chars_trunc)

        def mean(vals):
            return f'{sum(vals) / len(vals):.0f}' if vals else ''

        rows.append({
            'model': model,
            'samples': n,
            'think_tag_pct': f'{think / n * 100:.1f}',
            'multi_block_pct': f'{multi / n * 100:.1f}',
            'no_code_block': noblock,
            'unclosed_final_block': unclosed,
            'truncated': trunc,
            'truncated_pct': f'{trunc / n * 100:.2f}',
            'mean_chars_truncated': mean(chars_trunc),
            'mean_chars_other': mean(chars_other),
        })
    write_csv(f'{AUDIT_DIR}/format_compliance.csv', rows)


# ------------------------------------------------------------ token stats ---
def _raw_path_for(meta_path: str) -> str:
    """evals/.../cwe_020_0_meta.py.json -> evals/.../cwe_020_0_raw.py

    Reverses the naming generate.py writes (meta_path =
    out_path.replace('_raw.', '_meta.') + '.json').
    """
    return meta_path[: -len('.json')].replace('_meta.', '_raw.')


def _estimate_reasoning_tokens(meta_path: str, completion_tokens: int):
    """Character-proportional estimate of reasoning tokens from a raw
    response's own <think>/<mm:think> boundary, for models whose provider
    reports no separate reasoning counter (all 20 open-weight models -
    vLLM's usage stats don't split completion into the two).

    Not a token count - a token COUNT requires that model's own tokenizer,
    which isn't available for most of these (weight caches are deleted
    once each model's generation finishes). Instead: find the same
    reasoning-end tag strip_reasoning() itself uses, take the fraction of
    the response's CHARACTERS before it, and apply that fraction to
    completion_tokens - the one number this project already has a real
    count for, from the generation provider. Returns None (not zero) if
    the raw file doesn't exist or carries no reasoning tag at all - a
    model that never reasons is different from one this estimate couldn't
    be computed for.
    """
    raw_path = _raw_path_for(meta_path)
    if not os.path.exists(raw_path):
        return None
    raw = open(raw_path, encoding='utf-8', errors='replace').read()
    if not raw:
        return None
    matches = list(REASONING_END_RE.finditer(raw))
    if not matches:
        return 0  # no reasoning tag found - this sample did not reason
    frac = matches[-1].end() / len(raw)
    return completion_tokens * frac


def token_stats():
    rows = []
    for model in MODELS + TOKENS_ONLY:
        metas = glob(f'evals/eval_{model}/generated_*/**/*_meta.*.json',
                     recursive=True)
        toks = []
        # reasoning tokens are billed as completion tokens but are reported
        # separately by the provider. An absent field means the provider
        # reports no such counter, which is not the same as a counter of zero.
        reas = []
        # character-proportional estimate (see _estimate_reasoning_tokens),
        # used only where the provider itself reports no counter at all.
        reas_est = []
        for mp in metas:
            try:
                d = json.load(open(mp))
            except (json.JSONDecodeError, OSError):
                continue
            c = d.get('completion_tokens') or 0
            toks.append(c)
            if d.get('reasoning_tokens') is not None:
                reas.append(d['reasoning_tokens'])
            else:
                est = _estimate_reasoning_tokens(mp, c)
                if est is not None:
                    reas_est.append(est)
        if not toks:
            continue
        toks.sort()
        batched = batch_reported(metas)
        # a provider that reports a reasoning counter may still leave it at
        # zero on almost every response. Section 5.7 reads the table as zero
        # for the proprietary models and names the single exception, so the
        # maximum and the number of non-zero responses are reported next to
        # the mean.
        nonzero = [r for r in reas if r]
        reported = bool(reas)
        reas_shown = reas if reported else reas_est
        nonzero_shown = [r for r in reas_shown if r]
        rows.append({
            'model': model,
            'meta_files': len(toks),
            'completion_tokens_mean': f'{sum(toks) / len(toks):.0f}',
            'completion_tokens_max': '' if batched else toks[-1],
            'reasoning_tokens_mean': f'{sum(reas_shown) / len(reas_shown):.0f}' if reas_shown else '',
            'reasoning_tokens_max': f'{max(reas_shown):.0f}' if reas_shown else '',
            'reasoning_tokens_nonzero': len(nonzero_shown) if reas_shown else '',
            'reasoning_source': 'reported' if reported else
                                 ('estimated' if reas_est else ''),
        })
    write_csv(f'{AUDIT_DIR}/token_stats.csv', rows)


def main():
    os.makedirs(AUDIT_DIR, exist_ok=True)
    format_compliance()
    token_stats()


if __name__ == '__main__':
    main()
