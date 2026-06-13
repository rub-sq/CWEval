"""Aggregate output-token usage recorded during generation.

The generation step writes one `*_meta.*.json` sidecar per sample next to each
`*_raw.*` file (see cweval/generate.py). One eval_path == one model, so this
script reports the per-model average output (completion) tokens per response,
overall and per language, plus prompt/reasoning context for the cost table.

    python tools/token_report.py --eval_path evals/test_sonnet46
"""

import json
import os
from collections import defaultdict
from typing import Dict, List

import fire
from natsort import natsorted

USAGE_KEYS = ('completion_tokens', 'prompt_tokens', 'reasoning_tokens')


def _iter_meta_files(eval_path: str) -> List[str]:
    meta_files: List[str] = []
    for root, _, files in os.walk(eval_path):
        if '__pycache__' in root:
            continue
        for file in natsorted(files):
            if '_meta.' in file and file.endswith('.json'):
                meta_files.append(os.path.join(root, file))
    return meta_files


def token_report(eval_path: str) -> None:
    meta_files = _iter_meta_files(eval_path)
    if not meta_files:
        print(f'No *_meta.*.json sidecars found under {eval_path!r}.')
        return

    sums: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    n_samples: Dict[str, int] = defaultdict(int)
    model = None

    for mf in meta_files:
        with open(mf, 'r') as f:
            meta = json.load(f)
        model = meta.get('model', model)
        lang = meta.get('lang', '?')
        for scope in ('all', lang):
            n_samples[scope] += 1
            for key in USAGE_KEYS:
                v = meta.get(key)
                if v is not None:
                    sums[scope][key] += int(v)
                    counts[scope][key] += 1

    def avg(scope: str, key: str) -> float:
        c = counts[scope][key]
        return sums[scope][key] / c if c else 0.0

    print('=' * 56)
    print(f'Token report: {eval_path}')
    print(f'model: {model}')
    print(f'samples with a sidecar: {n_samples["all"]}')
    print('=' * 56)
    print(f'{"scope":<8}{"n":>5}{"avg_out":>12}{"avg_prompt":>12}{"avg_reason":>12}')
    for scope in ['all'] + natsorted(k for k in n_samples if k != 'all'):
        print(
            f'{scope:<8}{n_samples[scope]:>5}'
            f'{avg(scope, "completion_tokens"):>12.1f}'
            f'{avg(scope, "prompt_tokens"):>12.1f}'
            f'{avg(scope, "reasoning_tokens"):>12.1f}'
        )
    print('=' * 56)


if __name__ == '__main__':
    fire.Fire(token_report)
