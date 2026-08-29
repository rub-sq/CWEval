"""Verify that the reasoning-aware parser is equivalent to the original one
on responses without a reasoning section.

The code extraction of the evaluation phase was rebuilt on branch
fix/reasoning-parser (commit dc25e5e). The original parser took the first
Markdown code block and recognised a fence only at the very beginning of a
line; the new one strips reasoning sections, tolerates indented fences and
selects the last block defining the required entrypoint.

For models that emit no reasoning and a single code block both rules must
agree. This script checks that claim on the five proprietary models, which is the
group the fix was NOT written for, and it is the measurement Chapter 7
reports under internal validity: if the fix silently changed their extracted
code as well, the before/after comparison of the re-evaluation would be
confounded.

Method: the original parser is reproduced below verbatim from the state
before dc25e5e. Both parsers are applied to the stored raw responses, with
the surrounding logic of Evaler._get_code (entrypoint name read from the
reference task file, fallback to the full raw response when extraction
yields nothing) replicated so that two pipeline behaviours are compared and
not two isolated functions. The resulting strings are compared for exact
equality.

Usage (from the CWEval repo root, no Docker needed):
  python3 tools/check_parser_equivalence.py

Exits non-zero if any response differs.
"""

import glob
import os
import re
import sys
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cweval.commons import select_code_block  # new parser

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(REPO, 'benchmark')
EVALS = os.path.join(REPO, 'evals')

# the five proprietary models of this study. CAVEAT: gpt56sol, gpt56luna and
# gemini31pro are reasoning-capable under this study's uniform
# REASONING_MAX_TOKENS setting and do emit reasoning on most or all of their
# responses (see evals/audit/token_stats.csv) - this conflicts with the
# "none of which emits reasoning" premise stated in the module docstring
# above, and should be rechecked before trusting this script's result for
# those three.
FRONTIER = [
    'gpt56sol',
    'gpt56luna',
    'gemini31pro',
    'haiku45',
    'gemini37flash',
]


def get_code_from_original(msg: str, add_new_line: bool = False) -> str:
    """The `only_first=True` path of get_code_from as it stood before dc25e5e."""
    tail = '\n' if add_new_line else ''
    code_blocks: List[str] = []
    msg_lines = msg.splitlines()
    i_line = 0
    while i_line < len(msg_lines):
        line = msg_lines[i_line]
        if line.startswith('```'):
            code_lines = []
            i_line += 1
            while i_line < len(msg_lines):
                line = msg_lines[i_line]
                if line.startswith('```'):
                    break
                code_lines.append(line)
                i_line += 1
            code_blocks.append('\n'.join(code_lines) + tail)
            return code_blocks[0]
        i_line += 1
    return ''


def ref_task_file(raw_path: str) -> str:
    """Map evals/eval_<m>/generated_<i>/<rel>_raw.<ext> to its benchmark task file."""
    rel = raw_path.split('/generated_', 1)[1].split('/', 1)[1]
    return os.path.join(BENCH, rel.replace('_raw.', '_task.'))


def entrypoint_of(ref_path: str) -> str:
    """Entrypoint name as Evaler._get_code derives it (Python tasks only)."""
    if not ref_path.endswith('.py'):
        return ''
    with open(ref_path, 'r') as f:
        match = re.search(r'^def (\w+)', f.read(), re.MULTILINE)
    return match.group(1) if match else ''


def compare_model(model: str) -> Tuple[int, List[str]]:
    compared = 0
    differing: List[str] = []
    pattern = os.path.join(EVALS, f'eval_{model}', 'generated_*', '*', '*', '*_raw.*')
    for raw_path in sorted(glob.glob(pattern)):
        ref_path = ref_task_file(raw_path)
        if not os.path.exists(ref_path):
            raise FileNotFoundError(f'no reference task file for {raw_path}')
        with open(raw_path, 'r', encoding='utf8', errors='replace') as f:
            raw_str = f.read()

        original = get_code_from_original(raw_str) or raw_str
        new = select_code_block(raw_str, entrypoint_of(ref_path))
        if not new.strip():
            new = raw_str

        compared += 1
        if original != new:
            differing.append(raw_path)
    return compared, differing


def main() -> int:
    total = 0
    failed = False
    for model in FRONTIER:
        compared, differing = compare_model(model)
        total += compared
        print(f'{model:16s} compared={compared:6d} differing={len(differing):5d}')
        for path in differing[:5]:
            print(f'    {path}')
        failed |= bool(differing)
    print(f'{"total":16s} compared={total:6d}')
    if failed:
        print('FAILED: the two parsers disagree on at least one response')
        return 1
    print('OK: both parsers select the same code block for every response')
    return 0


if __name__ == '__main__':
    sys.exit(main())
