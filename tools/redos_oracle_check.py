"""Verification of the architecture-independent ReDoS oracle, Section 3.7.

The three variants of cwe_1333_0 are the only tasks whose security oracle
calls an external checker. This script establishes two things about them.

  1. How many reference tests the three variants carry, and how many of those
     depend on the checker. A test that never reaches the checker cannot be
     affected by replacing it.
  2. What the checker returns for the two reference patterns of each variant,
     invoked exactly the way the oracle invokes it, that is through the
     relative path ./third_party/recheck-linux-x64.

The secure reference pattern must be classified as safe and the insecure one
as vulnerable, otherwise the oracle cannot separate the two references and
the task is not measuring anything.

The wrapper shells out to `timeout`, which is part of GNU coreutils and is
present in the evaluation container but not on macOS. Where it is missing the
wrapper falls back to `unknown` for every pattern, so this script checks for
it first and says so rather than reporting a failure that is an artefact of
the host.

Writes evals/audit/redos_oracle.csv. Usage from the repo root:
  python3 tools/redos_oracle_check.py
"""

import csv
import os
import re
import shutil
import subprocess
import sys

CHECKER = './third_party/recheck-linux-x64'
OUT = 'evals/audit/redos_oracle.csv'
# Per variant: the test module, and where the two reference patterns live
# with the syntax that holds them. The secure pattern is always in the task
# file. The insecure one sits in the _unsafe file for the two compiled or
# interpreted variants, and inside the test module for Python, where the
# insecure reference is a function of the test itself.
BAD_RE = r"pattern = r'([^']*)'"
VARIANTS = [
    ('py',
     'benchmark/core/py/cwe_1333_0_test.py',
     ('benchmark/core/py/cwe_1333_0_task.py', BAD_RE),
     ('benchmark/core/py/cwe_1333_0_test.py', BAD_RE)),
    ('js',
     'benchmark/core/js/cwe_1333_0_js_test.py',
     ('benchmark/core/js/cwe_1333_0_js_task.js', r'const pattern = /(.*)/;'),
     ('benchmark/core/js/cwe_1333_0_js_unsafe.js', r'const pattern = /(.*)/;')),
    ('cpp',
     'benchmark/core/cpp/cwe_1333_0_cpp_test.py',
     ('benchmark/core/cpp/cwe_1333_0_cpp_task.cpp',
      r'std::string pattern = R"\((.*)\)";'),
     ('benchmark/core/cpp/cwe_1333_0_cpp_unsafe.cpp',
      r'std::string pattern = R"\((.*)\)";')),
]


def test_counts(test_path: str) -> tuple:
    """Total reference tests of a variant and the share reaching the checker.

    The test module parametrises three test functions over the two parameter
    lists. `test_check_email` runs over both lists, the two `_unsafe` tests
    over one each. Only the security parameters reach the checker.
    """
    src = open(test_path).read()
    func = len(re.findall(r'marks=pytest\.mark\.functionality', src))
    sec = len(re.findall(r'marks=pytest\.mark\.security', src))
    total = (func + sec) + func + sec
    checker_dependent = sec + sec
    return total, checker_dependent


def pattern_of(spec: tuple) -> str:
    path, needle = spec
    m = re.search(needle, open(path).read())
    return m.group(1) if m else ''


def status_of(pattern: str) -> str:
    """Run the checker the way the oracle does and read its Status line."""
    try:
        out = subprocess.run([CHECKER, f'/{pattern}/'], capture_output=True,
                             text=True, timeout=180).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        return f'error: {exc}'
    for line in out.splitlines():
        if line.startswith('Status'):
            return line.split(':', 1)[1].strip()
    return 'no verdict'


def main() -> int:
    if shutil.which('timeout') is None:
        print('timeout (GNU coreutils) is not on PATH. The wrapper reports '
              'unknown for every pattern without it, so the verdicts below '
              'would describe the host and not the checker. Run this inside '
              'the evaluation container.', file=sys.stderr)
        return 2

    rows, bad = [], 0
    total_tests = total_checker = 0
    for lang, test_path, secure_spec, insecure_spec in VARIANTS:
        total, dependent = test_counts(test_path)
        total_tests += total
        total_checker += dependent
        secure = pattern_of(secure_spec)
        insecure = pattern_of(insecure_spec)
        for kind, pattern, expected in (('secure', secure, 'safe'),
                                        ('insecure', insecure, 'vulnerable')):
            got = status_of(pattern) if pattern else 'pattern not found'
            ok = got == expected
            bad += not ok
            rows.append({'variant': lang, 'reference': kind,
                         'reference_tests': total, 'checker_dependent': dependent,
                         'pattern': pattern, 'expected': expected,
                         'status': got, 'ok': ok})

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'Wrote {OUT} ({len(rows)} rows)')
    print(f'reference tests over the three variants : {total_tests}')
    print(f'  of those depending on the checker     : {total_checker}')
    print(f'reference patterns classified as expected: {len(rows) - bad}/{len(rows)}')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
