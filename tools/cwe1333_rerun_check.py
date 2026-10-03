"""Helper of run_cwe1333_rerun.sh: re-evaluation of the three cwe_1333_0
variants with the fixed ReDoS oracle against the study repository's
results/ and raw_data/ (the parent of this CWEval checkout). Standard library
only, runs on the host.

  prepare <study_repo> <work> <eval_name>...
      builds the evaluation tree <work>/<eval_name>/ for run_eval.sh: the
      result files from results/<source>/<eval_name>/ (every res_all.json and
      generated_*/res.json, which must round-trip byte-exactly through
      json.dump so that the in-place update keeps every other entry
      byte-identical) and the generated code of the three tasks, extracted
      from raw_data/<source>/<eval_name>.zip
  verify <study_repo> <work> <pass_A> <pass_B> <pass_C> <eval_name>...
      after the three passes, against the untouched results/: (1) functional
      verdicts of the three tasks unchanged, (2) every entry of the other
      tasks byte-identical in every file, (3) pooled CWE-1333 CVR as recorded
      and per pass, (4) samples whose verdict differs between passes
  writeback <study_repo> <work> <pass_B> <pass_C> <eval_name>...
      copies the pass-A files (the live work tree) over results/<source>/
      <eval_name>/, and replaces only the three tasks' entries in the
      existing results/<source>/noise_passes/run_{B,C}/<eval_name>/res_all.json
"""
import json
import os
import sys
import zipfile
from collections import Counter

SOURCES = ('original_paper', 'openrouter_evals', 'hpc_evals')
TASKS = ('core/py/cwe_1333_0_test.py', 'core/js/cwe_1333_0_js_test.py', 'core/cpp/cwe_1333_0_cpp_test.py')
CODE = {'core/py/cwe_1333_0_test.py': 'core/py/cwe_1333_0_{}.py',
        'core/js/cwe_1333_0_js_test.py': 'core/js/cwe_1333_0_js_{}.js',
        'core/cpp/cwe_1333_0_cpp_test.py': 'core/cpp/cwe_1333_0_cpp_{}.cpp'}


def task_of(key):
    for t in TASKS:
        if key.endswith('/' + t):
            return t
    return None


def source_of(study, name):
    found = [s for s in SOURCES if os.path.isfile(os.path.join(study, 'results', s, name, 'res_all.json'))]
    if len(found) != 1:
        raise SystemExit(f'{name}: expected exactly one results/<source>/{name}/res_all.json, found {found}')
    return found[0]


def result_files(eval_dir):
    """[(relative path, indent)] of res_all.json and every generated_*/res.json."""
    gens = sorted((d for d in os.listdir(eval_dir) if d.startswith('generated_')),
                  key=lambda d: int(d.split('_')[1]))
    return [('res_all.json', 2)] + [(os.path.join(g, 'res.json'), 4) for g in gens
                                    if os.path.isfile(os.path.join(eval_dir, g, 'res.json'))]


def read(path):
    with open(path) as f:
        return f.read()


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(text)


def round_trips(text, indent):
    return json.dumps(json.loads(text), indent=indent) == text


def prepare(study, work, names):
    bad = 0
    for name in names:
        src = source_of(study, name)
        res_dir = os.path.join(study, 'results', src, name)
        problems, n_code = [], 0
        with zipfile.ZipFile(os.path.join(study, 'raw_data', src, f'{name}.zip')) as zf:
            members = set(zf.namelist())
            for rel, indent in result_files(res_dir):
                text = read(os.path.join(res_dir, rel))
                if not round_trips(text, indent):
                    problems.append(f'{rel} does not round-trip')
                found = {task_of(k) for k in json.loads(text)} - {None}
                if found != set(TASKS):
                    problems.append(f'{rel} lacks {sorted(set(TASKS) - found)}')
                write(os.path.join(work, name, rel), text)
                if rel == 'res_all.json':
                    continue
                g = os.path.dirname(rel)
                for t in found:
                    code = [f'{name}/{g}/{CODE[t].format(kind)}' for kind in ('raw', 'task')]
                    code = [c for c in code if c in members]
                    if not code:
                        problems.append(f'{g}: no generated code for {t}')
                    for c in code:
                        with zf.open(c) as f:
                            data = f.read()
                        dst = os.path.join(work, c)
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        with open(dst, 'wb') as f:
                            f.write(data)
                        n_code += 1
        n = len(result_files(res_dir)) - 1
        print(f'  {name:22} {src:17} {n} samples, {n_code} code files, '
              + ('ok' if not problems else 'PROBLEM: ' + '; '.join(problems[:3])))
        bad += bool(problems)
    return bad


def same_except_tasks(new_text, old_text, indent):
    """True if new equals old byte for byte once the three tasks' entries are
    put back to their old values."""
    new, old = json.loads(new_text), json.loads(old_text)
    if list(new) != list(old):
        return False
    restored = {k: (old[k] if task_of(k) else v) for k, v in new.items()}
    return json.dumps(restored, indent=indent) == old_text


def verdicts(dirs):
    """{(eval_name, generated_N, task): (functional, secure)}; dirs: {eval_name: eval dir}."""
    out = {}
    for name, d in dirs.items():
        for rel, _ in result_files(d)[1:]:
            for k, v in json.loads(read(os.path.join(d, rel))).items():
                if task_of(k):
                    out[(name, os.path.dirname(rel), task_of(k))] = (bool(v['functional']), bool(v['secure']))
    return out


def pooled(v, name=None):
    p = sum(1 for k, (f, s) in v.items() if f and (name is None or k[0] == name))
    b = sum(1 for k, (f, s) in v.items() if f and s and (name is None or k[0] == name))
    return p, b, ((p - b) / p if p else float('nan'))


def verify(study, work, passes, names):
    ok = True
    ref = {n: os.path.join(study, 'results', source_of(study, n), n) for n in names}
    trees = {'live (A)': {n: os.path.join(work, n) for n in names}}
    trees.update({P: {n: os.path.join(root, n) for n in names} for P, root in zip('ABC', passes)})

    print('== (2) every entry of the other 116 tasks byte-identical to results/')
    for label, dirs in trees.items():
        n = fails = 0
        for name in names:
            for rel, indent in result_files(ref[name]):
                new_p = os.path.join(dirs[name], rel)
                n += 1
                if not os.path.isfile(new_p) or not same_except_tasks(read(new_p), read(os.path.join(ref[name], rel)), indent):
                    fails += 1
                    if fails <= 5:
                        print(f'   DIFFERS: {new_p}')
        print(f'  {label:9}: {n - fails}/{n} files identical outside the three tasks')
        ok &= fails == 0
    diff = [n for n in names for rel, _ in result_files(ref[n])
            if read(os.path.join(trees['live (A)'][n], rel)) != read(os.path.join(trees['A'][n], rel))]
    print(f'  live tree == pass A archive: {"yes" if not diff else "NO"}')
    ok &= not diff

    rec = verdicts(ref)
    runs = {P: verdicts(trees[P]) for P in 'ABC'}
    print('== (1) functional verdicts of the three tasks unchanged vs. results/')
    for P, v in runs.items():
        changed = [k for k in rec if k not in v or v[k][0] != rec[k][0]]
        print(f'  pass {P}: {len(changed)} of {len(rec)} samples changed' + (f', e.g. {changed[:3]}' if changed else ''))
        ok &= not changed

    print('== (3) pooled CWE-1333 CVR over all models and the three tasks')
    print(f'  {"":10} {"p":>6} {"b":>6} {"CVR":>7}')
    for label, v in [('recorded', rec)] + [(f'pass {P}', runs[P]) for P in 'ABC']:
        p, b, c = pooled(v)
        print(f'  {label:10} {p:6} {b:6} {c:7.3f}')
    print('  per model (p / b recorded / b A / b B / b C):')
    for name in names:
        print(f'    {name:22} {pooled(rec, name)[0]:4} {pooled(rec, name)[1]:4} '
              + ' '.join(f'{pooled(runs[P], name)[1]:4}' for P in 'ABC'))

    print('== (4) samples whose verdict (functional, secure) differs')
    keys = sorted(rec)
    for a, b in [('A', 'B'), ('A', 'C'), ('B', 'C')]:
        print(f'  pass {a} vs pass {b}: {sum(runs[a].get(k) != runs[b].get(k) for k in keys)}')
    print(f'  not unanimous over A/B/C: {sum(len({runs[P].get(k) for P in "ABC"}) > 1 for k in keys)}')
    flips = Counter((rec[k][1], runs['A'][k][1]) for k in keys if k in runs['A'] and rec[k][1] != runs['A'][k][1])
    print(f'  recorded -> pass A secure flips: False->True {flips[(False, True)]}, True->False {flips[(True, False)]}')
    return ok


def writeback(study, work, noise_passes, names):
    n_files = n_noise = 0
    skipped = []
    for name in names:
        src = source_of(study, name)
        res_dir = os.path.join(study, 'results', src, name)
        for rel, indent in result_files(res_dir):
            new, old = read(os.path.join(work, name, rel)), read(os.path.join(res_dir, rel))
            if not same_except_tasks(new, old, indent):
                raise SystemExit(f'{name}/{rel}: differs outside the three tasks, not written back')
            if new != old:
                write(os.path.join(res_dir, rel), new)
                n_files += 1
        for P, root in zip('BC', noise_passes):
            target = os.path.join(study, 'results', src, 'noise_passes', f'run_{P}', name, 'res_all.json')
            if not os.path.isfile(target):
                skipped.append(f'{name} run_{P}')
                continue
            old_text = read(target)
            if not round_trips(old_text, 2):
                raise SystemExit(f'{target} does not round-trip, not written back')
            old, new = json.loads(old_text), json.loads(read(os.path.join(root, name, 'res_all.json')))
            for k in old:
                if task_of(k):
                    old[k] = new[k]
            write(target, json.dumps(old, indent=2))
            n_noise += 1
    print(f'  wrote {n_files} result files under results/ and {n_noise} noise-pass res_all.json files '
          f'(three cwe_1333_0 entries replaced in each)')
    if skipped:
        print(f'  no noise pass on record, left as is: {", ".join(skipped)}')


if __name__ == '__main__':
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == 'prepare':
        sys.exit(1 if prepare(args[0], args[1], args[2:]) else 0)
    elif cmd == 'verify':
        ok = verify(args[0], args[1], args[2:5], args[5:])
        print('VERIFY: ' + ('ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED (see above)'))
        sys.exit(0 if ok else 2)
    elif cmd == 'writeback':
        writeback(args[0], args[1], args[2:4], args[4:])
    else:
        sys.exit(f'unknown command {cmd}')
