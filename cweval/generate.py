"""
Expected directory structure:

benchmark
├── core
│   ├── c
│   │   ├── cwe_022_0_c_task.c
│   └── py
│   |   ├── cwe_020_0_task.py
└── lang

evals
├── eval_241110_014704
│   ├── generated_0
│   │   ├── core
│   │   │   ├── c
│   │   │   │   ├── cwe_022_0_c_raw.c    <--- to generate
│   │   │   └── py
│   │   │       ├── cwe_020_0_raw.py
│   │   └── lang
│   └── generated_1
└── pytest.ini
"""

import datetime
import json
import os
import shutil
from typing import Any, Dict, List, Tuple

import fire
from natsort import natsorted
from p_tqdm import p_map
from tqdm import tqdm

from cweval.ai import AIAPI, BatchState, OpenRouterBatch
from cweval.commons import BENCHMARK_DIR, LANGS
from cweval.ppt import make_prompt


class Gener:

    begin_prompt_anchor = 'BEGIN PROMPT'
    begin_solution_anchor = 'BEGIN SOLUTION'

    def __init__(
        self,
        eval_path: str = '',
        model: str = 'gpt-4o-mini-2024-07-18',
        ppt: str = 'direct',
        num_proc: int = 8,
        langs: List[str] = LANGS,
        exclude_path: List[str] = [],
        include_path: List[str] = [],
        # AI parameters
        n: int = 20,
        max_completion_tokens: int = 32768,
        temperature: float = 0.8,
        # OpenRouter Batch API: submits every missing sample as one batch
        # (up to a 24h completion window) instead of num_proc parallel
        # synchronous calls, at ~half the per-token price. Writes the exact
        # same generated_N/*_raw.*/*_meta.*.json files either way - only
        # meant for model="openrouter/..." (litellm handles every other
        # provider fine synchronously; this bypasses litellm entirely and
        # only knows OpenRouter's batch endpoint).
        batch: bool = False,
        **kwargs,
    ):
        self.model = model
        self.ppt = ppt
        self.num_proc = num_proc
        self.batch = batch
        self.langs = langs
        self.exclude_path = exclude_path
        self.include_path = include_path
        print(f'Using langs: {self.langs}')
        self.ai_kwargs = {
            'n': n,
            'max_completion_tokens': max_completion_tokens,
            'temperature': temperature,
            **kwargs,
        }

        if not eval_path:
            self.eval_path = os.path.join(
                'evals', f'eval_{datetime.datetime.now().strftime("%y%m%d_%H%M%S")}'
            )
        else:
            # check if eval_path exists
            if os.path.exists(eval_path):
                flag = (
                    input(f'{eval_path} already exists, overwrite? (y/n): ')
                    .strip()
                    .lower()
                )
                if flag != 'y':
                    print(f'Exiting...')
                    exit(0)

            self.eval_path = eval_path

        self.cases = self._get_cases()

    def _get_cases(self) -> Dict[str, Dict[str, str]]:
        cases: Dict[str, str] = {}
        for root, _, files in os.walk(BENCHMARK_DIR):
            if '__pycache__' in root:
                continue
            for file in natsorted(files):
                file_wo_ext, ext = os.path.splitext(file)
                task_file_path = os.path.join(root, file)
                lang = ext[1:]
                # filtering
                if not (ext and file_wo_ext.endswith('_task')):
                    continue
                if lang not in self.langs:
                    continue
                if any(exclude in task_file_path for exclude in self.exclude_path):
                    continue
                if self.include_path and not any(
                    include in task_file_path for include in self.include_path
                ):
                    continue
                # gather code prompt
                with open(task_file_path, 'r') as f:
                    task_code = f.read()
                begin_solution_line_src = ''
                for line in task_code.splitlines():
                    if self.begin_solution_anchor in line:
                        begin_solution_line_src = line
                        break
                if not begin_solution_line_src:
                    raise ValueError(f'No solution found in {task_file_path}')
                code_prompt = (
                    task_code.split(self.begin_prompt_anchor)[-1]
                    .split(begin_solution_line_src)[0]
                    .strip()
                )

                rel_task_file_path = os.path.relpath(task_file_path, BENCHMARK_DIR)
                gen_file_path_template = os.path.join(
                    self.eval_path,
                    'generated_{index}',
                    rel_task_file_path.replace('_task', '_raw'),
                )

                cases[task_file_path] = {
                    'task_file_path': task_file_path,
                    'code_prompt': code_prompt,
                    'lang': lang,
                    'out_path_template': gen_file_path_template,
                }

        return cases

    @staticmethod
    def _gen_case(
        ai: str,
        ppt: str,
        case: Dict[str, str],
        ai_kwargs: Dict[str, Any],
        rank: int,
    ) -> None:
        num_samples = ai_kwargs.get('n', 1)
        for i in range(num_samples):
            out_path = case['out_path_template'].format(index=i)
            if not os.path.exists(out_path):
                break
        else:
            print(
                f'{case["out_path_template"]} already completed, skipping', flush=True
            )
            return

        aiapi = AIAPI(ai, **ai_kwargs)
        prompt = make_prompt(ppt)
        resps = prompt.req_ai(
            aiapi,
            case['lang'],
            case['code_prompt'],
            metadata={
                k: v for k, v in case.items() if k not in ['code_prompt', 'lang']
            },
        )
        for i, resp in enumerate(resps):
            if not resp:
                continue
            out_path = case['out_path_template'].format(index=i)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, 'w') as f:
                f.write(resp)
            # token-usage sidecar; the '_meta.' name contains neither '_raw.' nor '_task.'
            # so evaluate.py / pytest ignore it and the func@k pipeline is unaffected
            meta = {
                'model': ai,
                'lang': case['lang'],
                'task_file_path': case.get('task_file_path'),
                'sample_index': i,
                **(aiapi.usages[i] if i < len(aiapi.usages) else {}),
            }
            meta_path = out_path.replace('_raw.', '_meta.') + '.json'
            with open(meta_path, 'w') as f:
                json.dump(meta, f)

    def _gen_batch(self) -> None:
        # python cweval/generate.py gen --batch True --model openrouter/... --eval_path evals/eval_X
        prompt = make_prompt(self.ppt)
        num_samples = self.ai_kwargs.get('n', 1)

        # Same per-task skip semantics as _gen_case (only the missing
        # samples), just gathered up front into one request array instead of
        # decided inline per synchronous call.
        targets: Dict[str, Dict[str, Any]] = {}
        entries: List[Tuple[str, List[Dict[str, str]]]] = []
        next_id = 0
        for case in self.cases.values():
            for i in range(num_samples):
                out_path = case['out_path_template'].format(index=i)
                if os.path.exists(out_path):
                    continue
                cid = str(next_id)
                next_id += 1
                msgs = prompt.build_messages(case['lang'], case['code_prompt'])
                entries.append((cid, msgs))
                targets[cid] = {
                    'out_path': out_path,
                    'lang': case['lang'],
                    'task_file_path': case.get('task_file_path'),
                    'sample_index': i,
                    'prompt_text': msgs[-1]['content'],
                }

        if not entries:
            print('All samples already exist, nothing to batch.', flush=True)
            return

        state = BatchState(os.path.join(self.eval_path, '.batch_state.json'))
        batcher = OpenRouterBatch(self.model, **self.ai_kwargs)

        if state.exists():
            print(f'Resuming batch tracked in {state.path}', flush=True)
            batch_id, targets = state.load()
        else:
            print(f'Submitting {len(entries)} requests as one OpenRouter batch...', flush=True)
            batch_id = batcher.submit(entries)
            state.save(batch_id, targets)
            print(f'Submitted. batch_id={batch_id} (tracked in {state.path})', flush=True)

        def on_tick(status: str, _data: Dict[str, Any]) -> None:
            print(f'  batch {batch_id}: {status}', flush=True)

        final = batcher.poll_until_done(batch_id, on_tick=on_tick)
        results = OpenRouterBatch.parse_results(final)

        written = failed = 0
        for cid, target in targets.items():
            r = results.get(cid)
            if not r or r.get('error') or not r.get('content'):
                failed += 1
                continue
            resp = prompt.postprocess(target['prompt_text'], r['content'])
            out_path = target['out_path']
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, 'w') as f:
                f.write(resp)
            meta = {
                'model': self.model,
                'lang': target['lang'],
                'task_file_path': target['task_file_path'],
                'sample_index': target['sample_index'],
                **(r.get('usage') or {}),
            }
            meta_path = out_path.replace('_raw.', '_meta.') + '.json'
            with open(meta_path, 'w') as f:
                json.dump(meta, f)
            written += 1

        state.clear()
        print(
            f'Batch done: {written} written, {failed} failed/empty '
            f'(rerun with the same --eval_path to retry just the gap).',
            flush=True,
        )

    def gen(self) -> None:
        if self.batch:
            self._gen_batch()
            return
        p_map(
            self._gen_case,
            [self.model] * len(self.cases),
            [self.ppt] * len(self.cases),
            self.cases.values(),
            [self.ai_kwargs] * len(self.cases),
            range(len(self.cases)),  # workaround: index as rank
            num_cpus=self.num_proc,
        )


if __name__ == "__main__":
    fire.Fire(Gener)
