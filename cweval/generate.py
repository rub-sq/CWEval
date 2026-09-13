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

import base64
import datetime
import glob
import json
import math
import os
import shutil
from typing import Any, Dict, List, Tuple

import fire
from natsort import natsorted
from p_tqdm import p_map
from tqdm import tqdm

from cweval.ai import AIAPI, BatchState, OpenAIBatch, OpenRouterBatch
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
        # When True (only meaningful together with batch=True), submits via
        # OpenAIBatch (real, first-party OpenAI Batch API) instead of
        # OpenRouterBatch - OpenRouter's batch endpoint currently rejects
        # every OpenAI-family model (see ai.py's OpenAIBatch docstring).
        # OpenRouter batch remains the default and still works for every
        # other provider (Anthropic, Google, ...) - this only swaps which
        # client is used, for this one call, when explicitly asked for.
        # Needs OPENAI_API_KEY in the environment; OPENROUTER_API_KEY is not
        # touched by this path at all.
        openai_direct: bool = False,
        # Splits the missing samples into this many separate batches instead
        # of one (only meaningful with batch=True). A provider can reserve
        # the FULL estimated cost of one big batch against your balance up
        # front, even though actual billing ends up far lower once the
        # cheap/short responses come back - if that reservation alone
        # exceeds what you have available, splitting into smaller batches
        # each reserves less. 1 (default) is today's unsplit behavior.
        batch_split: int = 1,
        # skips the "already exists, continue?" prompt below (used by
        # run_models.sh instead of piping 'y' into stdin)
        assume_yes: bool = False,
        # base64(json) form of extra_body (e.g. {"reasoning": {"max_tokens":
        # 2048}, "provider": {"order": ["openai"], "allow_fallbacks": false}}).
        # NOT a plain --extra_body '{"...": false}' flag: fire.Fire's CLI
        # parsing applies its own literal-eval-style coercion to string
        # arguments, and a JSON `false`/`true`/`null` is not a valid Python
        # literal (those are `False`/`True`/`None`) - confirmed live, this
        # silently turns a JSON boolean into the *string* 'false' instead of
        # the Python bool, which OpenRouter then rejects ("expected boolean,
        # received string") deep inside litellm with a confusing traceback.
        # Base64 makes the argument opaque to fire's parser (it no longer
        # looks like any Python literal), so it always arrives here as a
        # plain str for us to decode and json.loads ourselves, with correct
        # types guaranteed regardless of what's inside.
        extra_body_b64: str = '',
        **kwargs,
    ):
        self.model = model
        self.ppt = ppt
        self.num_proc = num_proc
        self.batch = batch
        self.openai_direct = openai_direct
        self.batch_split = max(1, batch_split)
        self.assume_yes = assume_yes
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
        if extra_body_b64:
            self.ai_kwargs['extra_body'] = json.loads(base64.b64decode(extra_body_b64))

        if not eval_path:
            self.eval_path = os.path.join(
                'evals', f'eval_{datetime.datetime.now().strftime("%y%m%d_%H%M%S")}'
            )
        else:
            # check if eval_path exists (nothing is ever deleted here either
            # way - answering 'y', or --assume_yes, just means "keep going
            # and fill in whatever samples are missing")
            if os.path.exists(eval_path):
                if self.assume_yes:
                    print(
                        f'{eval_path} already exists, continuing '
                        '(existing samples kept, only gaps filled).'
                    )
                elif (
                    input(
                        f'{eval_path} already exists, continue and fill in '
                        'missing samples? (y/n): '
                    )
                    .strip()
                    .lower()
                    != 'y'
                ):
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

        def write_one(index: int, resp: str, usage: Dict[str, Any]) -> None:
            if not resp:
                return
            out_path = case['out_path_template'].format(index=index)
            if os.path.exists(out_path):
                # Every call here requests a full fresh batch of num_samples
                # completions, so most indices usually already exist on a
                # resumed run. Without this check they'd be silently
                # overwritten with a brand new (different) sample: fine for a
                # truly fresh run where nothing exists yet, actively
                # destructive when pointing generate.py at an eval dir that
                # already has real data with just a few gaps in it (e.g. to
                # backfill specific missing samples) - this is exactly that
                # safeguard.
                return
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, 'w') as f:
                f.write(resp)
            # token-usage sidecar; the '_meta.' name contains neither '_raw.' nor '_task.'
            # so evaluate.py / pytest ignore it and the func@k pipeline is unaffected
            meta = {
                'model': ai,
                'lang': case['lang'],
                'task_file_path': case.get('task_file_path'),
                'sample_index': index,
                **(usage or {}),
            }
            meta_path = out_path.replace('_raw.', '_meta.') + '.json'
            with open(meta_path, 'w') as f:
                json.dump(meta, f)

        resps = prompt.req_ai(
            aiapi,
            case['lang'],
            case['code_prompt'],
            metadata={
                k: v for k, v in case.items() if k not in ['code_prompt', 'lang']
            },
            # Writes each sample to disk the moment it's ready (see
            # AIAPI.send_message) instead of only after every sample for
            # this task has returned - for a slow model with a large n,
            # waiting for the whole task can take hours before anything
            # becomes visible or resumable at all.
            on_sample=write_one,
        )
        # Safety net for anything on_sample missed (e.g. a provider path
        # that doesn't chunk requests down to size 1 the way local vLLM
        # does) - write_one already wrote everything in the normal case, so
        # this loop is a no-op then.
        for i, resp in enumerate(resps):
            write_one(i, resp, aiapi.usages[i] if i < len(aiapi.usages) else {})

    def _finish_one_batch(self, batcher, state: 'BatchState', prompt) -> Tuple[int, int]:
        """Polls an already-submitted (tracked) batch to completion and
        writes its results. Shared by every batch, split or not. Takes an
        already-constructed `batcher` (not a class) so submit and poll share
        one instance instead of building a second one just to poll."""
        batch_id, targets = state.load()

        def on_tick(status: str, _data: Dict[str, Any]) -> None:
            print(f'  batch {batch_id}: {status}', flush=True)

        try:
            final = batcher.poll_until_done(batch_id, on_tick=on_tick)
        except RuntimeError:
            # poll_until_done raises RuntimeError specifically for a terminal
            # failed/expired/cancelled status - that batch_id is dead and
            # will never complete. Clear the tracked state so the next
            # invocation submits a fresh batch instead of resuming (and
            # immediately re-failing on) this same dead one - without this,
            # every retry just re-polls the same batch and gets the same
            # terminal error forever, never actually retrying. A plain
            # TimeoutError (still in_progress past MAX_WAIT_S) is not caught
            # here: that batch may still complete on its own, so state stays
            # tracked and a later invocation resumes polling it rather than
            # submitting a wasteful duplicate.
            state.clear()
            raise
        results = batcher.parse_results(final)

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
        return written, failed

    def _gen_batch(self) -> None:
        # python cweval/generate.py gen --batch True --model openrouter/... --eval_path evals/eval_X
        # add --openai_direct True to submit via OpenAI's own Batch API
        # instead of OpenRouter's (needs OPENAI_API_KEY; see ai.py's
        # OpenAIBatch docstring for why this exists).
        # add --batch_split N to submit the missing samples as N separate
        # batches instead of one.
        prompt = make_prompt(self.ppt)
        num_samples = self.ai_kwargs.get('n', 1)

        if self.openai_direct:
            batcher_cls, label = OpenAIBatch, 'OpenAI'
        else:
            batcher_cls, label = OpenRouterBatch, 'OpenRouter'

        n_parts = self.batch_split

        # A batch from a split run that got interrupted mid-flight leaves
        # its numbered state file behind, uncleared. Finish those first -
        # before recomputing what's still missing below - so an interrupted
        # chunk is never silently orphaned and resubmitted as a duplicate.
        if n_parts > 1:
            for path in sorted(glob.glob(os.path.join(self.eval_path, '.batch_state_*.json'))):
                print(f'Finishing a batch left over from a previous run: {path}', flush=True)
                self._finish_one_batch(batcher_cls(self.model, **self.ai_kwargs), BatchState(path), prompt)

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

        chunk_size = math.ceil(len(entries) / n_parts)
        chunks = [entries[i:i + chunk_size] for i in range(0, len(entries), chunk_size)]

        total_written = total_failed = 0
        for idx, chunk in enumerate(chunks):
            chunk_targets = {cid: targets[cid] for cid, _ in chunk}
            state_path = (os.path.join(self.eval_path, '.batch_state.json') if n_parts == 1
                          else os.path.join(self.eval_path, f'.batch_state_{idx}.json'))
            state = BatchState(state_path)
            batcher = batcher_cls(self.model, **self.ai_kwargs)
            part_label = f' (part {idx + 1}/{len(chunks)})' if n_parts > 1 else ''

            if not state.exists():
                print(f'Submitting {len(chunk)} requests as one {label} batch{part_label}...',
                      flush=True)
                batch_id = batcher.submit(chunk)
                state.save(batch_id, chunk_targets)
                print(f'Submitted. batch_id={batch_id} (tracked in {state.path})', flush=True)
            else:
                print(f'Resuming batch tracked in {state.path}', flush=True)

            written, failed = self._finish_one_batch(batcher, state, prompt)
            total_written += written
            total_failed += failed

        print(
            f'Batch done: {total_written} written, {total_failed} failed/empty '
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
