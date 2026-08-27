import abc
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import litellm
import requests

# litellm.set_verbose = True

# Silences litellm's "Provider List: ..." banner, printed on a transient
# provider-resolution race under concurrent num_proc workers (each retried
# call eventually succeeds regardless). Only suppresses the print - the
# retry/error behavior itself is unaffected.
litellm.suppress_debug_info = True


class AIAPI(abc.ABC):

    def __init__(
        self,
        model: str,
        **kwargs,
    ) -> None:
        self.model = model
        self.provider = litellm.get_llm_provider(model)[1]
        self.req_kwargs = kwargs
        # per-response token usage, filled by send_message; index-aligned with its return
        self.usages: List[Dict] = []

    @staticmethod
    def _read(obj, name):
        # litellm usage may be a pydantic object or a plain dict depending on provider
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj.get(name)
        return getattr(obj, name, None)

    @classmethod
    def _per_response_usage(cls, comp, n_this: int) -> List[Dict]:
        u = cls._read(comp, 'usage')
        completion_tokens = cls._read(u, 'completion_tokens')
        prompt_tokens = cls._read(u, 'prompt_tokens')
        details = cls._read(u, 'completion_tokens_details')
        reasoning_tokens = cls._read(details, 'reasoning_tokens')

        # Batched providers (openai/gemini, n_this > 1) report one summed usage for the
        # whole batch -> split it across samples so the per-model SUM (and thus average)
        # stays exact. For the OpenRouter path n_this == 1 -> exact per sample.
        def _split(v):
            if v is None:
                return [None] * n_this
            base, extra = divmod(int(v), n_this)
            return [base + (1 if j < extra else 0) for j in range(n_this)]

        comp_split = _split(completion_tokens)
        prompt_split = _split(prompt_tokens)
        reason_split = _split(reasoning_tokens)
        return [
            {
                'completion_tokens': comp_split[j],
                'prompt_tokens': prompt_split[j],
                'reasoning_tokens': reason_split[j],
            }
            for j in range(n_this)
        ]

    def send_message(self, messages: List[Dict[str, str]], **kwargs) -> List[str]:
        all_kwargs = self.req_kwargs.copy()
        all_kwargs.update(kwargs)

        if self.provider == ['gemini', 'vertex_ai'] and 'gemini' in self.model:
            all_kwargs['safety_settings'] = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                },
            ]

        n_samples = all_kwargs.pop('n', 1)
        max_n_per_req: int = {
            'openai': 128,
            'gemini': 8,
        }.get(self.provider, 1)

        resp: List[str] = []
        usages: List[Dict] = []
        for i, idx in enumerate(range(0, n_samples, max_n_per_req)):
            n_this = min(max_n_per_req, n_samples - i * max_n_per_req)
            if n_this > 1:
                all_kwargs['n'] = n_this
            else:
                all_kwargs.pop('n', 1)

            resp_this = [''] * n_this
            comp = None
            for attempt in range(4):
                comp = litellm.completion(
                    model=self.model,
                    messages=messages,
                    num_retries=3,
                    **all_kwargs,
                )
                resp_this = [c.message.content or '' for c in comp.choices]
                if all(resp_this):
                    break
                for c in comp.choices:
                    if not (c.message.content or ''):
                        print(
                            f'  [warn] empty content: finish_reason={c.finish_reason}, '
                            f'usage={getattr(comp, "usage", None)}',
                            flush=True,
                        )
                if attempt < 3:
                    print(f'  [warn] retrying ({attempt + 1}/4)...', flush=True)
            assert len(resp_this) == n_this, f'{resp_this = } != {n_this = }'
            resp.extend(resp_this)
            # usage from the last attempt (matches the stored resp_this), one dict per sample
            usages.extend(self._per_response_usage(comp, n_this))

        # index-aligned with `resp`; consumed by generate.py to write token sidecars
        self.usages = usages
        return resp


class OpenRouterBatch:
    """Direct OpenRouter Batch API client (submit an array of requests, poll,
    get results at ~half price). litellm has NO support for this - its
    `create_batch` only implements OpenAI's own batch endpoint - so this
    bypasses litellm entirely for this one path; the synchronous AIAPI class
    above is untouched.

    Usage-dict shape returned by parse_results matches AIAPI._per_response_usage
    exactly ({completion_tokens, prompt_tokens, reasoning_tokens}), so callers
    (generate.py) write the identical meta.json sidecar regardless of which
    path produced a response.

    NOTE ON CONFIDENCE: submission (POST) and polling (GET status) are
    implemented directly from OpenRouter's published API docs
    (openrouter.ai/docs/batch-quickstart) and are solid. The exact shape of
    the RESULTS payload once status=="completed" is not fully documented
    there beyond "results are returned inline"; parse_results below is
    written defensively (tries a couple of plausible key names, modeled on
    OpenAI's batch output format, which OpenRouter's docs describe theirs as
    mirroring) but has not been verified against a real completed batch.
    If a live run's parsed results come back empty/wrong, this is the one
    function to inspect against the actual JSON - print batch_response to see
    its real shape.
    """

    BASE_URL = 'https://openrouter.ai/api/beta/batches'
    POLL_INTERVAL_S = 30
    MAX_WAIT_S = 26 * 3600  # a bit over the documented 24h completion window

    def __init__(self, model: str, **ai_kwargs) -> None:
        # AIAPI.model carries litellm's "openrouter/" routing prefix (e.g.
        # "openrouter/anthropic/claude-haiku-4.5"); OpenRouter's own API wants
        # its native slug without that prefix ("anthropic/claude-haiku-4.5").
        model = model[len('openrouter/') :] if model.startswith('openrouter/') else model
        # OpenRouter catalogs the batch-discounted rate as a DISTINCT model id
        # (e.g. "anthropic/claude-haiku-4.5:batch", confirmed via
        # /api/v1/models: exactly half the price of the plain id, and its
        # supported_parameters list drops "max_completion_tokens" - only
        # "max_tokens" is listed, which is already the field name used below).
        # Append the suffix so this actually gets billed at that rate.
        self.model = model if model.endswith(':batch') else f'{model}:batch'
        self.api_key = os.environ['OPENROUTER_API_KEY']
        self.ai_kwargs = ai_kwargs

    def _headers(self) -> Dict[str, str]:
        return {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }

    def _request_body(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        body: Dict[str, Any] = {'model': self.model, 'messages': messages}
        if 'temperature' in self.ai_kwargs:
            body['temperature'] = self.ai_kwargs['temperature']
        # OpenRouter's own chat completions field is `max_tokens`, not
        # litellm's `max_completion_tokens` alias - we're bypassing litellm's
        # translation layer here, so it has to be done explicitly.
        if 'max_completion_tokens' in self.ai_kwargs:
            body['max_tokens'] = self.ai_kwargs['max_completion_tokens']
        extra_body = self.ai_kwargs.get('extra_body')
        if extra_body:
            body.update(extra_body)  # e.g. {"reasoning": {"max_tokens": N}}
        return body

    def submit(self, entries: List[Tuple[str, List[Dict[str, str]]]]) -> str:
        """entries: list of (custom_id, messages). Returns the batch id."""
        payload = {
            'endpoint': '/v1/chat/completions',
            'model': self.model,
            'requests': [
                {'custom_id': cid, 'body': self._request_body(msgs)} for cid, msgs in entries
            ],
        }
        resp = requests.post(self.BASE_URL, headers=self._headers(), json=payload, timeout=120)
        if not resp.ok:
            # raise_for_status() alone drops OpenRouter's actual error body (e.g.
            # the specific reason behind a 402) - surface it instead of just the
            # bare status code.
            raise RuntimeError(
                f'Batch submit failed: {resp.status_code} {resp.reason} - {resp.text}'
            )
        data = resp.json()
        batch_id = data.get('id') or data.get('batch_id')
        if not batch_id:
            raise RuntimeError(f'Batch submit response had no id field: {data}')
        return batch_id

    def get_status(self, batch_id: str) -> Dict[str, Any]:
        resp = requests.get(f'{self.BASE_URL}/{batch_id}', headers=self._headers(), timeout=60)
        resp.raise_for_status()
        return resp.json()

    # Observed live (2026-08-27): a GET immediately after a successful
    # submit 404'd - the batch id isn't queryable the instant it's created.
    # Tolerate 404s as "not indexed yet" for a bounded grace window rather
    # than failing outright; past that window a 404 is treated as real.
    NOT_FOUND_GRACE_S = 300
    INITIAL_DELAY_S = 10  # before the first status check, not just between retries

    def poll_until_done(self, batch_id: str, on_tick=None) -> Dict[str, Any]:
        """Blocks (polling every POLL_INTERVAL_S) until the batch reaches a
        terminal state. Returns the final status response, which carries the
        results once status == 'completed'. Safe to call again after a
        process restart - polling is idempotent, no local state required."""
        start = time.time()
        time.sleep(self.INITIAL_DELAY_S)  # give the batch a moment to become queryable at all
        while True:
            try:
                data = self.get_status(batch_id)
            except requests.exceptions.HTTPError as e:
                not_found = e.response is not None and e.response.status_code == 404
                if not_found and time.time() - start < self.NOT_FOUND_GRACE_S:
                    if on_tick:
                        on_tick('not_found_yet', {})
                    time.sleep(self.POLL_INTERVAL_S)
                    continue
                raise
            status = data.get('status')
            if on_tick:
                on_tick(status, data)
            if status == 'completed':
                return data
            if status in ('failed', 'expired', 'cancelled'):
                raise RuntimeError(f'Batch {batch_id} ended with status={status}: {data}')
            if time.time() - start > self.MAX_WAIT_S:
                raise TimeoutError(
                    f'Batch {batch_id} still "{status}" after {self.MAX_WAIT_S}s - '
                    f'past the documented 24h window; check it manually.'
                )
            time.sleep(self.POLL_INTERVAL_S)

    @staticmethod
    def parse_results(batch_response: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """custom_id -> {'content': str|None, 'usage': dict|None, 'error': str|None}.
        See the class docstring: this is the one function to double-check
        against a real response, key names here are a best-effort guess.
        """
        out: Dict[str, Dict[str, Any]] = {}
        results = batch_response.get('results') or batch_response.get('output') or []
        for row in results:
            cid = row.get('custom_id')
            if cid is None:
                continue
            err = row.get('error')
            if err:
                out[cid] = {'content': None, 'usage': None, 'error': str(err)}
                continue
            body = (row.get('response') or {}).get('body') or row.get('body') or {}
            choices = body.get('choices') or []
            content = choices[0]['message']['content'] if choices else None
            usage = body.get('usage') or {}
            details = usage.get('completion_tokens_details') or {}
            out[cid] = {
                'content': content,
                'usage': {
                    'completion_tokens': usage.get('completion_tokens'),
                    'prompt_tokens': usage.get('prompt_tokens'),
                    'reasoning_tokens': details.get('reasoning_tokens'),
                },
                'error': None,
            }
        return out


class BatchState:
    """Persists {batch_id, custom_id -> target} to a file next to the eval
    output so a killed/disconnected process (real risk at up to 24h) can
    resume polling the SAME batch on restart instead of resubmitting - that
    would double-pay for whatever already ran. One state file per eval_path;
    deleted once results are written."""

    def __init__(self, path: str) -> None:
        self.path = path

    def exists(self) -> bool:
        return os.path.exists(self.path)

    def save(self, batch_id: str, targets: Dict[str, Dict[str, Any]]) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'w') as f:
            json.dump({'batch_id': batch_id, 'targets': targets}, f)

    def load(self) -> Tuple[str, Dict[str, Dict[str, Any]]]:
        with open(self.path) as f:
            d = json.load(f)
        return d['batch_id'], d['targets']

    def clear(self) -> None:
        if os.path.exists(self.path):
            os.remove(self.path)
