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
        # Self-hosted vLLM (hpc/gen_part_0X.slurm) resolves to the same
        # 'openai' provider tag as everything else on litellm's generic path,
        # so it would otherwise inherit the 128-per-request default below -
        # one request per task carrying all n=100 samples, with nothing
        # written to disk until that whole request returns. Chunking to one
        # completion per request here means each sample lands as soon as
        # it's done, so a slow model still makes resumable progress.
        if 'api_base' in all_kwargs and '127.0.0.1' in (all_kwargs.get('api_base') or ''):
            max_n_per_req = 1
        else:
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
        # CORRECTED (2026-09-12): OpenRouter's own documented example payload
        # (openrouter.ai/docs/batch-quickstart) uses the plain model slug with
        # NO ":batch" suffix in the request body - the discounted pricing is
        # selected by hitting the batch submission endpoint itself
        # (POST /api/beta/batches), not by a special model id. The
        # ":batch"-suffixed catalog entries (e.g. "openai/gpt-5:batch") exist
        # only as separate pricing-page listings; sending that literal string
        # as the `model` field is what produced every "does not have a
        # :batch endpoint" failure so far (sol, luna, and gpt-5 all failed
        # identically) - it was never a per-model provider gap. Strip any
        # ":batch" suffix a caller might still pass in, rather than adding one.
        self.model = model[: -len(':batch')] if model.endswith(':batch') else model
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

    # Transient-failure retry knobs for submit() below. OpenRouter's batch
    # endpoint has been observed to intermittently return a bare Cloudflare
    # 502 page or simply not respond within a normal window (a clean
    # ReadTimeout at exactly SUBMIT_TIMEOUT_S, not an actual hang - confirmed
    # 2026-09-12 by letting one run to completion instead of Ctrl+C'ing it
    # early) - both look like transient infra flakiness on their side, worth
    # retrying automatically. A 402 (balance) or the :batch-endpoint error are
    # NOT retried here: those are permanent for the current request and a
    # retry would just burn the same wait for the same outcome.
    SUBMIT_TIMEOUT_S = 120
    SUBMIT_MAX_ATTEMPTS = 4
    SUBMIT_BACKOFF_S = 20  # doubles each attempt: 20, 40, 80

    def submit(self, entries: List[Tuple[str, List[Dict[str, str]]]]) -> str:
        """entries: list of (custom_id, messages). Returns the batch id."""
        payload = {
            'endpoint': '/v1/chat/completions',
            'model': self.model,
            'requests': [
                {'custom_id': cid, 'body': self._request_body(msgs)} for cid, msgs in entries
            ],
        }
        backoff = self.SUBMIT_BACKOFF_S
        for attempt in range(1, self.SUBMIT_MAX_ATTEMPTS + 1):
            try:
                resp = requests.post(
                    self.BASE_URL, headers=self._headers(), json=payload,
                    timeout=self.SUBMIT_TIMEOUT_S,
                )
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                if attempt == self.SUBMIT_MAX_ATTEMPTS:
                    raise RuntimeError(
                        f'Batch submit failed after {attempt} attempts (network/timeout): {e}'
                    ) from e
                print(f'  batch submit attempt {attempt} failed ({e.__class__.__name__}), '
                      f'retrying in {backoff}s...')
                time.sleep(backoff)
                backoff *= 2
                continue
            if resp.ok:
                data = resp.json()
                batch_id = data.get('id') or data.get('batch_id')
                if not batch_id:
                    raise RuntimeError(f'Batch submit response had no id field: {data}')
                return batch_id
            # raise_for_status() alone drops OpenRouter's actual error body (e.g.
            # the specific reason behind a 402) - surface it instead of just the
            # bare status code.
            if 'does not have a :batch endpoint' in resp.text:
                # Historically this was misdiagnosed as a per-model provider
                # gap. Root cause (2026-09-12): a ":batch"-suffixed model id
                # was being sent in the request body itself; that suffix is
                # only a pricing-page catalog convention, not a value the
                # submission API accepts. self.model no longer carries the
                # suffix (see __init__), so seeing this again means something
                # else changed - print resp.text and check the payload shape
                # against openrouter.ai/docs/batch-quickstart before assuming
                # the model is unsupported.
                raise RuntimeError(
                    f"Batch submit failed for {self.model}: {resp.text}"
                )
            # A raw HTML error body (e.g. a Cloudflare 502 page instead of
            # JSON) is the same transient-infra symptom as the network
            # exceptions above, just surfaced as a 5xx status instead of a
            # socket error - retry it the same way.
            if 500 <= resp.status_code < 600 and attempt < self.SUBMIT_MAX_ATTEMPTS:
                print(f'  batch submit attempt {attempt} got {resp.status_code}, '
                      f'retrying in {backoff}s...')
                time.sleep(backoff)
                backoff *= 2
                continue
            raise RuntimeError(
                f'Batch submit failed: {resp.status_code} {resp.reason} - {resp.text}'
            )
        raise RuntimeError('Batch submit failed: exhausted retries without a definitive result')

    def get_status(self, batch_id: str) -> Dict[str, Any]:
        resp = requests.get(f'{self.BASE_URL}/{batch_id}', headers=self._headers(), timeout=60)
        resp.raise_for_status()
        return resp.json()

    # A GET immediately after a successful submit can 404 - the batch id
    # isn't queryable the instant it's created. Tolerate 404s as "not
    # indexed yet" for a bounded grace window rather than failing outright;
    # past that window a 404 is treated as real.
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


class OpenAIBatch:
    """Direct OpenAI Batch API client - bypasses OpenRouter entirely.

    WHY THIS EXISTS (2026-09-12): OpenRouter's batch endpoint rejects every
    OpenAI-family model tested so far (gpt-5, gpt-5-mini, and previously
    gpt-5.6-sol/gpt-5.6-luna) with "does not have a :batch endpoint" - this
    was chased down at length and is NOT a bug in how we call OpenRouter
    (confirmed against OpenRouter's own documented example, which fails
    identically). It looks like OpenRouter's batch pass-through simply isn't
    wired up for OpenAI yet - OpenAI's real Batch API works nothing like
    Anthropic's/Google's (inline JSON array): it requires uploading a JSONL
    *file*, creating a batch that references that file's id, and downloading
    a separate output file once done. This class talks to that real,
    first-party API directly for OpenAI models specifically, while
    OpenRouterBatch (above) remains untouched and still the path for every
    other provider - and still usable for OpenAI models too, if OpenRouter
    ever fixes this; nothing here removes that option.

    Returns the exact same per-sample shape as OpenRouterBatch.parse_results
    (custom_id -> {content, usage, error}), so generate.py's _gen_batch
    doesn't need to know or care which of the two it's using.
    """

    BASE_URL = 'https://api.openai.com/v1'
    POLL_INTERVAL_S = 30
    MAX_WAIT_S = 26 * 3600
    SUBMIT_TIMEOUT_S = 120
    SUBMIT_MAX_ATTEMPTS = 4
    SUBMIT_BACKOFF_S = 20

    def __init__(self, model: str, **ai_kwargs) -> None:
        # generate.py is invoked the same way as for the OpenRouter path
        # (--model openrouter/openai/<slug>, since that's litellm's routing
        # convention used everywhere else in this codebase) - strip both
        # litellm's "openrouter/" prefix and OpenRouter's "openai/" catalog
        # prefix, since OpenAI's own API wants the bare model id ("gpt-5",
        # not "openai/gpt-5" or "openrouter/openai/gpt-5").
        for prefix in ('openrouter/', 'openai/'):
            if model.startswith(prefix):
                model = model[len(prefix):]
        self.model = model
        self.api_key = os.environ['OPENAI_API_KEY']
        self.ai_kwargs = ai_kwargs

    def _headers(self, content_type: str = None) -> Dict[str, str]:
        h = {'Authorization': f'Bearer {self.api_key}'}
        if content_type:
            h['Content-Type'] = content_type
        return h

    # extra_body keys that only mean something in OpenRouter's unified
    # request schema and are rejected outright by OpenAI's own API:
    # 'reasoning' (OpenRouter's {"max_tokens": N} token-budget dict - OpenAI's
    # real equivalent is a string `reasoning_effort`, a different unit
    # entirely, not a drop-in rename) and 'provider' (OpenRouter's
    # multi-provider routing control - meaningless once you're calling
    # OpenAI directly, there's no routing to control). CONFIRMED live
    # (2026-09-13): sending 'reasoning' verbatim here 400s every single
    # request in the batch with "Unknown parameter: 'reasoning'".
    _OPENROUTER_ONLY_EXTRA_BODY_KEYS = ('reasoning', 'provider')

    def _request_body(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        # mirrors OpenRouterBatch._request_body above. temperature is always
        # forwarded when given - fixed at 0.8 across every model in this
        # study is a non-negotiable experimental control, not a per-model
        # choice this code gets to make. (An earlier version of this method
        # dropped it here based on OpenRouter's /v1/models metadata listing
        # no "temperature" in openai/gpt-5's supported_parameters - that was
        # inference, not a confirmed rejection like 'reasoning' got below,
        # and it should never have overridden a fixed experimental
        # parameter on that basis. Reverted. If OpenAI's real API does
        # reject it, that will surface as an explicit per-request error in
        # the batch's error file, the same way 'reasoning' did - a genuine
        # model constraint to surface and decide on, not something to work
        # around silently.)
        body: Dict[str, Any] = {'model': self.model, 'messages': messages}
        if 'temperature' in self.ai_kwargs:
            body['temperature'] = self.ai_kwargs['temperature']
        if 'max_completion_tokens' in self.ai_kwargs:
            body['max_tokens'] = self.ai_kwargs['max_completion_tokens']
        extra_body = self.ai_kwargs.get('extra_body')
        if extra_body:
            dropped = {k: v for k, v in extra_body.items()
                       if k in self._OPENROUTER_ONLY_EXTRA_BODY_KEYS}
            if dropped:
                print(f'  OpenAIBatch: dropping OpenRouter-only extra_body '
                      f'key(s) not valid on the direct OpenAI API: {dropped}')
            body.update({k: v for k, v in extra_body.items()
                         if k not in self._OPENROUTER_ONLY_EXTRA_BODY_KEYS})
        return body

    def _upload_input_file(self, entries: List[Tuple[str, List[Dict[str, str]]]]) -> str:
        lines = [
            json.dumps({
                'custom_id': cid,
                'method': 'POST',
                'url': '/v1/chat/completions',
                'body': self._request_body(msgs),
            })
            for cid, msgs in entries
        ]
        jsonl = '\n'.join(lines).encode('utf-8')
        resp = requests.post(
            f'{self.BASE_URL}/files',
            headers=self._headers(),
            files={'file': ('batch_input.jsonl', jsonl, 'application/jsonl')},
            data={'purpose': 'batch'},
            timeout=self.SUBMIT_TIMEOUT_S,
        )
        if not resp.ok:
            raise RuntimeError(f'OpenAI batch input file upload failed: '
                                f'{resp.status_code} {resp.text}')
        return resp.json()['id']

    def submit(self, entries: List[Tuple[str, List[Dict[str, str]]]]) -> str:
        """entries: list of (custom_id, messages). Returns the batch id."""
        input_file_id = self._upload_input_file(entries)
        payload = {
            'input_file_id': input_file_id,
            'endpoint': '/v1/chat/completions',
            'completion_window': '24h',
        }
        backoff = self.SUBMIT_BACKOFF_S
        for attempt in range(1, self.SUBMIT_MAX_ATTEMPTS + 1):
            try:
                resp = requests.post(
                    f'{self.BASE_URL}/batches',
                    headers=self._headers('application/json'),
                    json=payload, timeout=self.SUBMIT_TIMEOUT_S,
                )
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                if attempt == self.SUBMIT_MAX_ATTEMPTS:
                    raise RuntimeError(
                        f'OpenAI batch create failed after {attempt} attempts: {e}'
                    ) from e
                print(f'  batch create attempt {attempt} failed '
                      f'({e.__class__.__name__}), retrying in {backoff}s...')
                time.sleep(backoff)
                backoff *= 2
                continue
            if resp.ok:
                return resp.json()['id']
            if 500 <= resp.status_code < 600 and attempt < self.SUBMIT_MAX_ATTEMPTS:
                print(f'  batch create attempt {attempt} got {resp.status_code}, '
                      f'retrying in {backoff}s...')
                time.sleep(backoff)
                backoff *= 2
                continue
            raise RuntimeError(f'OpenAI batch create failed: {resp.status_code} {resp.text}')
        raise RuntimeError('OpenAI batch create failed: exhausted retries without a definitive result')

    def get_status(self, batch_id: str) -> Dict[str, Any]:
        resp = requests.get(f'{self.BASE_URL}/batches/{batch_id}',
                             headers=self._headers(), timeout=60)
        resp.raise_for_status()
        return resp.json()

    NOT_FOUND_GRACE_S = 300
    INITIAL_DELAY_S = 10

    def poll_until_done(self, batch_id: str, on_tick=None) -> Dict[str, Any]:
        """Same polling contract as OpenRouterBatch.poll_until_done, but on
        reaching 'completed' it additionally downloads the output (and
        error) file content, since OpenAI's batch object only carries file
        ids, never inline results."""
        start = time.time()
        time.sleep(self.INITIAL_DELAY_S)
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
                return self._download_results(data)
            if status in ('failed', 'expired', 'cancelled'):
                raise RuntimeError(f'Batch {batch_id} ended with status={status}: {data}')
            if time.time() - start > self.MAX_WAIT_S:
                raise TimeoutError(
                    f'Batch {batch_id} still "{status}" after {self.MAX_WAIT_S}s - '
                    f'past the documented 24h window; check it manually.'
                )
            time.sleep(self.POLL_INTERVAL_S)

    def _download_results(self, final_status: Dict[str, Any]) -> Dict[str, Any]:
        rows: List[Dict[str, Any]] = []
        output_file_id = final_status.get('output_file_id')
        if output_file_id:
            resp = requests.get(f'{self.BASE_URL}/files/{output_file_id}/content',
                                 headers=self._headers(), timeout=120)
            resp.raise_for_status()
            rows += [json.loads(line) for line in resp.text.splitlines() if line.strip()]
        # error_file_id holds requests OpenAI rejected outright (bad params,
        # etc.) - surface those as errored rows too, so they show up in
        # generate.py's `failed` count instead of silently vanishing.
        error_file_id = final_status.get('error_file_id')
        if error_file_id:
            resp = requests.get(f'{self.BASE_URL}/files/{error_file_id}/content',
                                 headers=self._headers(), timeout=120)
            resp.raise_for_status()
            for line in resp.text.splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                row.setdefault('error', row.get('response') or 'unknown error')
                rows.append(row)
        return {'results': rows}

    @staticmethod
    def parse_results(batch_response: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        # OpenAI's real per-line output shape ({"custom_id", "response":
        # {"body": {...chat completion...}}, "error"}) is exactly what
        # OpenRouterBatch.parse_results already handles - reuse it rather
        # than duplicate the same parsing logic.
        return OpenRouterBatch.parse_results(batch_response)


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
