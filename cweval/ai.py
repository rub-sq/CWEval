import abc
import os
from typing import Dict, List

import litellm

# litellm.set_verbose = True


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
