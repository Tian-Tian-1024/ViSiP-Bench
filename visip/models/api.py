"""Proprietary models through official APIs.

    OpenAIModel       - OpenAI Chat Completions (GPT-5.5, ...). With `base_url`, it also serves any
                        OpenAI-compatible endpoint (vLLM / SGLang / LMDeploy servers, gateways, proxies).
    AnthropicModel    - Anthropic Messages API (Claude Sonnet 4.6, ...).
    GeminiModel       - Google Gen AI SDK (Gemini-3-flash-preview, ...).

API keys are read from environment variables (never from config files):
    OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY (or GOOGLE_API_KEY);
    OpenAI-compatible endpoints use `api_key_env` (default OPENAI_API_KEY).

Closed models do not expose full token distributions, so VIG is not reported for them (as in the paper).
`OpenAIModel(request_logprobs=True)` can approximate option probabilities from top-20 logprobs for
endpoints that support it; these are marked `prob_source="api_top_logprobs"`.
"""

from __future__ import annotations

import base64
import math
import mimetypes
import os
from typing import Any, Dict, List, Optional

from ..parsing import parse_answer
from .base import BaseModel, ModelOutput, with_retries


def _encode_image(path: str) -> tuple[str, str]:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return mime, data


def _require_env(var: str) -> str:
    value = os.environ.get(var)
    if not value:
        raise EnvironmentError(f"Environment variable {var} is not set")
    return value


class _APIModel(BaseModel):
    thread_safe = True

    def __init__(self, name: str, model: str, max_tokens: int = 512, temperature: Optional[float] = None,
                 system_prompt: Optional[str] = None, max_retries: int = 5, timeout: float = 120.0,
                 extra_body: Optional[Dict[str, Any]] = None):
        super().__init__(name)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.max_retries = max_retries
        self.timeout = timeout
        self.extra_body = extra_body or {}


class OpenAIModel(_APIModel):
    """OpenAI Chat Completions, or any OpenAI-compatible server via `base_url`."""

    def __init__(self, name: str, model: str, base_url: Optional[str] = None, api_key_env: str = "OPENAI_API_KEY",
                 max_tokens_param: str = "max_completion_tokens", request_logprobs: bool = False,
                 image_detail: Optional[str] = None, **kwargs):
        super().__init__(name, model, **kwargs)
        from openai import OpenAI

        # Local servers often accept any key; fall back to a placeholder only when base_url is set.
        api_key = os.environ.get(api_key_env) or (None if base_url is None else "EMPTY")
        if api_key is None:
            _require_env(api_key_env)
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=self.timeout, max_retries=0)
        self.max_tokens_param = max_tokens_param
        self.request_logprobs = request_logprobs
        self.supports_probs = request_logprobs
        self.image_detail = image_detail

    def _messages(self, prompt: str, image_path: Optional[str]) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        if image_path is not None:
            mime, data = _encode_image(image_path)
            image_url: Dict[str, Any] = {"url": f"data:{mime};base64,{data}"}
            if self.image_detail:
                image_url["detail"] = self.image_detail
            content.append({"type": "image_url", "image_url": image_url})
        content.append({"type": "text", "text": prompt})
        msgs: List[Dict[str, Any]] = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs.append({"role": "user", "content": content})
        return msgs

    def _probs_from_logprobs(self, choice, response: str, options: Dict[str, str]):
        lp = getattr(choice, "logprobs", None)
        tokens = getattr(lp, "content", None) if lp is not None else None
        if not tokens:
            return None
        letters = list(options)
        parsed = parse_answer(response, letters, options)
        # Find the generated token that carries the answer letter (fallback: first token).
        target = tokens[0]
        if parsed.char_pos is not None:
            pos = 0
            for tok in tokens:
                pos += len(tok.token)
                if pos > parsed.char_pos:
                    target = tok
                    break
        mass = {k: 0.0 for k in letters}
        for cand in getattr(target, "top_logprobs", None) or []:
            key = cand.token.strip().strip("().*").upper()
            if key in mass:
                mass[key] += math.exp(cand.logprob)
        total = sum(mass.values())
        return {k: v / total for k, v in mass.items()} if total > 0 else None

    def predict(self, prompt: str, image_path: Optional[str], options: Dict[str, str]) -> ModelOutput:
        kwargs: Dict[str, Any] = {"model": self.model, "messages": self._messages(prompt, image_path),
                                  self.max_tokens_param: self.max_tokens}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.request_logprobs:
            kwargs.update(logprobs=True, top_logprobs=20)
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body

        resp = with_retries(lambda: self.client.chat.completions.create(**kwargs), self.max_retries)
        choice = resp.choices[0]
        text = (choice.message.content or "").strip()
        probs = self._probs_from_logprobs(choice, text, options) if self.request_logprobs else None
        usage = getattr(resp, "usage", None)
        return ModelOutput(
            response=text,
            option_probs=probs,
            prob_source="api_top_logprobs" if probs else None,
            meta={"finish_reason": choice.finish_reason,
                  "usage": usage.model_dump() if hasattr(usage, "model_dump") else None},
        )


class AnthropicModel(_APIModel):
    def __init__(self, name: str, model: str, api_key_env: str = "ANTHROPIC_API_KEY", **kwargs):
        super().__init__(name, model, **kwargs)
        import anthropic

        self.client = anthropic.Anthropic(api_key=_require_env(api_key_env), timeout=self.timeout, max_retries=0)

    def predict(self, prompt: str, image_path: Optional[str], options: Dict[str, str]) -> ModelOutput:
        content: List[Dict[str, Any]] = []
        if image_path is not None:
            mime, data = _encode_image(image_path)
            content.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}})
        content.append({"type": "text", "text": prompt})
        kwargs: Dict[str, Any] = {"model": self.model, "max_tokens": self.max_tokens,
                                  "messages": [{"role": "user", "content": content}]}
        if self.system_prompt:
            kwargs["system"] = self.system_prompt
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body

        resp = with_retries(lambda: self.client.messages.create(**kwargs), self.max_retries)
        text = "".join(block.text for block in resp.content if getattr(block, "type", None) == "text").strip()
        usage = getattr(resp, "usage", None)
        return ModelOutput(
            response=text,
            meta={"stop_reason": resp.stop_reason,
                  "usage": usage.model_dump() if hasattr(usage, "model_dump") else None},
        )


class GeminiModel(_APIModel):
    def __init__(self, name: str, model: str, api_key_env: Optional[str] = None, **kwargs):
        super().__init__(name, model, **kwargs)
        from google import genai

        env = api_key_env or ("GEMINI_API_KEY" if os.environ.get("GEMINI_API_KEY") else "GOOGLE_API_KEY")
        self.client = genai.Client(api_key=_require_env(env))

    def predict(self, prompt: str, image_path: Optional[str], options: Dict[str, str]) -> ModelOutput:
        from google.genai import types

        parts: List[Any] = []
        if image_path is not None:
            mime, _ = _encode_image(image_path)
            with open(image_path, "rb") as f:
                parts.append(types.Part.from_bytes(data=f.read(), mime_type=mime))
        parts.append(prompt)
        config_kwargs: Dict[str, Any] = {"max_output_tokens": self.max_tokens}
        if self.temperature is not None:
            config_kwargs["temperature"] = self.temperature
        if self.system_prompt:
            config_kwargs["system_instruction"] = self.system_prompt
        config_kwargs.update(self.extra_body)
        config = types.GenerateContentConfig(**config_kwargs)

        resp = with_retries(
            lambda: self.client.models.generate_content(model=self.model, contents=parts, config=config),
            self.max_retries,
        )
        text = (resp.text or "").strip()
        finish = None
        if getattr(resp, "candidates", None):
            finish = str(getattr(resp.candidates[0], "finish_reason", None))
        return ModelOutput(response=text, meta={"finish_reason": finish})
