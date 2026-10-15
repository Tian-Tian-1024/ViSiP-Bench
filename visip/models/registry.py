"""Model presets and factory.

`--model` accepts either a preset name below or "<backend>:<model id or path>", e.g.

    --model qwen3-vl-8b
    --model hf:/path/to/my/finetuned-qwen3-vl
    --model openai:gpt-5.5
    --model anthropic:claude-sonnet-4-6
    --model gemini:gemini-3-flash-preview
    --model openai_compatible:Qwen/Qwen3-VL-8B-Instruct --model-args '{"base_url": "http://localhost:8000/v1"}'
    --model dummy

`--model-args` (JSON) overrides any constructor argument of the backend.

API model ids below follow the names used in the paper; check your provider's model list and override
with "<backend>:<id>" if an id differs.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

from .base import BaseModel

BACKENDS = ("hf", "openai", "openai_compatible", "anthropic", "gemini", "dummy")

_HF = {
    # InternVL3.5 (transformers-native checkpoints)
    "internvl3.5-2b": "OpenGVLab/InternVL3_5-2B-HF",
    "internvl3.5-8b": "OpenGVLab/InternVL3_5-8B-HF",
    "internvl3.5-14b": "OpenGVLab/InternVL3_5-14B-HF",
    "internvl3.5-38b": "OpenGVLab/InternVL3_5-38B-HF",
    # InternVL3
    "internvl3-2b": "OpenGVLab/InternVL3-2B-hf",
    "internvl3-14b": "OpenGVLab/InternVL3-14B-hf",
    "internvl3-38b": "OpenGVLab/InternVL3-38B-hf",
    "internvl3-78b": "OpenGVLab/InternVL3-78B-hf",
    # Qwen2.5-VL
    "qwen2.5-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2.5-vl-32b": "Qwen/Qwen2.5-VL-32B-Instruct",
    "qwen2.5-vl-72b": "Qwen/Qwen2.5-VL-72B-Instruct",
    # Qwen3-VL
    "qwen3-vl-2b": "Qwen/Qwen3-VL-2B-Instruct",
    "qwen3-vl-4b": "Qwen/Qwen3-VL-4B-Instruct",
    "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
    # Qwen3.5
    "qwen3.5-2b": "Qwen/Qwen3.5-2B",
    "qwen3.5-9b": "Qwen/Qwen3.5-9B",
    "qwen3.5-27b": "Qwen/Qwen3.5-27B",
    # LLaVA-OneVision
    "llava-onevision-qwen2-0.5b": "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    "llava-onevision-qwen2-7b": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    "llava-onevision-qwen2-72b": "llava-hf/llava-onevision-qwen2-72b-ov-hf",
}

PRESETS: Dict[str, Tuple[str, Dict[str, Any]]] = {name: ("hf", {"path": path}) for name, path in _HF.items()}
PRESETS.update(
    {
        "gpt-5.5": ("openai", {"model": "gpt-5.5"}),
        "claude-sonnet-4.6": ("anthropic", {"model": "claude-sonnet-4-6"}),
        "gemini-3-flash-preview": ("gemini", {"model": "gemini-3-flash-preview"}),
        "dummy": ("dummy", {}),
    }
)

# Qwen3.5 checkpoints are hybrid-thinking models; the paper evaluates direct answers.
for _name in ("qwen3.5-2b", "qwen3.5-9b", "qwen3.5-27b"):
    PRESETS[_name][1]["chat_template_kwargs"] = {"enable_thinking": False}


def resolve(spec: str) -> Tuple[str, str, Dict[str, Any]]:
    """Return (display name, backend, kwargs) for a --model spec."""
    key = spec.lower()
    if key in PRESETS:
        backend, kwargs = PRESETS[key]
        return key, backend, dict(kwargs)
    if ":" in spec:
        backend, target = spec.split(":", 1)
        backend = backend.lower()
        if backend not in BACKENDS:
            raise ValueError(f"Unknown backend {backend!r}; choose from {BACKENDS}")
        if backend == "hf":
            kwargs = {"path": target}
        elif backend == "dummy":
            kwargs = {}
        else:
            kwargs = {"model": target}
        name = target.rstrip("/").split("/")[-1]
        return name, backend, kwargs
    raise ValueError(
        f"Unknown model {spec!r}. Use a preset ({', '.join(sorted(PRESETS))}) or '<backend>:<id>' "
        f"with backend in {BACKENDS}."
    )


def build_model(spec: str, model_args: Optional[Dict[str, Any]] = None, name: Optional[str] = None) -> BaseModel:
    display, backend, kwargs = resolve(spec)
    kwargs.update(model_args or {})
    display = name or display

    if backend == "hf":
        from .hf import HFModel

        return HFModel(display, **kwargs)
    if backend in ("openai", "openai_compatible"):
        from .api import OpenAIModel

        if backend == "openai_compatible":
            if not kwargs.get("base_url"):
                raise ValueError('openai_compatible needs --model-args \'{"base_url": "http://host:port/v1"}\'')
            kwargs.setdefault("max_tokens_param", "max_tokens")
        return OpenAIModel(display, **kwargs)
    if backend == "anthropic":
        from .api import AnthropicModel

        return AnthropicModel(display, **kwargs)
    if backend == "gemini":
        from .api import GeminiModel

        return GeminiModel(display, **kwargs)
    if backend == "dummy":
        from .dummy import DummyModel

        return DummyModel(display, **kwargs)
    raise AssertionError(backend)


def parse_model_args(text: Optional[str]) -> Dict[str, Any]:
    if not text:
        return {}
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"--model-args must be a JSON object: {e}") from e
    if not isinstance(obj, dict):
        raise ValueError("--model-args must be a JSON object")
    return obj
