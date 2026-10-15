"""Open-source MLLMs through Hugging Face transformers.

One generic adapter covers every model that loads with `AutoModelForImageTextToText` + `AutoProcessor`
and has a chat template, which includes the families evaluated in the paper:

    Qwen/Qwen2.5-VL-*-Instruct, Qwen/Qwen3-VL-*-Instruct, Qwen/Qwen3.5-*
    OpenGVLab/InternVL3-*-hf, OpenGVLab/InternVL3_5-*-HF        (use the transformers-native "-hf" repos)
    llava-hf/llava-onevision-qwen2-*-ov-hf

Option probabilities (for VIG) follow the paper: generate the answer greedily, locate the generated token
that carries the chosen option letter, and renormalize the softmax at that position over the option
letters. If the letter cannot be located (e.g. the model answered with the option text), the first
generated token is used instead and `prob_source` is set to "first_token".
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ..parsing import parse_answer
from ..utils import logger
from .base import BaseModel, ModelOutput
from .probs import bare_letter_tokens, char_to_token_index, letter_variant_tokens, option_distribution


def _resolve_dtype(name: str):
    import torch

    table = {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float16": torch.float16, "fp16": torch.float16,
             "float32": torch.float32, "fp32": torch.float32}
    if name == "auto":
        return "auto"
    if name not in table:
        raise ValueError(f"Unknown dtype {name!r}; choose from auto/bfloat16/float16/float32")
    return table[name]


class HFModel(BaseModel):
    supports_probs = True
    thread_safe = False

    def __init__(
        self,
        name: str,
        path: str,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        attn_implementation: Optional[str] = None,
        trust_remote_code: bool = False,
        max_new_tokens: int = 64,
        do_sample: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        processor_kwargs: Optional[Dict[str, Any]] = None,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
    ):
        super().__init__(name)
        import torch
        import transformers
        from packaging import version
        from transformers import AutoProcessor

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.generation_kwargs = generation_kwargs or {}
        self.system_prompt = system_prompt

        load_kwargs: Dict[str, Any] = {"device_map": device_map, "trust_remote_code": trust_remote_code}
        # `torch_dtype` was renamed to `dtype` in transformers 4.56.
        dtype_key = "dtype" if version.parse(transformers.__version__) >= version.parse("4.56.0") else "torch_dtype"
        load_kwargs[dtype_key] = _resolve_dtype(dtype)
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation
        load_kwargs.update(model_kwargs or {})

        self.model = self._load_model(path, load_kwargs)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=trust_remote_code, **(processor_kwargs or {}))
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        if getattr(self.processor, "chat_template", None) is None and getattr(self.tokenizer, "chat_template", None) is None:
            raise ValueError(f"{path} has no chat template; ViSiP-Bench needs chat-formatted prompts.")
        self._bare_tokens_cache: Dict[tuple, Dict[str, List[int]]] = {}
        logger.info("Loaded %s (%s)", path, type(self.model).__name__)

    # ------------------------------------------------------------------ loading
    @staticmethod
    def _load_model(path: str, load_kwargs: Dict[str, Any]):
        import transformers

        errors = []
        for cls_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModelForCausalLM"):
            cls = getattr(transformers, cls_name, None)
            if cls is None:
                continue
            try:
                return cls.from_pretrained(path, **load_kwargs)
            except (ValueError, KeyError) as e:  # model type not registered for this auto class
                errors.append(f"{cls_name}: {e}")
        hint = ""
        if "internvl" in path.lower() and not path.lower().rstrip("/").endswith("-hf"):
            hint = "\nHint: use the transformers-native InternVL checkpoints, e.g. OpenGVLab/InternVL3-8B-hf."
        raise RuntimeError(f"Could not load {path}:\n  " + "\n  ".join(errors) + hint)

    # ------------------------------------------------------------------ inputs
    def _build_inputs(self, prompt: str, image_path: Optional[str]):
        from PIL import Image

        content: List[Dict[str, Any]] = []
        images = None
        if image_path is not None:
            content.append({"type": "image"})
            with Image.open(image_path) as im:
                images = [im.convert("RGB")]
        content.append({"type": "text", "text": prompt})
        messages: List[Dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})
        messages.append({"role": "user", "content": content})

        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, **self.chat_template_kwargs
        )
        if images is None:
            inputs = self.processor(text=[text], return_tensors="pt")
        else:
            inputs = self.processor(text=[text], images=images, return_tensors="pt")
        return inputs.to(self.model.device)

    # ------------------------------------------------------------------ probabilities
    def _token_to_id(self, raw: str) -> Optional[int]:
        tid = self.tokenizer.convert_tokens_to_ids(raw)
        if tid is None or isinstance(tid, list):
            return None
        if self.tokenizer.convert_ids_to_tokens(tid) != raw:
            return None
        return tid

    def _bare_tokens(self, letters: Sequence[str]) -> Dict[str, List[int]]:
        key = tuple(letters)
        if key not in self._bare_tokens_cache:
            self._bare_tokens_cache[key] = bare_letter_tokens(
                letters, lambda s: self.tokenizer.encode(s, add_special_tokens=False)
            )
        return self._bare_tokens_cache[key]

    def _decode(self, ids) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def _locate_answer_token(self, gen_ids: List[int], char_pos: int, letter: str) -> Optional[int]:
        prefix_lens: List[int] = []
        for i in range(len(gen_ids)):
            prefix_lens.append(len(self._decode(gen_ids[: i + 1])))
            if prefix_lens[-1] > char_pos:
                break
        idx = char_to_token_index(prefix_lens, char_pos)
        if idx is None:
            return None
        # Guard against small misalignments: prefer the nearest token whose text contains the letter.
        for cand in (idx, idx - 1, idx + 1):
            if 0 <= cand < len(gen_ids) and letter in self._decode([gen_ids[cand]]):
                return cand
        return None

    def _option_probs(self, gen_ids: List[int], step_logits, response: str, letters: Sequence[str], options_text=None):
        parsed = parse_answer(response, letters, options_text)
        letter_tokens = None
        step = None
        source = None
        if parsed.letter is not None and parsed.char_pos is not None:
            step = self._locate_answer_token(gen_ids, parsed.char_pos, parsed.letter)
            if step is not None:
                raw = self.tokenizer.convert_ids_to_tokens(gen_ids[step])
                letter_tokens = letter_variant_tokens(raw, parsed.letter, letters, self._token_to_id)
                if letter_tokens is None:
                    letter_tokens = self._bare_tokens(letters)
                source = "answer_token"
        if step is None:
            step = 0
            letter_tokens = self._bare_tokens(letters)
            source = "first_token"
        if step >= len(step_logits):
            return None, None, parsed, {}

        logprobs = self.torch.log_softmax(step_logits[step][0].float(), dim=-1)
        needed = {t for ids in letter_tokens.values() for t in ids}
        lp = {t: float(logprobs[t]) for t in needed}
        probs, mass = option_distribution(lp, letter_tokens)
        return probs, source, parsed, {"answer_step": step, "option_mass": mass}

    # ------------------------------------------------------------------ predict
    def predict(self, prompt: str, image_path: Optional[str], options: Dict[str, str]) -> ModelOutput:
        torch = self.torch
        letters = list(options)
        inputs = self._build_inputs(prompt, image_path)
        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            return_dict_in_generate=True,
            output_logits=True,  # raw logits, before repetition penalty / warpers
        )
        if self.do_sample:
            if self.temperature is not None:
                gen_kwargs["temperature"] = self.temperature
            if self.top_p is not None:
                gen_kwargs["top_p"] = self.top_p
        else:
            gen_kwargs.update(temperature=None, top_p=None, top_k=None)
        gen_kwargs.update(self.generation_kwargs)

        with torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kwargs)
        input_len = inputs["input_ids"].shape[1]
        gen_ids = out.sequences[0, input_len:].tolist()
        response = self._decode(gen_ids).strip()
        # strip() may remove leading spaces; parse on the unstripped text so positions line up with tokens
        raw_response = self._decode(gen_ids)

        probs, source, parsed, meta = self._option_probs(gen_ids, out.logits, raw_response, letters, options)
        meta["n_generated_tokens"] = len(gen_ids)
        return ModelOutput(
            response=response,
            option_probs=probs,
            prob_source=source,
            parsed_letter=parsed.letter,
            meta=meta,
        )

    def close(self) -> None:
        del self.model
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

