"""A deterministic fake model for testing the pipeline without GPUs or API keys."""

from __future__ import annotations

import hashlib
import math
import random
from typing import Dict, Optional

from .base import BaseModel, ModelOutput


class DummyModel(BaseModel):
    """Simulates a model with a fixed language prior that the image overrides with probability `visual_strength`.

    The prior answer for each prompt is a hash of the prompt, so the Blank answer is stable across runs.
    """

    supports_probs = True
    thread_safe = True

    def __init__(self, name: str = "dummy", visual_strength: float = 0.6, seed: int = 0, oracle_path: Optional[str] = None):
        super().__init__(name)
        self.visual_strength = visual_strength
        self.seed = seed
        # Optional {image_path: gt letter} oracle; without it, the image "points" to a hashed letter.
        self.oracle: Dict[str, str] = {}
        if oracle_path:
            import json

            with open(oracle_path, "r", encoding="utf-8") as f:
                self.oracle = json.load(f)

    @staticmethod
    def _h(*parts: str) -> int:
        return int(hashlib.sha256("||".join(parts).encode()).hexdigest()[:12], 16)

    def _softmax(self, scores: Dict[str, float]) -> Dict[str, float]:
        m = max(scores.values())
        exp = {k: math.exp(v - m) for k, v in scores.items()}
        z = sum(exp.values())
        return {k: v / z for k, v in exp.items()}

    def predict(self, prompt: str, image_path: Optional[str], options: Dict[str, str]) -> ModelOutput:
        letters = list(options)
        prior = letters[self._h(prompt) % len(letters)]
        scores = {k: 0.0 for k in letters}
        scores[prior] = 3.0
        if image_path is not None:
            rng = random.Random(self._h(prompt, image_path, str(self.seed)))
            visual = self.oracle.get(image_path) or letters[self._h(image_path) % len(letters)]
            strength = rng.uniform(0, 8) * self.visual_strength
            scores[visual] += strength
        probs = self._softmax(scores)
        letter = max(probs, key=probs.get)
        return ModelOutput(response=f"{letter}. {options[letter]}", option_probs=probs, prob_source="dummy")
