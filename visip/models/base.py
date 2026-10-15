"""Common interface for all model backends."""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from ..utils import logger


@dataclass
class ModelOutput:
    response: str
    # Distribution over option letters (sums to 1), or None if the backend cannot provide it.
    option_probs: Optional[Dict[str, float]] = None
    # How option_probs was obtained: "answer_token", "first_token", "api_top_logprobs", "dummy", or None.
    prob_source: Optional[str] = None
    # Letter parsed by the backend while locating the answer token (the runner re-parses anyway).
    parsed_letter: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


class BaseModel(abc.ABC):
    """A model answers one prompt, with or without an image.

    `image_path=None` means the Blank condition. Implementations must not add anything else to the
    prompt, so that the only difference between the two conditions is the image.
    """

    #: True if `predict` can return option probabilities (needed for VIG).
    supports_probs: bool = False
    #: True if `predict` may be called from several threads at once (API backends).
    thread_safe: bool = False

    def __init__(self, name: str):
        self.name = name

    @abc.abstractmethod
    def predict(self, prompt: str, image_path: Optional[str], options: Dict[str, str]) -> ModelOutput:
        """Answer `prompt`.

        Args:
            prompt: the full text prompt (already contains the options).
            image_path: image file for the Visual condition, or None for the Blank condition.
            options: ordered letter -> option text; used to read option probabilities and to parse.
        """
        ...

    def close(self) -> None:  # optional cleanup
        pass


def with_retries(fn: Callable[[], Any], max_retries: int = 5, base_delay: float = 2.0, max_delay: float = 60.0) -> Any:
    """Call `fn`, retrying with exponential backoff on any exception."""
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - SDKs raise many different error types
            attempt += 1
            if attempt > max_retries:
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            logger.warning("Request failed (%s: %s). Retry %d/%d in %.1fs", type(e).__name__, e, attempt, max_retries, delay)
            time.sleep(delay)
