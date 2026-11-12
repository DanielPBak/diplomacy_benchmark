"""Model-neutral token counting for press budgets.

Each lab's tokenizer splits the same text differently, so a budget in a model's own
tokens would allow different amounts of text for different models. Press budgets use one
fixed count for everyone instead: one token per 4 characters, rounded up.
"""
from __future__ import annotations

import math

CHARS_PER_TOKEN = 4


def count_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)
