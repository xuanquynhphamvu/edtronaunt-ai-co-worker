from __future__ import annotations

import re


GLOBAL_FORBIDDEN_PATTERNS = {
    "bet": re.compile(r"\bbet(?:s|ting)?\b", re.IGNORECASE),
    "gamble": re.compile(r"\bgambl(?:e|es|ed|ing)\b", re.IGNORECASE),
    "emoji": re.compile(r"\bemojis?\b", re.IGNORECASE),
    "wager": re.compile(r"\bwager(?:s|ed|ing)?\b", re.IGNORECASE),
    "stake": re.compile(r"\bstakes?\b", re.IGNORECASE),
}


def find_forbidden_language(text: str) -> list[str]:
    return [
        f"Forbidden keyword detected: '{keyword}'"
        for keyword, pattern in GLOBAL_FORBIDDEN_PATTERNS.items()
        if pattern.search(text)
    ]
