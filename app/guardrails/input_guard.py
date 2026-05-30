# app/guardrails/input_guard.py
# DEFENDS AGAINST: prompt injection + jailbreak attempts
# METHOD: regex pattern matching — fast, no model needed, hard to bypass

import re

# Patterns that indicate prompt injection or jailbreak attempts
# These are phrases real attackers use to try to override system prompts
INJECTION_PATTERNS = [
    r"ignore (previous|prior|all|your) instructions",
    r"forget (your|all|previous) instructions",
    r"disregard (your|all|previous|the above)",
    r"you are now",
    r"pretend (you are|to be)",
    r"act as (if|though|an?)",
    r"roleplay as",
    r"you('re| are) (now |)an? (AI|assistant|bot|model) (with(out)?|that)",
    r"new (system |)prompt",
    r"(reveal|show|print|output|display|repeat|tell me) (your |the |)(system prompt|instructions|prompt)",
    r"what (are|were) your instructions",
    r"repeat (everything|the above|all|your context|verbatim)",
    r"(say|output|print|write) (everything|the above) (above|back|verbatim)",
    r"DAN",                        # "Do Anything Now" jailbreak
    r"jailbreak",
    r"bypass (your |all |)(restrictions|filters|safety|guidelines)",
    r"override (your |all |)(instructions|settings|rules)",
    r"(you have|you've) no (restrictions|limits|rules)",
    r"sudo",
    r"developer mode",
    r"training data",
    r"ignore ethics",
]

# Compile once at import time for speed
COMPILED_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS
]

class InputGuardResult:
    def __init__(self, is_safe: bool, reason: str = ""):
        self.is_safe = is_safe
        self.reason  = reason

def check_input(user_query: str) -> InputGuardResult:
    """
    Scans the user query for prompt injection and jailbreak patterns.
    Returns InputGuardResult with is_safe=False if any pattern matches.
    """
    for pattern in COMPILED_PATTERNS:
        match = pattern.search(user_query)
        if match:
            return InputGuardResult(
                is_safe=False,
                reason=f"Blocked pattern: '{match.group()}'"
            )

    # Extra check: unusually high special character ratio
    # Attackers sometimes encode injections with unicode tricks
    special_chars = sum(1 for c in user_query if not c.isalnum() and not c.isspace())
    if len(user_query) > 20 and special_chars / len(user_query) > 0.35:
        return InputGuardResult(
            is_safe=False,
            reason="Suspicious character pattern detected"
        )

    return InputGuardResult(is_safe=True)