"""Free, deterministic reply-quality metrics (no API keys, no cost).

Compares each drafted reply against the dataset's reference reply:

- ROUGE-L F1  — longest-common-subsequence overlap (order-sensitive)
- token F1    — bag-of-words overlap (order-insensitive)
- placeholder fidelity — when the reference uses {{Placeholder}} slots, does the
  reply also use placeholders instead of inventing concrete details?

Honest limitation, stated in the README too: n-gram similarity to a single
reference is a blunt instrument — a good reply worded differently scores lower
than a mediocre one that parrots the reference. It is still a fair *relative*
signal when comparing two models on the same tickets, and it is 100% free.
An optional LLM judge (eval/judge.py) exists for anyone willing to pay for one.
"""

from __future__ import annotations

import re

_PLACEHOLDER = re.compile(r"\{\{[^}]+\}\}")
_TOKEN = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(_PLACEHOLDER.sub(" placeholder ", text.lower()))


def token_f1(candidate: str, reference: str) -> float:
    cand, ref = _tokens(candidate), _tokens(reference)
    if not cand or not ref:
        return 0.0
    ref_counts: dict[str, int] = {}
    for t in ref:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    overlap = 0
    for t in cand:
        if ref_counts.get(t, 0) > 0:
            overlap += 1
            ref_counts[t] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(cand)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def rouge_l(candidate: str, reference: str) -> float:
    cand, ref = _tokens(candidate), _tokens(reference)
    if not cand or not ref:
        return 0.0
    # LCS length via DP over the shorter sequence for memory
    prev = [0] * (len(ref) + 1)
    for c in cand:
        curr = [0]
        for j, r in enumerate(ref, 1):
            curr.append(prev[j - 1] + 1 if c == r else max(prev[j], curr[-1]))
        prev = curr
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    precision = lcs / len(cand)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def placeholder_ok(candidate: str, reference: str) -> bool:
    """True unless the reference uses placeholders and the reply invented none."""
    if not _PLACEHOLDER.search(reference):
        return True
    return bool(_PLACEHOLDER.search(candidate))


def score_replies(rows: list[dict], reply_key: str = "reply",
                  reference_key: str = "reference_reply") -> dict:
    """rows: [{reply, reference_reply, ...}]. Returns means across rows."""
    if not rows:
        return {"n": 0}
    n = len(rows)
    return {
        "n": n,
        "rouge_l": round(sum(rouge_l(r[reply_key], r[reference_key]) for r in rows) / n, 4),
        "token_f1": round(sum(token_f1(r[reply_key], r[reference_key]) for r in rows) / n, 4),
        "placeholder_fidelity": round(
            sum(placeholder_ok(r[reply_key], r[reference_key]) for r in rows) / n, 4
        ),
    }
