"""LLM-judge for reply quality: Claude scores each drafted reply against a fixed rubric.

Design decisions that keep this honest:
- Fixed rubric with anchored 1-5 scales, not vibes.
- The judge never knows which model (base vs fine-tuned) wrote a reply; callers
  pass replies independently and pair the scores afterwards.
- Structured outputs via client.messages.parse() + a Pydantic schema, so scores
  are always machine-readable integers in range.

Requires ANTHROPIC_API_KEY (or an `ant auth login` profile).
"""

from __future__ import annotations

from ..schema import JudgeScore

JUDGE_MODEL = "claude-opus-4-8"

RUBRIC = """\
You are grading a customer-support reply drafted by an AI assistant. Score it on three
dimensions, each an integer from 1 to 5. Judge only the reply text you are given.

RELEVANCE — does the reply address the specific issue in the ticket?
  1: Ignores or misreads the issue.  3: Addresses it partially or generically.
  5: Directly and completely addresses the customer's issue.

TONE — is it professional, empathetic customer-support writing?
  1: Rude, robotic, or inappropriate.  3: Acceptable but flat or awkward.
  5: Warm, professional, well-structured.

CORRECTNESS — is it free of made-up specifics and internal contradictions?
  Placeholders like {{Order Number}} are CORRECT usage (the system fills them in later);
  do not penalize them. Penalize invented policies, invented order details, contradictions,
  truncated/garbled text, or instructions that could not work.
  1: Contains fabrications or is unusable.  3: Minor unsupported claims.
  5: Everything stated is safe and consistent with the ticket.

Score strictly; a generic but harmless reply is a 3, not a 4."""


def judge_reply(client, ticket: str, reply: str) -> JudgeScore:
    """Score one reply. `client` is an anthropic.Anthropic instance (injected for testability)."""
    response = client.messages.parse(
        model=JUDGE_MODEL,
        max_tokens=1000,
        system=[{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral"}}],
        messages=[
            {
                "role": "user",
                "content": f"TICKET:\n{ticket.strip()}\n\nREPLY TO GRADE:\n{reply.strip()}",
            }
        ],
        output_format=JudgeScore,
    )
    return response.parsed_output


def judge_batch(client, rows: list[dict], reply_key: str) -> list[JudgeScore | None]:
    """Score rows[i][reply_key] against rows[i]['ticket']. Returns None on per-row failure."""
    scores: list[JudgeScore | None] = []
    for row in rows:
        try:
            scores.append(judge_reply(client, row["ticket"], row[reply_key]))
        except Exception as exc:  # keep going; report coverage at the end
            print(f"judge failed on ticket ({row['ticket'][:40]!r}...): {exc}")
            scores.append(None)
    return scores


def summarize_scores(scores: list[JudgeScore | None]) -> dict:
    valid = [s for s in scores if s is not None]
    if not valid:
        return {"n": 0}
    n = len(valid)
    return {
        "n": n,
        "n_failed": len(scores) - n,
        "relevance": round(sum(s.relevance for s in valid) / n, 2),
        "tone": round(sum(s.tone for s in valid) / n, 2),
        "correctness": round(sum(s.correctness for s in valid) / n, 2),
        "mean": round(sum(s.mean for s in valid) / n, 2),
    }
