"""Prompt templates for the three triage tasks.

The same templates are used for training data generation, evaluation, and serving,
so the fine-tuned model always sees prompts in the exact format it was trained on.
"""

from __future__ import annotations

INTENT_LABELS = [
    "cancel_order", "change_order", "change_shipping_address", "check_cancellation_fee",
    "check_invoice", "check_payment_methods", "check_refund_policy", "complaint",
    "contact_customer_service", "contact_human_agent", "create_account", "delete_account",
    "delivery_options", "delivery_period", "edit_account", "get_invoice", "get_refund",
    "newsletter_subscription", "payment_issue", "place_order", "recover_password",
    "registration_problems", "review", "set_up_shipping_address", "switch_account",
    "track_order", "track_refund",
]

PRIORITY_LABELS = ["low", "medium", "high", "urgent"]

SYSTEM_PROMPT = (
    "You are TicketTriage, an assistant for a customer support team. "
    "You classify support tickets, score their priority, and draft replies. "
    "Follow the task instruction exactly and answer in the requested format."
)

CLASSIFY_INSTRUCTION = (
    "Classify the customer ticket below into exactly one intent label. "
    "Answer with only the label, nothing else.\n"
    "Valid labels: {labels}\n\n"
    "Ticket: {ticket}"
)

PRIORITY_INSTRUCTION = (
    "Assign a priority to the customer ticket below. "
    "Answer on two lines:\n"
    "priority: <low|medium|high|urgent>\n"
    "reason: <one sentence>\n\n"
    "Ticket: {ticket}"
)

REPLY_INSTRUCTION = (
    "Draft a complete, polite reply to the customer ticket below. "
    "Keep placeholders like {{Order Number}} for values you do not know.\n\n"
    "Ticket: {ticket}"
)


def classify_prompt(ticket: str) -> str:
    return CLASSIFY_INSTRUCTION.format(labels=", ".join(INTENT_LABELS), ticket=ticket.strip())


def priority_prompt(ticket: str) -> str:
    return PRIORITY_INSTRUCTION.format(ticket=ticket.strip())


def reply_prompt(ticket: str) -> str:
    return REPLY_INSTRUCTION.format(ticket=ticket.strip())


def chat_example(user_prompt: str, assistant_response: str) -> dict:
    """One training example in messages (ChatML-compatible) format."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response},
        ]
    }


def parse_priority_output(text: str) -> tuple[str | None, str]:
    """Parse 'priority: X\nreason: Y' model output. Returns (priority|None, reason)."""
    priority, reason = None, ""
    for line in text.strip().splitlines():
        line = line.strip()
        lower = line.lower()
        if lower.startswith("priority:"):
            candidate = line.split(":", 1)[1].strip().lower().rstrip(".")
            if candidate in PRIORITY_LABELS:
                priority = candidate
        elif lower.startswith("reason:"):
            reason = line.split(":", 1)[1].strip()
    if priority is None:
        # fall back to first priority word anywhere in the output
        lowered = text.lower()
        found = [(lowered.find(p), p) for p in PRIORITY_LABELS if p in lowered]
        if found:
            priority = min(found)[1]
    return priority, reason


def parse_intent_output(text: str) -> str | None:
    """Parse an intent label from model output; tolerant of extra prose."""
    cleaned = text.strip().lower()
    if cleaned in INTENT_LABELS:
        return cleaned
    # first exact label mentioned anywhere wins
    found = [(cleaned.find(label), label) for label in INTENT_LABELS if label in cleaned]
    if found:
        return min(found)[1]
    return None
