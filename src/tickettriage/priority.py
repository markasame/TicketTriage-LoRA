"""Rule-based priority labeling for the Bitext dataset.

The Bitext dataset has intent labels but no priority labels, so training targets
for the priority task are derived from (a) a base priority per intent and
(b) urgency signals in the ticket text. These are documented, deterministic
"silver" labels — the eval measures agreement with them, and the README calls
this out honestly rather than pretending they are human annotations.
"""

from __future__ import annotations

import re

# Base priority per intent: how disruptive is this issue when expressed neutrally?
INTENT_BASE_PRIORITY: dict[str, str] = {
    # money or access is broken right now
    "payment_issue": "urgent",
    "recover_password": "high",
    "registration_problems": "high",
    "complaint": "high",
    "cancel_order": "high",
    "get_refund": "high",
    "track_refund": "medium",
    "check_cancellation_fee": "medium",
    "change_order": "medium",
    "change_shipping_address": "medium",
    "set_up_shipping_address": "medium",
    "track_order": "medium",
    "delivery_period": "medium",
    "delivery_options": "low",
    "place_order": "medium",
    "check_invoice": "low",
    "get_invoice": "low",
    "check_payment_methods": "low",
    "check_refund_policy": "low",
    "contact_customer_service": "medium",
    "contact_human_agent": "medium",
    "create_account": "low",
    "delete_account": "medium",
    "edit_account": "low",
    "switch_account": "low",
    "newsletter_subscription": "low",
    "review": "low",
}

_ESCALATE_PATTERNS = re.compile(
    r"\b(urgent|immediately|right now|asap|as soon as possible|straight ?away|"
    r"unauthorized|fraud|charged twice|double charge|can'?t access|cannot access|"
    r"locked out|furious|angry|outraged|unacceptable|last time|lawyer|legal action)\b",
    re.IGNORECASE,
)

_ORDER = ["low", "medium", "high", "urgent"]

_REASONS = {
    "low": "Routine informational request with no time pressure.",
    "medium": "Standard account or order task that should be handled in normal queue order.",
    "high": "The customer is blocked or money is involved, so it should be handled soon.",
    "urgent": "The customer signals strong urgency or a payment/access failure needing immediate action.",
}


def escalation_signals(ticket: str) -> list[str]:
    """Urgency phrases found in the ticket text."""
    return [m.group(0).lower() for m in _ESCALATE_PATTERNS.finditer(ticket)]


def label_priority(intent: str, ticket: str) -> tuple[str, str]:
    """Return (priority, one-sentence reason) for a ticket with a known intent."""
    base = INTENT_BASE_PRIORITY.get(intent, "medium")
    signals = escalation_signals(ticket)
    level = _ORDER.index(base)
    if signals:
        level = min(level + 1, len(_ORDER) - 1)
    priority = _ORDER[level]
    reason = _REASONS[priority]
    if signals:
        reason = f"The customer uses urgent language ('{signals[0]}'); " + reason[0].lower() + reason[1:]
    return priority, reason


def priority_target_text(intent: str, ticket: str) -> str:
    """Training target string for the priority task."""
    priority, reason = label_priority(intent, ticket)
    return f"priority: {priority}\nreason: {reason}"
