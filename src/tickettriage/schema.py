"""Shared data models for triage results."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TriageResult(BaseModel):
    """The /triage endpoint response and the unit of evaluation."""

    intent: str | None = Field(None, description="Predicted intent label, or null if unparseable")
    priority: str | None = Field(None, description="low | medium | high | urgent")
    priority_reason: str = ""
    draft_reply: str = ""
    model: str = Field("", description="Which model produced this result")
    latency_s: float = Field(0.0, description="Wall-clock seconds for all three tasks")


class JudgeScore(BaseModel):
    """LLM-judge rubric scores for one drafted reply (1 = poor, 5 = excellent)."""

    relevance: int = Field(ge=1, le=5, description="Does the reply address the ticket's issue?")
    tone: int = Field(ge=1, le=5, description="Professional, empathetic customer-support tone?")
    correctness: int = Field(
        ge=1, le=5, description="Free of fabricated policies, contradictions, and format errors?"
    )
    rationale: str = Field(description="One or two sentences justifying the scores")

    @property
    def mean(self) -> float:
        return (self.relevance + self.tone + self.correctness) / 3
