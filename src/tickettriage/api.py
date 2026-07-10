"""FastAPI service exposing POST /triage.

Backend is chosen with the TRIAGE_BACKEND env var (see inference.make_backend),
e.g. TRIAGE_BACKEND="ollama:tickettriage" or "hf:unsloth/Qwen3-8B-bnb-4bit@models/adapter".
Defaults to the echo backend so the service starts without a model (useful in CI).
"""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .inference import Backend, make_backend
from .schema import TriageResult

app = FastAPI(title="TicketTriage-LoRA", version="0.1.0")


class TicketRequest(BaseModel):
    ticket: str = Field(min_length=3, max_length=8000)


@lru_cache(maxsize=1)
def get_backend() -> Backend:
    return make_backend(os.environ.get("TRIAGE_BACKEND", "echo"))


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "backend": get_backend().name}


@app.post("/triage", response_model=TriageResult)
def triage(req: TicketRequest) -> TriageResult:
    return get_backend().triage(req.ticket)
