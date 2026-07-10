# FastAPI triage service. The model itself is served by Ollama (or vLLM) next to
# this container; point TRIAGE_BACKEND at it, e.g.:
#   docker run -e TRIAGE_BACKEND="ollama:tickettriage" -e OLLAMA_HOST=... -p 8000:8000 tickettriage
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir ".[serve]"

ENV TRIAGE_BACKEND=echo
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["uvicorn", "tickettriage.api:app", "--host", "0.0.0.0", "--port", "8000"]
