from fastapi.testclient import TestClient

from tickettriage.api import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["backend"] == "echo"


def test_triage_echo_backend():
    resp = client.post("/triage", json={"ticket": "where is my order? I need it urgently"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "track_order"
    assert body["priority"] == "medium"
    assert "{{Order Number}}" in body["draft_reply"]
    assert body["latency_s"] >= 0


def test_triage_rejects_empty():
    resp = client.post("/triage", json={"ticket": ""})
    assert resp.status_code == 422
