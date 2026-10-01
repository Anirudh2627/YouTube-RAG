"""API endpoint tests via FastAPI TestClient (offline config)."""
import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_container
from app.api.main import create_app
from tests.conftest import FIXTURE_URL


@pytest.fixture()
def client(container):
    app = create_app()
    app.dependency_overrides[get_container] = lambda: container
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["embedding_model"] == "hashing"


def test_process_video_fixture(client):
    r = client.post("/videos/process", json={"url": FIXTURE_URL})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["video"]["video_id"] == "demoLctr001"
    assert body["video"]["n_chunks"] > 0
    assert body["cached"] is False

    # second call hits the cache
    r2 = client.post("/videos/process", json={"url": FIXTURE_URL})
    assert r2.json()["cached"] is True


def test_process_bad_url(client):
    r = client.post("/videos/process", json={"url": "https://example.com/nope"})
    assert r.status_code == 400


def test_get_video_and_sources(client):
    client.post("/videos/process", json={"url": FIXTURE_URL})
    r = client.get("/videos/demoLctr001")
    assert r.status_code == 200
    assert r.json()["title"]

    r = client.get("/videos/demoLctr001/sources")
    assert r.status_code == 200
    chunks = r.json()
    assert chunks and all("start_time" in c and "chunk_id" in c for c in chunks)

    assert client.get("/videos").status_code == 200
    assert client.get("/videos/doesNotExist1").status_code == 404
    assert client.get("/videos/doesNotExist1/sources").status_code == 404


def test_chat_flow(client):
    client.post("/videos/process", json={"url": FIXTURE_URL})
    r = client.post("/chat", json={
        "video_id": "demoLctr001",
        "query": "Why is the scaling by root d_k necessary?",
        "debug": True,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"]
    assert body["conversation_id"]
    assert body["sources"]
    for s in body["sources"]:
        assert s["url"].startswith("https://www.youtube.com/watch?v=demoLctr001&t=")
        assert s["timestamp_label"]
    assert body["debug"]["final_context"]

    # follow-up keeps the conversation
    r2 = client.post("/chat", json={
        "video_id": "demoLctr001",
        "query": "Why does that matter for gradients?",
        "conversation_id": body["conversation_id"],
    })
    assert r2.status_code == 200
    assert r2.json()["conversation_id"] == body["conversation_id"]


def test_chat_validation(client):
    assert client.post("/chat", json={"query": "  "}).status_code == 400
    assert client.post("/chat", json={"query": "hi", "video_id": "notIngested1"}).status_code == 404


def test_delete_video(client):
    client.post("/videos/process", json={"url": FIXTURE_URL})
    assert client.delete("/videos/demoLctr001").status_code == 204
    assert client.get("/videos/demoLctr001").status_code == 404
    assert client.delete("/videos/demoLctr001").status_code == 404
