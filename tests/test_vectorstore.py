import numpy as np
import pytest

from app.vectorstore.base import get_vector_store
from app.vectorstore.memory_store import MemoryVectorStore


def make_store(backend, tmp_path):
    if backend == "chroma":
        return get_vector_store("chroma", persist_dir=str(tmp_path / "chroma"))
    return get_vector_store("memory", persist_dir=None)


def vecs():
    v = np.eye(4, dtype=np.float32)
    return v


@pytest.fixture(params=["memory", "chroma"])
def store(request, tmp_path):
    return make_store(request.param, tmp_path)


def test_add_query_filter_delete(store):
    store.add(
        ids=["a1", "a2", "b1", "b2"],
        vectors=vecs(),
        texts=["alpha one", "alpha two", "beta one", "beta two"],
        metadatas=[
            {"video_id": "vid_a", "start_time": 0.0, "end_time": 5.0, "index": 0},
            {"video_id": "vid_a", "start_time": 5.0, "end_time": 10.0, "index": 1},
            {"video_id": "vid_b", "start_time": 0.0, "end_time": 5.0, "index": 0},
            {"video_id": "vid_b", "start_time": 5.0, "end_time": 10.0, "index": 1},
        ],
    )
    assert store.count() == 4
    assert store.count("vid_a") == 2
    assert sorted(store.video_ids()) == ["vid_a", "vid_b"]

    q = np.array([1, 0, 0, 0], dtype=np.float32)
    hits = store.query(q, top_k=2)
    assert hits[0].id == "a1"
    assert hits[0].score == pytest.approx(1.0, abs=1e-4)

    hits = store.query(q, top_k=4, where={"video_id": "vid_b"})
    assert {h.id for h in hits} <= {"b1", "b2"}

    store.delete_video("vid_a")
    assert store.count() == 2
    assert store.video_ids() == ["vid_b"]

    docs = store.get_all_documents()
    assert {cid for cid, _, _ in docs} == {"b1", "b2"}
    md = store.get_metadata("b1")
    assert md and md["video_id"] == "vid_b"
    assert store.get_metadata("nope") is None


def test_upsert_semantics(store):
    v = np.array([[1, 0, 0, 0]], dtype=np.float32)
    store.add(["x"], v, ["first"], [{"video_id": "v"}])
    store.add(["x"], v, ["second"], [{"video_id": "v"}])
    assert store.count() == 1
    hits = store.query(v[0], top_k=1)
    assert hits[0].text == "second"


def test_empty_query(store):
    assert store.query(np.ones(4, dtype=np.float32), top_k=5) == []


def test_memory_persistence(tmp_path):
    p = tmp_path / "mem"
    s1 = MemoryVectorStore(persist_path=p)
    s1.add(["a"], np.array([[1, 0, 0, 0]], dtype=np.float32), ["txt"],
           [{"video_id": "v", "start_time": 0.0, "end_time": 1.0, "index": 0}])
    s2 = MemoryVectorStore(persist_path=p)
    assert s2.count() == 1
    assert s2.query(np.array([1, 0, 0, 0], dtype=np.float32), 1)[0].text == "txt"
