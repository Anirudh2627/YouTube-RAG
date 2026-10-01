import numpy as np
import pytest

from app.embeddings.base import get_embedder
from app.embeddings.hashing import HashingEmbedder


def test_hashing_deterministic_and_normalized():
    e = HashingEmbedder(dim=128)
    a = e.embed(["hello world", "another text"])
    b = e.embed(["hello world", "another text"])
    assert np.allclose(a, b)
    norms = np.linalg.norm(a, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    assert a.shape == (2, 128)


def test_hashing_similarity_ordering():
    e = HashingEmbedder(dim=512)
    q = e.embed_query("the transformer uses attention heads")
    close = e.embed(["transformers use multiple attention heads"])[0]
    far = e.embed(["cooking pasta with tomato sauce and basil"])[0]
    assert float(q @ close) > float(q @ far)


def test_factory_hashing():
    assert isinstance(get_embedder("hashing", "x", dim=64), HashingEmbedder)
    with pytest.raises(ValueError):
        get_embedder("nope", "x")


@pytest.mark.slow
def test_sentence_transformers_embedder():
    e = get_embedder("sentence-transformers", "BAAI/bge-small-en-v1.5")
    v = e.embed(["hello world"])
    assert v.shape == (1, e.dim)
    assert np.isclose(np.linalg.norm(v[0]), 1.0, atol=1e-4)
    q = e.embed_query("what is attention?")
    assert q.shape == (e.dim,)
