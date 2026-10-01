"""Shared pytest fixtures — everything offline & deterministic."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.services.container import build_container

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = PROJECT_ROOT / "data" / "fixtures" / "demo_lecture.json"
FIXTURE_URL = f"fixture:{FIXTURE_PATH}"


@pytest.fixture()
def settings(tmp_path) -> Settings:
    return Settings(
        embedding_provider="hashing",
        embedding_dim=256,
        vectorstore_backend="memory",
        chroma_persist=False,
        llm_provider="mock",
        reranker_enabled=False,          # tests opt into HeuristicReranker
        data_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        vectorstore_dir=tmp_path / "vectorstore",
        log_level="WARNING",
    )


@pytest.fixture()
def container(settings):
    return build_container(settings)


@pytest.fixture()
def demo(container):
    """Ingested demo fixture; returns VideoMeta."""
    meta, cached, _ = container.videos.process(FIXTURE_URL)
    assert not cached
    return meta


@pytest.fixture()
def demo_container(container, demo):
    """Container with the demo video already ingested."""
    return container
