from __future__ import annotations
from functools import lru_cache
from app.services.container import Container, build_container
@lru_cache
def get_container() -> Container:
    return build_container()
