from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from teklif.library import Library, load_library
from teklif.library.loader import DEFAULT_LIBRARY_DIR


@pytest.fixture(scope="session")
def lib() -> Library:
    return load_library(DEFAULT_LIBRARY_DIR)


@pytest.fixture
def lib_copy(tmp_path: Path) -> Path:
    """A writable copy of the library for tests that break it on purpose."""
    dst = tmp_path / "library"
    shutil.copytree(DEFAULT_LIBRARY_DIR, dst)
    return dst
