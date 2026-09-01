# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest

from tensorrt_llm import _cache

try:
    import fcntl
except ModuleNotFoundError:
    fcntl = None


@pytest.fixture(autouse=True)
def _clean_cache_environment(monkeypatch: pytest.MonkeyPatch):
    _cache.release_cache_initialization_locks()
    for name, _, _ in _cache._CACHE_PATHS:
        monkeypatch.delenv(name, raising=False)
    for name in (
        _cache.UNIFIED_CACHE_ROOT_ENV,
        _cache.CACHE_LOCK_TIMEOUT_ENV,
        _cache.CACHE_RANK_ENV,
        _cache._DERIVED_CACHE_ENV_VARS,
        _cache._CACHE_LOCK_MANAGED_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    yield
    _cache.release_cache_initialization_locks()


def test_configures_every_cache_per_rank(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(_cache.UNIFIED_CACHE_ROOT_ENV, str(tmp_path / "cache"))

    assert _cache.configure_unified_caches(rank=3)

    root = (tmp_path / "cache").absolute()
    for env_name, cache_name, filename in _cache._CACHE_PATHS:
        rank_dir = root / cache_name / "rank-3"
        expected = rank_dir / filename if filename else rank_dir
        assert os.environ[env_name] == str(expected)
        assert rank_dir.is_dir()


def test_explicit_cache_variable_takes_precedence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    explicit = tmp_path / "explicit-triton"
    monkeypatch.setenv(_cache.UNIFIED_CACHE_ROOT_ENV, str(tmp_path / "cache"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(explicit))

    _cache.configure_unified_caches(rank=1)

    assert os.environ["TRITON_CACHE_DIR"] == str(explicit)
    assert "TRITON_CACHE_DIR" not in _cache.derived_cache_env_vars()


def test_mpi_worker_reranks_only_derived_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    explicit = tmp_path / "explicit-cuda"
    monkeypatch.setenv(_cache.UNIFIED_CACHE_ROOT_ENV, str(tmp_path / "cache"))
    monkeypatch.setenv("CUDA_CACHE_PATH", str(explicit))
    _cache.configure_unified_caches(rank=0)

    _cache.configure_unified_caches(rank=5, rerank=True)

    assert Path(os.environ["TRITON_CACHE_DIR"]).name == "rank-5"
    assert os.environ["CUDA_CACHE_PATH"] == str(explicit)


def test_uses_launcher_rank(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(_cache.UNIFIED_CACHE_ROOT_ENV, str(tmp_path / "cache"))
    monkeypatch.setenv(_cache.CACHE_RANK_ENV, "6")

    _cache.configure_unified_caches()

    assert Path(os.environ["TRITON_CACHE_DIR"]).name == "rank-6"


def test_without_unified_root_does_nothing() -> None:
    assert not _cache.configure_unified_caches(rank=0)
    assert all(name not in os.environ for name, _, _ in _cache._CACHE_PATHS)


def test_cache_lock_times_out_and_releases_partial_lease(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    if fcntl is None:
        pytest.skip("POSIX lock contention test")
    root = tmp_path / "cache"
    monkeypatch.setenv(_cache.UNIFIED_CACHE_ROOT_ENV, str(root))
    monkeypatch.setenv(_cache.CACHE_LOCK_TIMEOUT_ENV, "0.05")
    _cache.configure_unified_caches(rank=0)
    lock_path = root / "autotuner" / ".trtllm-cache.lock"

    with lock_path.open("a") as competing_lock:
        fcntl.flock(competing_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(TimeoutError, match="autotuner"):
            _cache.acquire_cache_initialization_locks()
        fcntl.flock(competing_lock, fcntl.LOCK_UN)

    assert not _cache._held_cache_locks


def test_one_lock_per_cache_root_not_per_rank(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "cache"
    monkeypatch.setenv(_cache.UNIFIED_CACHE_ROOT_ENV, str(root))
    _cache.configure_unified_caches(rank=7)

    assert _cache.acquire_cache_initialization_locks()

    lock_paths = {Path(lock.name) for lock in _cache._held_cache_locks}
    assert lock_paths == {
        root / cache_name / ".trtllm-cache.lock" for _, cache_name, _ in _cache._CACHE_PATHS
    }
    assert all("rank-7" not in str(path) for path in lock_paths)
