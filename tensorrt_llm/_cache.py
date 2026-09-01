# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Early cache configuration and cache-initialization locking.

This module intentionally depends only on the Python standard library. It is
imported by :mod:`tensorrt_llm._bootstrap` before Torch and can also be executed
directly as the bootstrap program for dynamically spawned MPI workers.
"""

import contextlib
import os
import sys
import time
from pathlib import Path
from typing import Iterator, TextIO

if os.name == "nt":
    import msvcrt
else:
    import fcntl

UNIFIED_CACHE_ROOT_ENV = "TRTLLM_CACHE_DIR"
CACHE_LOCK_TIMEOUT_ENV = "TRTLLM_CACHE_LOCK_TIMEOUT"
CACHE_RANK_ENV = "TRTLLM_CACHE_RANK"
_DERIVED_CACHE_ENV_VARS = "TRTLLM_UNIFIED_CACHE_ENV_VARS"
_CACHE_LOCK_MANAGED_ENV = "TRTLLM_CACHE_LOCK_MANAGED"
_DEFAULT_CACHE_LOCK_TIMEOUT = 600.0

# Keep the order stable. Every endpoint acquires cache locks in this order,
# preventing lock-order inversions when several endpoints start concurrently.
_CACHE_PATHS = (
    ("TLLM_AUTOTUNER_CACHE_PATH", "autotuner", "cache.json"),
    ("TORCHINDUCTOR_CACHE_DIR", "torchinductor", None),
    ("TRITON_CACHE_DIR", "triton", None),
    ("TORCH_EXTENSIONS_DIR", "torch_extensions", None),
    ("FLASHINFER_WORKSPACE_BASE", "flashinfer", None),
    ("CUTE_DSL_CACHE_DIR", "cute_dsl", None),
    ("DG_JIT_CACHE_DIR", "deep_gemm", None),
    ("TRTLLM_DG_CACHE_DIR", "trtllm_deep_gemm", None),
    ("CUDA_CACHE_PATH", "cuda", None),
)

_held_cache_locks: list[TextIO] = []


def _try_lock(lock: TextIO) -> None:
    if os.name == "nt":
        lock.seek(0, os.SEEK_END)
        if lock.tell() == 0:
            lock.write("\0")
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise BlockingIOError from error
    else:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(lock: TextIO) -> None:
    if os.name == "nt":
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(lock, fcntl.LOCK_UN)


def _distributed_rank() -> int:
    """Return a launcher-provided global rank without importing MPI."""
    for name in (
        CACHE_RANK_ENV,
        "OMPI_COMM_WORLD_RANK",
        "PMIX_RANK",
        "PMI_RANK",
        "SLURM_PROCID",
        "RANK",
    ):
        value = os.environ.get(name)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                continue
    return 0


def _normalized_cache_root() -> Path | None:
    value = os.environ.get(UNIFIED_CACHE_ROOT_ENV)
    if not value:
        return None
    root = Path(value).expanduser().absolute()
    os.environ[UNIFIED_CACHE_ROOT_ENV] = str(root)
    return root


def unified_cache_enabled() -> bool:
    return bool(os.environ.get(UNIFIED_CACHE_ROOT_ENV))


def configure_unified_caches(rank: int | None = None, *, rerank: bool = False) -> bool:
    """Configure cache environment variables beneath the unified cache root.

    Explicitly configured cache variables take precedence. ``rerank`` is used
    by a dynamically spawned MPI worker to replace only values that this
    function derived earlier in the parent process.
    """
    root = _normalized_cache_root()
    if root is None:
        return False

    rank = _distributed_rank() if rank is None else rank
    derived = set(filter(None, os.environ.get(_DERIVED_CACHE_ENV_VARS, "").split(",")))
    rank_dir = f"rank-{rank}"
    for env_name, cache_name, filename in _CACHE_PATHS:
        if env_name in os.environ and not (rerank and env_name in derived):
            continue
        path = root / cache_name / rank_dir
        path.mkdir(parents=True, exist_ok=True)
        os.environ[env_name] = str(path / filename if filename else path)
        derived.add(env_name)

    os.environ[_DERIVED_CACHE_ENV_VARS] = ",".join(
        env_name for env_name, _, _ in _CACHE_PATHS if env_name in derived
    )
    return True


def derived_cache_env_vars() -> set[str]:
    return set(filter(None, os.environ.get(_DERIVED_CACHE_ENV_VARS, "").split(",")))


def cache_bootstrap_path() -> str:
    return str(Path(__file__).resolve())


def cache_lock_timeout() -> float:
    value = os.environ.get(CACHE_LOCK_TIMEOUT_ENV)
    if value is None:
        return _DEFAULT_CACHE_LOCK_TIMEOUT
    try:
        timeout = float(value)
    except ValueError as error:
        raise ValueError(f"{CACHE_LOCK_TIMEOUT_ENV} must be a number, got {value!r}") from error
    if timeout < 0:
        raise ValueError(f"{CACHE_LOCK_TIMEOUT_ENV} must be non-negative, got {value!r}")
    return timeout


def acquire_cache_initialization_locks() -> bool:
    """Acquire one initialization lock for every unified cache directory."""
    if not unified_cache_enabled() or _held_cache_locks:
        return False

    root = _normalized_cache_root()
    assert root is not None
    deadline = time.monotonic() + cache_lock_timeout()
    try:
        for _, cache_name, _ in _CACHE_PATHS:
            cache_dir = root / cache_name
            cache_dir.mkdir(parents=True, exist_ok=True)
            lock = (cache_dir / ".trtllm-cache.lock").open("a")
            while True:
                try:
                    _try_lock(lock)
                    break
                except BlockingIOError as error:
                    if time.monotonic() >= deadline:
                        lock.close()
                        raise TimeoutError(
                            f"Timed out waiting for cache initialization lock {lock.name}"
                        ) from error
                    time.sleep(0.1)
            _held_cache_locks.append(lock)
    except (OSError, TimeoutError, ValueError):
        release_cache_initialization_locks()
        raise
    return True


def release_cache_initialization_locks() -> None:
    while _held_cache_locks:
        lock = _held_cache_locks.pop()
        try:
            _unlock(lock)
        except OSError:
            # Closing the descriptor also drops the kernel lock. Do not strand
            # subordinate ranks at their post-initialization synchronization.
            pass
        finally:
            lock.close()


@contextlib.contextmanager
def cache_initialization_lock_if_unmanaged() -> Iterator[None]:
    """Hold the unified cache lease for a directly constructed worker."""
    managed = os.environ.get(_CACHE_LOCK_MANAGED_ENV) == "1"
    acquired = False
    if not managed:
        acquired = acquire_cache_initialization_locks()
    try:
        yield
    finally:
        if acquired:
            release_cache_initialization_locks()


def mark_cache_lock_managed() -> None:
    os.environ[_CACHE_LOCK_MANAGED_ENV] = "1"


def unmark_cache_lock_managed() -> None:
    os.environ.pop(_CACHE_LOCK_MANAGED_ENV, None)


def _run_legacy_flashinfer_bootstrap(workspace_root: str) -> None:
    """Preserve the opt-in per-process FlashInfer behavior used without a root."""
    from mpi4py import MPI
    from mpi4py.futures.server import main

    workspace_lock = None
    rank: int | str = "unknown"
    if "FLASHINFER_WORKSPACE_BASE" not in os.environ:
        try:
            root = Path(workspace_root).expanduser()
            rank = MPI.COMM_WORLD.Get_rank()
            slot = rank
            slot_stride = MPI.COMM_WORLD.Get_size()
            while True:
                workspace = root / f"rank-{slot}"
                workspace.mkdir(parents=True, exist_ok=True)
                workspace_lock = (workspace / ".lock").open("a")
                try:
                    _try_lock(workspace_lock)
                    break
                except BlockingIOError:
                    workspace_lock.close()
                    workspace_lock = None
                    slot += slot_stride

            os.environ.setdefault(
                "FLASHINFER_CUBIN_DIR",
                str(Path.home() / ".cache" / "flashinfer" / "cubins"),
            )
            os.environ["FLASHINFER_WORKSPACE_BASE"] = str(workspace)
        except Exception as error:  # noqa: BLE001
            if workspace_lock is not None:
                try:
                    workspace_lock.close()
                except Exception as close_error:  # noqa: BLE001
                    print(
                        f"[trtllm] rank {rank} could not close a failed FlashInfer "
                        f"workspace lock ({close_error})",
                        file=sys.stderr,
                    )
            workspace_lock = None
            print(
                f"[trtllm] rank {rank} could not isolate its FlashInfer workspace "
                f"({error}); falling back to FlashInfer's shared defaults",
                file=sys.stderr,
            )

    try:
        main()
    finally:
        if workspace_lock is not None:
            try:
                _unlock(workspace_lock)
            except OSError as error:
                print(
                    f"[trtllm] rank {rank} could not unlock the FlashInfer workspace ({error})",
                    file=sys.stderr,
                )
            try:
                workspace_lock.close()
            except OSError as error:
                print(
                    f"[trtllm] rank {rank} could not close the FlashInfer workspace lock ({error})",
                    file=sys.stderr,
                )


def _run_unified_mpi_bootstrap() -> None:
    from mpi4py import MPI
    from mpi4py.futures.server import main

    configure_unified_caches(MPI.COMM_WORLD.Get_rank(), rerank=True)
    main()


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "unified":
        _run_unified_mpi_bootstrap()
    elif mode == "flashinfer-isolate":
        _run_legacy_flashinfer_bootstrap(sys.argv[2])
    else:
        raise ValueError(f"Unknown TensorRT-LLM cache bootstrap mode: {mode}")
