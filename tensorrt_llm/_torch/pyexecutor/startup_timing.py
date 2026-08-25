# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Iterator

from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.logger import logger

STARTUP_CPU_TIMINGS: dict[str, float] = {}
"""Cumulative wall-clock startup timings in seconds, keyed by NVTX range."""

_STARTUP_CPU_TIMINGS_LOCK = Lock()
_STARTUP_CPU_TIMINGS_PATH = Path("/tmp/trt-llm-timings.json")


@contextmanager
def startup_timing(
    key: str,
    color: str = "grey",
    domain: str = "TensorRT-LLM",
    category: str | None = None,
) -> Iterator[None]:
    """Record an NVTX range and its CPU wall-clock duration.

    Repeated uses of a key are accumulated. This accounts for startup paths
    that run more than once, such as KV-cache estimation followed by final
    executor initialization.
    """
    with nvtx_range(key, color=color, domain=domain, category=category):
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed_time = time.perf_counter() - start_time
            with _STARTUP_CPU_TIMINGS_LOCK:
                STARTUP_CPU_TIMINGS[key] = STARTUP_CPU_TIMINGS.get(key, 0.0) + elapsed_time


def reset_startup_cpu_timings() -> None:
    """Clear timings before starting a new executor initialization."""
    with _STARTUP_CPU_TIMINGS_LOCK:
        STARTUP_CPU_TIMINGS.clear()
    _STARTUP_CPU_TIMINGS_PATH.unlink(missing_ok=True)


def log_startup_cpu_timings() -> None:
    """Log all recorded startup CPU timings in milliseconds."""
    with _STARTUP_CPU_TIMINGS_LOCK:
        timings = STARTUP_CPU_TIMINGS.copy()

    temporary_path = _STARTUP_CPU_TIMINGS_PATH.with_name(
        f".{_STARTUP_CPU_TIMINGS_PATH.name}.{os.getpid()}.tmp"
    )
    with temporary_path.open("w") as timings_file:
        json.dump(timings, timings_file, indent=2, sort_keys=True)
        timings_file.write("\n")
    temporary_path.replace(_STARTUP_CPU_TIMINGS_PATH)

    if not timings:
        return

    lines = ["TRT-LLM startup CPU timings:"]
    lines.extend(
        f"  {key}: {elapsed_time * 1000:.2f} ms" for key, elapsed_time in sorted(timings.items())
    )
    logger.info("\n".join(lines))
