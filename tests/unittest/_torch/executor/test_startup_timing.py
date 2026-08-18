# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from typing import Iterator
from unittest.mock import patch

import pytest

from tensorrt_llm._torch.pyexecutor.startup_timing import (
    STARTUP_CPU_TIMINGS,
    log_startup_cpu_timings,
    reset_startup_cpu_timings,
    startup_timing,
)


@pytest.fixture(autouse=True)
def clear_startup_timings() -> Iterator[None]:
    reset_startup_cpu_timings()
    yield
    reset_startup_cpu_timings()


def test_startup_timing_wraps_nvtx_and_accumulates_repeated_keys() -> None:
    with (
        patch(
            "tensorrt_llm._torch.pyexecutor.startup_timing.nvtx_range",
            return_value=nullcontext(),
        ) as nvtx,
        patch(
            "tensorrt_llm._torch.pyexecutor.startup_timing.time.perf_counter",
            side_effect=[1.0, 1.25, 2.0, 2.75],
        ),
    ):
        with startup_timing("startup.test", color="blue"):
            pass
        with startup_timing("startup.test", color="blue"):
            pass

    assert STARTUP_CPU_TIMINGS == {"startup.test": pytest.approx(1.0)}
    assert nvtx.call_count == 2
    nvtx.assert_called_with(
        "startup.test",
        color="blue",
        domain="TensorRT-LLM",
        category=None,
    )


def test_log_startup_cpu_timings_reports_milliseconds() -> None:
    STARTUP_CPU_TIMINGS.update({"startup.second": 0.002, "startup.first": 1.0})

    with patch("tensorrt_llm._torch.pyexecutor.startup_timing.logger.info") as log_info:
        log_startup_cpu_timings()

    log_info.assert_called_once_with(
        "TRT-LLM startup CPU timings:\n  startup.first: 1000.00 ms\n  startup.second: 2.00 ms"
    )
