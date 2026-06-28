# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for bridge-owned BFM tracking curriculum events."""

from __future__ import annotations

import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from phase2_adapter.curriculum import run_curriculum_event, tracking_priorities
from phase2_adapter.evaluation import EXPECTED_BFM_MOTIONS


def _metrics() -> dict[str, dict[str, float | int]]:
    return {
        f"motion-{motion_id}": {
            "motion_id": motion_id,
            "emd": motion_id / (EXPECTED_BFM_MOTIONS - 1) * 3.0,
        }
        for motion_id in reversed(range(EXPECTED_BFM_MOTIONS))
    }


def test_tracking_priorities_match_released_exponential_law() -> None:
    """Metrics should be reordered by motion id before applying the released law."""
    priorities = tracking_priorities(_metrics(), "cpu")

    assert priorities.shape == (EXPECTED_BFM_MOTIONS,)
    torch.testing.assert_close(priorities[0], torch.tensor(2.0))
    torch.testing.assert_close(priorities[-1], torch.tensor(16.0))


def test_curriculum_event_updates_native_sampling_and_writes_once(tmp_path, monkeypatch) -> None:
    """One event should atomically publish the exact weights applied to the motion library."""
    metrics = _metrics()
    monkeypatch.setattr(
        "phase2_adapter.curriculum.run_native_tracking",
        lambda *_args, **_kwargs: (metrics, 3.5),
    )
    updates: list[tuple[list[float], list[int]]] = []
    motion_lib = SimpleNamespace(update_sampling_weight_by_id=lambda *, priorities, motions_id: updates.append((priorities, motions_id)))
    env = SimpleNamespace(_env=SimpleNamespace(_motion_lib=motion_lib))

    priorities = run_curriculum_event(
        object(),
        env=env,
        num_envs=1024,
        transition=0,
        output_dir=tmp_path,
        device="cpu",
        update_expert_priorities=lambda _values: None,
    )

    record = json.loads((tmp_path / "0.json").read_text())
    assert record["schema"] == "bfm_zero_curriculum_event_v1"
    assert record["duration_seconds"] == 3.5
    assert record["priorities"] == priorities.tolist()
    assert updates == [(priorities.tolist(), list(range(EXPECTED_BFM_MOTIONS)))]
    with pytest.raises(FileExistsError):
        run_curriculum_event(
            object(),
            env=env,
            num_envs=1024,
            transition=0,
            output_dir=tmp_path,
            device="cpu",
            update_expert_priorities=lambda _values: None,
        )


def test_curriculum_evaluator_cannot_advance_training_rng(tmp_path, monkeypatch) -> None:
    """Only declared sampling weights, not evaluator mechanics, may affect training state."""
    metrics = _metrics()

    def consume_rng(*_args, **_kwargs):
        random.random()
        np.random.rand()
        torch.rand(1)
        return metrics, 1.0

    monkeypatch.setattr("phase2_adapter.curriculum.run_native_tracking", consume_rng)
    motion_lib = SimpleNamespace(update_sampling_weight_by_id=lambda **_kwargs: None)
    env = SimpleNamespace(_env=SimpleNamespace(_motion_lib=motion_lib))
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    expected = (random.random(), np.random.rand(), torch.rand(1))
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)

    run_curriculum_event(
        object(),
        env=env,
        num_envs=1024,
        transition=0,
        output_dir=tmp_path,
        device="cpu",
        update_expert_priorities=lambda _values: None,
    )
    actual = (random.random(), np.random.rand(), torch.rand(1))

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2])
