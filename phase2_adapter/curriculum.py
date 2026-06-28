# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Apply the released BFM tracking curriculum at declared training events."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .evaluation import EXPECTED_BFM_MOTIONS, run_native_tracking


def tracking_priorities(
    metrics: Mapping[str, Mapping[str, Any]],
    device: str | torch.device,
) -> torch.Tensor:
    """Return released exponential priorities in native motion-id order."""
    if len(metrics) != EXPECTED_BFM_MOTIONS:
        raise ValueError(f"Expected {EXPECTED_BFM_MOTIONS} curriculum motions, got {len(metrics)}.")
    emd_by_motion = [0.0] * EXPECTED_BFM_MOTIONS
    seen: set[int] = set()
    for values in metrics.values():
        raw_motion_id = float(values["motion_id"])
        if not math.isfinite(raw_motion_id) or not raw_motion_id.is_integer():
            raise ValueError(f"Curriculum motion id must be an integer, got {raw_motion_id}.")
        motion_id = int(raw_motion_id)
        if motion_id < 0 or motion_id >= EXPECTED_BFM_MOTIONS:
            raise ValueError(f"Curriculum motion id is out of range: {motion_id}.")
        if motion_id in seen:
            raise ValueError(f"Duplicate curriculum motion id: {motion_id}.")
        emd = float(values["emd"])
        if not math.isfinite(emd):
            raise ValueError(f"Curriculum EMD for motion {motion_id} must be finite.")
        seen.add(motion_id)
        emd_by_motion[motion_id] = emd
    if len(seen) != EXPECTED_BFM_MOTIONS:
        raise ValueError("Curriculum metrics do not cover every native motion id.")
    emd = torch.tensor(emd_by_motion, dtype=torch.float32, device=device)
    return torch.pow(2.0, torch.clamp(emd, min=0.5, max=2.0) * 2.0)


def run_curriculum_event(
    agent_or_model: Any,
    *,
    env: Any,
    num_envs: int,
    transition: int,
    output_dir: str | Path,
    device: str | torch.device,
    update_expert_priorities: Callable[[torch.Tensor], None],
) -> torch.Tensor:
    """Measure tracking, update native motion sampling, and persist the training input."""
    with _preserve_evaluator_rng(device):
        metrics, duration = run_native_tracking(agent_or_model, env=env, num_envs=num_envs)
    priorities = tracking_priorities(metrics, device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{transition}.json"
    if destination.exists():
        raise FileExistsError(f"Curriculum event already exists: {destination}")
    temporary = destination.with_suffix(".tmp")
    with temporary.open("x") as stream:
        json.dump(
            {
                "schema": "bfm_zero_curriculum_event_v1",
                "transition": transition,
                "duration_seconds": duration,
                "priorities": priorities.cpu().tolist(),
                "metrics": metrics,
            },
            stream,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
    try:
        env._env._motion_lib.update_sampling_weight_by_id(
            priorities=priorities.tolist(),
            motions_id=list(range(EXPECTED_BFM_MOTIONS)),
        )
        update_expert_priorities(priorities)
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return priorities


@contextmanager
def _preserve_evaluator_rng(device: str | torch.device):
    """Prevent evaluator mechanics from becoming an undeclared training input."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_device = torch.device(device)
    cuda_devices = []
    if torch_device.type == "cuda":
        cuda_devices = [torch_device.index if torch_device.index is not None else torch.cuda.current_device()]
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}.")
