# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the bounded no-checkpoint BFM candidate throughput entrypoint."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from phase2_adapter import throughput


def _contract(tmp_path: Path) -> Path:
    reference = tmp_path / "reference.json"
    data = tmp_path / "lafan.pkl"
    reference.write_text("{}\n")
    data.write_bytes(b"motion")
    shared = {
        **throughput._REQUIRED_SHARED_VALUES,
        "batch_size": 1_024,
        "replay_capacity_transitions": 5_120_000,
    }
    payload = {
        "schema": throughput.THROUGHPUT_CONTRACT_SCHEMA,
        "shared_run_spec": shared,
        "timing_contract": {
            "boundary_hooks": throughput.BOUNDARY_HOOKS,
            "boundary_synchronizations": 2,
        },
        "profiles": [
            {
                "model_profile": name,
                "hidden_dim": topology[0],
                "hidden_layers": topology[1],
            }
            for name, topology in throughput.BFM_MODEL_PROFILES.items()
        ],
        "training_inputs": {
            "reference_config": str(reference),
            "data": str(data),
            "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        },
    }
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _curriculum(output: Path) -> None:
    path = output / "curriculum_events/0.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema": "bfm_zero_curriculum_event_v1",
                "transition": 0,
                "duration_seconds": 1.0,
                "priorities": [1.0] * throughput.MOTION_COUNT,
                "metrics": {str(index): {} for index in range(throughput.MOTION_COUNT)},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _complete_timing(
    runner: throughput.BFMThroughputRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> list[torch.device]:
    synchronizations = []
    clock = iter((10.0, 12.0))
    monkeypatch.setattr(throughput.torch.cuda, "synchronize", synchronizations.append)
    monkeypatch.setattr(throughput.time, "monotonic", lambda: next(clock))
    for iteration in range(throughput.TOTAL_ITERATIONS):
        runner._observe_iteration_start(iteration, iteration * throughput.NUM_ENVS)
        runner._observe_iteration_learning_complete(
            iteration,
            (iteration + 1) * throughput.NUM_ENVS,
        )
        runner._observe_iteration_complete(
            iteration,
            (iteration + 1) * throughput.NUM_ENVS,
            0.25,
            0.75,
        )
    return synchronizations


def _runner() -> throughput.BFMThroughputRunner:
    runner = object.__new__(throughput.BFMThroughputRunner)
    runner.current_learning_iteration = 0
    runner.collected_transitions = 0
    runner.device = "cuda:0"
    runner.env = SimpleNamespace(num_envs=throughput.NUM_ENVS)
    runner.prepare_timing()
    return runner


def test_throughput_contract_freezes_exact_runner_cadence(tmp_path: Path) -> None:
    contract = _contract(tmp_path)

    spec = throughput.load_throughput_spec(contract, "residual_6x1024")

    assert spec.contract == contract
    assert spec.model_profile == "residual_6x1024"
    assert throughput.PRE_UPDATE_ITERATIONS == 11
    assert throughput.MEASUREMENT_ITERATION_FIRST == 43
    assert throughput.MEASUREMENT_ITERATION_LAST == 170
    assert throughput.TOTAL_ITERATIONS == 171
    assert throughput.MEASUREMENT_TRANSITIONS == 128 * 1_024


def test_throughput_contract_rejects_window_drift(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    payload = json.loads(contract.read_text())
    payload["shared_run_spec"]["measurement_iteration_first"] = 42
    contract.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="measurement_iteration_first"):
        throughput.load_throughput_spec(contract, "residual_6x1024")


def test_throughput_runner_uses_two_exact_sync_boundaries_and_128_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = throughput.load_throughput_spec(_contract(tmp_path), "residual_6x1024")
    output = tmp_path / "output"
    output.mkdir()
    _curriculum(output)
    runner = _runner()

    synchronizations = _complete_timing(runner, monkeypatch)
    summary = runner.write_timing(output, spec)

    assert synchronizations == [torch.device("cuda:0"), torch.device("cuda:0")]
    assert summary["measurement_wall_seconds"] == 2.0
    assert summary["total_transitions_per_second"] == throughput.MEASUREMENT_TRANSITIONS / 2.0
    assert summary["boundary_hooks"] == throughput.BOUNDARY_HOOKS
    with (output / throughput.TIMING_CSV_NAME).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 128
    assert int(rows[0]["iteration"]) == 43
    assert int(rows[-1]["iteration"]) == 170
    assert throughput.load_throughput_timing(output, spec) == summary


def test_throughput_timing_rejects_tensor_or_logger_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = throughput.load_throughput_spec(_contract(tmp_path), "residual_6x1024")
    output = tmp_path / "output"
    output.mkdir()
    _curriculum(output)
    runner = _runner()
    _complete_timing(runner, monkeypatch)
    runner.write_timing(output, spec)

    tensor = output / "policy.pt"
    tensor.write_bytes(b"forbidden")
    with pytest.raises(ValueError, match="tensor artifact"):
        throughput.load_throughput_timing(output, spec)
    tensor.unlink()
    event = output / "events.out.tfevents.test"
    event.write_bytes(b"forbidden")
    with pytest.raises(ValueError, match="logger output"):
        throughput.load_throughput_timing(output, spec)


def test_throughput_runner_rejects_nonsequential_boundary() -> None:
    runner = _runner()

    with pytest.raises(RuntimeError, match="start order"):
        runner._observe_iteration_start(1, throughput.NUM_ENVS)


def test_run_throughput_uses_canonical_builder_and_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract(tmp_path)
    output = tmp_path / "output"
    events = []
    build_call = {}

    class FakeRunner(throughput.BFMThroughputRunner):
        def prepare_timing(self) -> None:
            events.append("prepare")

        def learn(self, iterations: int) -> None:
            events.append(("learn", iterations))

        def write_timing(
            self,
            destination: Path,
            spec: throughput.ThroughputSpec,
        ) -> dict[str, object]:
            events.append(("write", destination, spec.model_profile))
            return {"written": True}

    runner = object.__new__(FakeRunner)
    env = SimpleNamespace(close=lambda: events.append("close"))

    def build(args, schedule, sink, **kwargs):
        build_call.update(
            {
                "args": args,
                "schedule": schedule,
                "sink": sink,
                **kwargs,
            }
        )
        return env, runner

    def initialize(built_runner, destination, schedule) -> None:
        assert built_runner is runner
        events.append(("initialize", destination, schedule.total_iterations))

    monkeypatch.setattr(throughput, "build_candidate_runner", build)
    monkeypatch.setattr(throughput, "initialize_candidate_runner", initialize)
    monkeypatch.setattr(
        throughput,
        "load_throughput_timing",
        lambda destination, spec: {
            "destination": destination,
            "profile": spec.model_profile,
        },
    )

    result = throughput.run_throughput(
        contract,
        output,
        "residual_6x1024",
    )

    assert build_call["runner_class"] is throughput.BFMThroughputRunner
    assert build_call["artifacts_enabled"] is False
    assert build_call["logging_enabled"] is False
    assert build_call["sink"] is None
    assert build_call["schedule"] == throughput.BFMTrainingSchedule(
        total_iterations=171,
        save_interval=9_375,
        save_initial_evaluation_checkpoint=False,
    )
    assert events == [
        ("initialize", output / "evaluation_checkpoints", 171),
        "prepare",
        ("learn", 171),
        ("write", output, "residual_6x1024"),
        "close",
    ]
    assert result == {
        "destination": output,
        "profile": "residual_6x1024",
    }
