# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the canonical compiled BFM source throughput entrypoint."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from phase2_adapter import source_throughput as throughput


def _contract(tmp_path: Path) -> Path:
    reference = tmp_path / "reference.json"
    data = tmp_path / "motion.pkl"
    reference.write_text("{}\n")
    data.write_bytes(b"motion")
    payload = {
        "schema": throughput.SOURCE_THROUGHPUT_CONTRACT_SCHEMA,
        "shared_run_spec": dict(throughput._REQUIRED_SHARED_VALUES),
        "profile": {
            "model_profile": throughput.MODEL_PROFILE,
            "hidden_dim": 1_024,
            "hidden_layers": 6,
        },
        "timing_contract": {
            "boundary_hooks": throughput.BOUNDARY_HOOKS,
            "boundary_synchronizations": 2,
        },
        "training_inputs": {
            "reference_config": str(reference),
            "reference_config_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
            "data": str(data),
            "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
            "base_bfm_commit": throughput.TRAINING_BFM_COMMIT,
            "base_rsl_rl_commit": throughput.TRAINING_RSL_COMMIT,
        },
    }
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return contract


def _curriculum(output: Path) -> None:
    path = output / "curriculum_events/0.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}\n")


def _complete(observer: throughput.SourceThroughputObserver) -> None:
    for iteration in range(throughput.TOTAL_ITERATIONS):
        observer.observe_iteration_start(iteration, iteration * throughput.NUM_ENVS)
        observer.observe_iteration_learning_complete(iteration, (iteration + 1) * throughput.NUM_ENVS)
        observer.observe_iteration_complete(
            iteration,
            (iteration + 1) * throughput.NUM_ENVS,
            0.25,
            0.75,
        )


def test_source_throughput_contract_freezes_exact_role_and_window(tmp_path: Path) -> None:
    spec = throughput.load_source_throughput_spec(_contract(tmp_path))

    assert spec.contract_sha256 == hashlib.sha256(spec.contract.read_bytes()).hexdigest()
    assert throughput.TRAINING_BFM_COMMIT == "f0495e864ffcf332f346bd7b55e9aa108cb8b38f"
    assert throughput.TRAINING_RSL_COMMIT == "087a6051b6603e9543fdeaf209703261801902d5"
    assert throughput.PRE_UPDATE_ITERATIONS == 11
    assert throughput.MEASUREMENT_ITERATION_FIRST == 43
    assert throughput.MEASUREMENT_ITERATION_LAST == 170
    assert throughput.MEASUREMENT_TRANSITIONS == 131_072


def test_source_throughput_observer_uses_two_syncs_and_128_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synchronizations = []
    clock = iter((10.0, 12.0))
    monkeypatch.setattr(throughput.torch.cuda, "synchronize", synchronizations.append)
    monkeypatch.setattr(throughput.time, "monotonic", lambda: next(clock))
    observer = throughput.SourceThroughputObserver("cuda:0")
    output = tmp_path / "output"
    output.mkdir()
    _curriculum(output)

    _complete(observer)
    summary = observer.write(output, throughput.load_source_throughput_spec(_contract(tmp_path)))
    loaded = throughput.load_source_throughput_timing(output, throughput.load_source_throughput_spec(tmp_path / "contract.json"))

    assert synchronizations == [throughput.torch.device("cuda:0"), throughput.torch.device("cuda:0")]
    assert summary == loaded
    assert summary["measurement_wall_seconds"] == 2.0
    assert summary["total_transitions_per_second"] == 65_536.0
    assert len((output / throughput.TIMING_CSV_NAME).read_text().splitlines()) == 129


def test_source_throughput_calls_canonical_train_without_final_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract(tmp_path)
    output = tmp_path / "output"
    calls = []
    clock = iter((10.0, 12.0))
    monkeypatch.setattr(throughput.torch.cuda, "set_device", lambda device: calls.append(("device", device)))
    monkeypatch.setattr(throughput.torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(throughput.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(throughput, "set_seed_everywhere", lambda seed: calls.append(("seed", seed)))
    monkeypatch.setattr(throughput, "_load_config", lambda args, schedule: (args, schedule))
    monkeypatch.setattr(throughput, "Workspace", lambda config: config)

    def train(workspace, schedule, *, observer, save_final_checkpoint):
        calls.append(("train", workspace, schedule, save_final_checkpoint))
        _complete(observer)
        _curriculum(workspace[0].output_dir)

    monkeypatch.setattr(throughput, "_train", train)

    result = throughput.run_source_throughput(contract, output)

    train_call = calls[-1]
    assert train_call[0] == "train"
    assert train_call[1][0].compile is True
    assert train_call[2].total_iterations == 171
    assert train_call[2].save_interval == 9_375
    assert train_call[3] is False
    assert result["total_transitions_per_second"] == 65_536.0
    assert not (output / "checkpoint").exists()
    assert not (output / "evaluation_checkpoints").exists()


def test_source_throughput_rejects_training_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    contract = _contract(tmp_path)
    spec = throughput.load_source_throughput_spec(contract)
    output = tmp_path / "output"
    output.mkdir()
    _curriculum(output)
    clock = iter((10.0, 12.0))
    monkeypatch.setattr(throughput.torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(throughput.time, "monotonic", lambda: next(clock))
    observer = throughput.SourceThroughputObserver("cuda:0")
    _complete(observer)
    observer.write(output, spec)
    (output / "checkpoint.safetensors").write_bytes(b"forbidden")

    with pytest.raises(ValueError, match="forbidden training artifacts"):
        throughput.load_source_throughput_timing(output, spec)
