"""Tests for the shared Phase 2F compact-state milestone sink."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import safetensors.torch
import torch

from phase2_adapter import compact_state
from phase2_adapter.milestone import (
    MilestoneSink,
    _rng_fingerprint,
    replace_recovery_directory,
    replace_recovery_file,
)


class _Lifecycle:
    def __init__(self, ready: list[bool] | None = None) -> None:
        self.ready = list(ready or [True])
        self.publications: list[dict[str, object]] = []

    def checkpoint_slot_ready(self, run_root: Path, previous: int | None) -> bool:
        del run_root, previous
        return self.ready.pop(0) if len(self.ready) > 1 else self.ready[0]

    def publish_checkpoint(self, run_root: Path, **payload) -> None:
        self.publications.append({"run_root": run_root, **payload})

    def load_checkpoint_handoff(self, run_root: Path, transition: int) -> dict[str, object]:
        del run_root
        return {
            "producer": {
                "run_id": "run",
                "transition": transition,
                "previous_transition": None,
                "final_transition": transition,
            }
        }


class _SourceModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._actor = torch.nn.Linear(3, 2)
        self._backward_map = torch.nn.Linear(3, 2)
        self._obs_normalizer = torch.nn.Linear(3, 3)
        self.unused_critic = torch.nn.Linear(3, 1)


class _CandidateModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor_network = torch.nn.Linear(3, 2)
        self.backward_network = torch.nn.Linear(3, 2)
        self.observation_normalizers = torch.nn.ModuleDict({"state": torch.nn.Linear(3, 3)})
        self.action_distribution = torch.nn.Linear(2, 2)
        self.unused_forward = torch.nn.Linear(3, 3)


def _assert_state_equal(left: torch.nn.Module, right: torch.nn.Module) -> None:
    assert left.state_dict().keys() == right.state_dict().keys()
    for name, tensor in left.state_dict().items():
        assert torch.equal(tensor, right.state_dict()[name])


def _sink(tmp_path: Path, lifecycle: _Lifecycle, transitions: tuple[int, ...] = (0,)) -> MilestoneSink:
    root = (tmp_path / "run").resolve()
    root.mkdir(parents=True)
    return MilestoneSink(root, "run", transitions, lifecycle=lifecycle, sleep=lambda _seconds: None)


def test_sink_applies_backpressure_before_materializing_next_state(tmp_path: Path) -> None:
    lifecycle = _Lifecycle([False, False, True])
    sink = _sink(tmp_path, lifecycle, (0, 10))
    events = []
    sink.sleep = lambda _seconds: events.append("wait")

    path = sink.publish(10, lambda destination: (events.append("export"), destination.write_bytes(b"state")))

    assert events == ["wait", "wait", "export"]
    assert path.read_bytes() == b"state"
    assert lifecycle.publications[0]["previous_transition"] == 0


def test_sink_rejects_export_that_changes_training_rng(tmp_path: Path) -> None:
    lifecycle = _Lifecycle()
    sink = _sink(tmp_path, lifecycle)
    original = torch.get_rng_state()

    def export(destination: Path) -> None:
        torch.rand(1)
        destination.write_bytes(b"state")

    try:
        try:
            sink.publish(0, export)
        except RuntimeError as error:
            assert "RNG" in str(error)
        else:
            raise AssertionError("RNG-changing export was accepted")
    finally:
        torch.set_rng_state(original)

    assert lifecycle.publications == []
    assert (sink.run_root / "compact_states/0.safetensors.RNG_CHANGED").is_file()


def test_source_and_candidate_export_one_file_without_rng_change(tmp_path: Path) -> None:
    source_lifecycle = _Lifecycle()
    source_sink = _sink(tmp_path / "source", source_lifecycle)
    source_model = _SourceModel()
    source_rng = _rng_fingerprint()

    source_path = source_sink.publish(0, lambda destination: compact_state.export_source_state(source_model, destination))

    assert _rng_fingerprint() == source_rng
    source_loaded = _SourceModel()
    safetensors.torch.load_model(compact_state._source_inference_modules(source_loaded), source_path)
    assert source_path.is_file()
    _assert_state_equal(
        compact_state._source_inference_modules(source_model),
        compact_state._source_inference_modules(source_loaded),
    )
    assert {name.split(".", 1)[0] for name in safetensors.torch.load_file(source_path)} == {
        "actor",
        "backward",
        "normalizer",
    }

    candidate_lifecycle = _Lifecycle()
    candidate_sink = _sink(tmp_path / "candidate", candidate_lifecycle)
    candidate_model = _CandidateModel()
    candidate_rng = _rng_fingerprint()

    candidate_path = candidate_sink.publish(
        0,
        lambda destination: compact_state.export_candidate_state(candidate_model, destination),
    )

    assert _rng_fingerprint() == candidate_rng
    candidate_loaded = _CandidateModel()
    compact_state.load_candidate_state(candidate_loaded, candidate_path, "cpu")
    assert candidate_path.is_file()
    _assert_state_equal(
        compact_state._candidate_inference_modules(candidate_model),
        compact_state._candidate_inference_modules(candidate_loaded),
    )
    assert {name.split(".", 1)[0] for name in safetensors.torch.load_file(candidate_path)} == {
        "action_distribution",
        "actor",
        "backward",
        "normalizers",
    }


def test_source_single_file_loader_rebuilds_declared_topology(tmp_path: Path, monkeypatch) -> None:
    source_model = _SourceModel()
    checkpoint = tmp_path / "source.pt"
    compact_state.export_source_state(source_model, checkpoint)
    reference_config = tmp_path / "reference.json"
    reference_config.write_text("{}")
    built_spaces = []

    def build(observation_space, action_dim: int):
        built_spaces.append((observation_space, action_dim))
        return _SourceModel()

    config = object()
    profile_calls = []

    def select_model_config(config_value, profile: str, device: str):
        profile_calls.append((config_value, profile, device))
        return SimpleNamespace(build=build)

    monkeypatch.setattr(compact_state, "source_model_config", select_model_config)
    train_module = SimpleNamespace(TrainConfig=SimpleNamespace(model_validate_json=lambda _payload: config))
    monkeypatch.setitem(sys.modules, "humanoidverse.train", train_module)
    observation_space = SimpleNamespace(spaces={"state": object(), "time": object()})

    loaded = compact_state.load_source_state(
        reference_config,
        checkpoint,
        observation_space,
        2,
        "cpu",
        "residual_6x1024",
    )

    assert profile_calls == [(config, "residual_6x1024", "cpu")]
    assert set(observation_space.spaces) == {"state", "time"}
    assert set(built_spaces[0][0].spaces) == {"state"}
    assert built_spaces[0][1] == 2
    _assert_state_equal(
        compact_state._source_inference_modules(source_model),
        compact_state._source_inference_modules(loaded),
    )


def test_rolling_recovery_replaces_one_owned_state(tmp_path: Path) -> None:
    file_path = tmp_path / "candidate/recovery.pt"
    replace_recovery_file(file_path, lambda destination: destination.write_bytes(b"first"))
    replace_recovery_file(file_path, lambda destination: destination.write_bytes(b"second"))
    assert file_path.read_bytes() == b"second"
    assert not file_path.with_suffix(".tmp").exists()

    directory = tmp_path / "source/recovery"

    def export(value: bytes):
        def write(destination: Path) -> None:
            destination.mkdir(parents=True)
            (destination / "state").write_bytes(value)

        return write

    replace_recovery_directory(directory, export(b"first"))
    replace_recovery_directory(directory, export(b"second"))
    assert (directory / "state").read_bytes() == b"second"
    assert not directory.with_name(".recovery.previous").exists()
    assert not directory.with_name(".recovery.staging").exists()


def test_sink_resumes_an_existing_handoff_before_waiting_on_its_predecessor(tmp_path: Path) -> None:
    """A resumed producer should revalidate its own publication instead of deadlocking."""
    lifecycle = _Lifecycle()
    sink = _sink(tmp_path, lifecycle)
    destination = sink.run_root / "compact_states/0.safetensors"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"state")
    (sink.run_root / "checkpoint_handoffs/transition_0").mkdir(parents=True)
    lifecycle.checkpoint_slot_ready = lambda *_args: (_ for _ in ()).throw(
        AssertionError("an existing handoff re-entered predecessor backpressure")
    )

    resumed = sink.publish(
        0,
        lambda _destination: (_ for _ in ()).throw(AssertionError("an existing handoff was exported again")),
    )

    assert resumed == destination
