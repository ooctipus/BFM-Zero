"""Tests for the shared source/candidate transition schedule."""

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from phase2_adapter import source
from phase2_adapter.candidate import candidate_config
from phase2_adapter.specification import resolve_training_schedule


def test_training_schedule_derives_shared_iteration_cadence() -> None:
    """Source transition cadence and candidate iteration cadence should be one contract."""
    schedule = resolve_training_schedule(
        transitions=28_800_000,
        num_envs=1_024,
        evaluation_checkpoint_every_transitions=9_600_000,
    )

    assert schedule.total_iterations == 28_125
    assert schedule.save_interval == 9_375


@pytest.mark.parametrize(
    ("transitions", "num_envs", "cadence", "message"),
    (
        (0, 1_024, 9_600_000, "positive"),
        (28_800_000, 0, 9_600_000, "positive"),
        (28_800_000, 1_024, 0, "positive"),
        (28_800_001, 1_024, 9_600_000, "transitions must be divisible"),
        (28_800_000, 1_024, 9_600_001, "cadence must be divisible"),
        (28_800_000, 1_024, 19_200_000, "final transition must align"),
    ),
)
def test_training_schedule_rejects_nonintegral_or_final_misaligned_contracts(
    transitions: int,
    num_envs: int,
    cadence: int,
    message: str,
) -> None:
    """The final policy must be an exact scheduled evaluation checkpoint."""
    with pytest.raises(ValueError, match=message):
        resolve_training_schedule(
            transitions=transitions,
            num_envs=num_envs,
            evaluation_checkpoint_every_transitions=cadence,
        )


class _Node(SimpleNamespace):
    def model_copy(self, *, update: dict[str, object]):
        return _Node(**{**vars(self), **update})


def test_source_and_candidate_consume_the_same_derived_schedule(tmp_path, monkeypatch) -> None:
    """Both bridge call sites should consume schedule values instead of hidden constants."""
    schedule = resolve_training_schedule(
        transitions=28_800_000,
        num_envs=1_024,
        evaluation_checkpoint_every_transitions=9_600_000,
    )
    candidate = candidate_config(lambda *_args, **_kwargs: None, seed=4728, save_interval=schedule.save_interval)
    assert candidate["save_interval"] == schedule.save_interval

    architecture = _Node(f=_Node(), actor=_Node(), critic=_Node(), aux_critic=_Node())
    config = _Node(env=_Node(), agent=_Node(model=_Node(archi=architecture)))
    monkeypatch.setattr(source.TrainConfig, "model_validate_json", lambda _payload: config)
    reference = tmp_path / "reference.json"
    data = tmp_path / "lafan.pkl"
    reference.write_text("{}")
    data.write_bytes(b"motion")
    args = Namespace(
        reference_config=reference,
        data_path=data,
        output_dir=tmp_path / "output",
        transitions=28_800_000,
        seed=4728,
        num_envs=1_024,
        device="cuda:0",
        log_every_transitions=384_000,
        compile=True,
        model_profile="residual_6x1024",
    )

    loaded = source._load_config(args, schedule)

    assert loaded.num_env_steps == 28_800_000
    assert loaded.checkpoint_every_steps == 9_600_000


def test_source_transition_zero_curriculum_does_not_serialize_unused_policy(tmp_path, monkeypatch) -> None:
    """Source should run its initial curriculum without creating an out-of-contract checkpoint."""
    events = []
    monkeypatch.setattr(source, "load_expert_trajectories_from_motion_lib", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(source, "_source_curriculum_event", lambda *_args, transition, **_kwargs: events.append(transition))
    monkeypatch.setattr(
        source,
        "BFMZeroVecEnv",
        lambda *_args, **_kwargs: SimpleNamespace(get_observations=lambda: {}, close=lambda: None),
    )
    monkeypatch.setattr(
        source,
        "_SourceReplay",
        lambda *_args, **_kwargs: SimpleNamespace(assert_no_errors=lambda: None),
    )
    agent = SimpleNamespace(_model=object(), device="cpu", save=lambda _path: None)
    config = SimpleNamespace(
        agent=SimpleNamespace(model=SimpleNamespace(archi=SimpleNamespace(z_dim=256))),
        buffer_device="cpu",
        online_parallel_envs=1,
        env=SimpleNamespace(device="cpu"),
        num_env_steps=0,
        seed=4728,
    )
    workspace = SimpleNamespace(
        cfg=config,
        agent=agent,
        train_env=SimpleNamespace(_env=object()),
        work_dir=tmp_path,
        action_dim=29,
    )

    source._train(workspace)

    assert events == [0]
    assert not (tmp_path / "evaluation_checkpoints").exists()


def test_source_checkpoint_publication_is_atomic_and_rejects_stale_targets(tmp_path) -> None:
    """Readers should see either no source milestone or one complete final directory."""
    final = tmp_path / "evaluation_checkpoints/9600000"

    class Model:
        def save(self, path: str) -> None:
            destination = Path(path)
            assert destination.parent.name == ".9600000.staging"
            assert not final.exists()
            destination.mkdir()
            (destination / "model.safetensors").write_bytes(b"complete")

    source._save_evaluation_checkpoint(Model(), tmp_path, 9_600_000)
    assert (final / "model/model.safetensors").read_bytes() == b"complete"
    assert not (final.parent / ".9600000.staging").exists()
    with pytest.raises(FileExistsError, match="target is not empty"):
        source._save_evaluation_checkpoint(Model(), tmp_path, 9_600_000)

    stale = tmp_path / "evaluation_checkpoints/.19200000.staging"
    stale.mkdir()
    with pytest.raises(FileExistsError, match="target is not empty"):
        source._save_evaluation_checkpoint(Model(), tmp_path, 19_200_000)


def test_source_checkpoint_failure_never_publishes_partial_directory(tmp_path) -> None:
    """A normal save failure should remove staging and leave the final milestone absent."""

    class FailingModel:
        def save(self, path: str) -> None:
            destination = Path(path)
            destination.mkdir()
            (destination / "partial").write_bytes(b"partial")
            raise RuntimeError("save failed")

    with pytest.raises(RuntimeError, match="save failed"):
        source._save_evaluation_checkpoint(FailingModel(), tmp_path, 9_600_000)

    checkpoints = tmp_path / "evaluation_checkpoints"
    assert not (checkpoints / "9600000").exists()
    assert not (checkpoints / ".9600000.staging").exists()
