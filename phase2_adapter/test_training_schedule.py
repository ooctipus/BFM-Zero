"""Tests for the shared source/candidate transition schedule."""

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from phase2_adapter import source
from phase2_adapter.candidate import (
    BFMEvaluationCheckpointRunner,
    candidate_config,
    initialize_candidate_runner,
)
from phase2_adapter.specification import resolve_training_schedule


def test_training_schedule_derives_shared_iteration_cadence() -> None:
    """Source transition cadence and candidate iteration cadence should be one contract."""
    schedule = resolve_training_schedule(
        transitions=28_800_000,
        num_envs=1_024,
        evaluation_checkpoint_every_transitions=9_600_000,
        save_initial_evaluation_checkpoint=False,
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
            save_initial_evaluation_checkpoint=False,
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
        save_initial_evaluation_checkpoint=False,
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
    assert loaded.agent.model.archi.f.hidden_dim == 1_024
    assert loaded.agent.model.archi.f.hidden_layers == 6
    assert loaded.agent.model.archi.actor.hidden_dim == 1_024
    assert loaded.agent.model.archi.critic.hidden_dim == 1_024
    assert loaded.agent.model.archi.aux_critic.hidden_dim == 1_024


@pytest.mark.parametrize("save_initial", (False, True))
def test_source_transition_zero_curriculum_and_policy_follow_schedule(tmp_path, monkeypatch, save_initial) -> None:
    """Source should always run its initial curriculum and publish only when requested."""
    events = []
    checkpoints = []
    monkeypatch.setattr(source, "load_expert_trajectories_from_motion_lib", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(source, "_source_curriculum_event", lambda *_args, transition, **_kwargs: events.append(transition))
    monkeypatch.setattr(source, "_save_evaluation_checkpoint", lambda *_args, **_kwargs: checkpoints.append(_args[2]))
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

    source._train(
        workspace,
        resolve_training_schedule(
            transitions=1,
            num_envs=1,
            evaluation_checkpoint_every_transitions=1,
            save_initial_evaluation_checkpoint=save_initial,
        ),
    )

    assert events == [0]
    assert checkpoints == ([0] if save_initial else [])


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


def test_training_schedule_rejects_nonboolean_initial_checkpoint_flag() -> None:
    """The shared transition-zero choice must be an explicit boolean."""
    with pytest.raises(ValueError, match="must be boolean"):
        resolve_training_schedule(
            transitions=1,
            num_envs=1,
            evaluation_checkpoint_every_transitions=1,
            save_initial_evaluation_checkpoint=0,
        )


@pytest.mark.parametrize(
    ("save_initial", "expected"),
    (
        (False, ("curriculum",)),
        (True, ("checkpoint", "curriculum")),
    ),
)
def test_candidate_transition_zero_curriculum_and_policy_follow_schedule(tmp_path, save_initial, expected) -> None:
    """Candidate should implement the same transition-zero policy as source."""
    events = []
    runner = SimpleNamespace(
        _artifact_dir=tmp_path,
        _artifacts_enabled=True,
        _last_published_transition=None,
        _milestone_sink=None,
        publish_evaluation_checkpoint=lambda _destination: events.append("checkpoint"),
        curriculum_event=lambda: events.append("curriculum"),
    )
    schedule = resolve_training_schedule(
        transitions=1,
        num_envs=1,
        evaluation_checkpoint_every_transitions=1,
        save_initial_evaluation_checkpoint=save_initial,
    )

    initialize_candidate_runner(runner, tmp_path, schedule)

    assert tuple(events) == expected


def test_source_and_candidate_publish_through_the_same_sink_contract(tmp_path) -> None:
    """Both learner call sites should provide one transition and one regular tensor file."""
    publications = []

    class Sink:
        def publish(self, transition: int, export) -> Path:
            destination = tmp_path / f"{len(publications)}.pt"
            export(destination)
            publications.append((transition, destination))
            return destination

    sink = Sink()
    source_model = SimpleNamespace(
        _actor=torch.nn.Linear(3, 2),
        _backward_map=torch.nn.Linear(3, 2),
        _obs_normalizer=torch.nn.Linear(3, 3),
    )
    candidate_model = SimpleNamespace(
        actor_network=torch.nn.Linear(3, 2),
        backward_network=torch.nn.Linear(3, 2),
        observation_normalizers=torch.nn.ModuleDict({"state": torch.nn.Linear(3, 3)}),
        action_distribution=torch.nn.Linear(2, 2),
    )
    source_path = source._save_evaluation_checkpoint(source_model, tmp_path, 7, sink)
    runner = object.__new__(BFMEvaluationCheckpointRunner)
    runner._milestone_sink = sink
    runner.collected_transitions = 7
    runner.alg = SimpleNamespace(get_policy=lambda: candidate_model)
    candidate_path = runner.publish_evaluation_checkpoint(tmp_path)

    assert publications == [(7, source_path), (7, candidate_path)]
    assert source_path.is_file()
    assert candidate_path.is_file()
