"""Contract tests for the native BFM-Zero Phase 2 adapter."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.storage.forward_backward_replay import ForwardBackwardTransitionBatch
from tensordict import TensorDict

from phase2_adapter.environment import (
    BFM_ACTION_DIM,
    BFM_AUXILIARY_EVIDENCE_NAMES,
    BFM_FIELD_WIDTHS,
    BFMZeroVecEnv,
)
from phase2_adapter.evaluate_all_motions import evaluation_protocol
from phase2_adapter.evaluation import normalize_tracking_metrics
from phase2_adapter.source import _save_evaluation_checkpoint, _SourceReplay
from phase2_adapter.specification import replay_config


class _ActionSpace:
    shape = (2, BFM_ACTION_DIM)


def test_evaluation_protocol_keeps_randomization_and_noise_as_one_contract() -> None:
    """Stochastic and deterministic evaluation should be explicit paired protocols."""
    assert evaluation_protocol(False, False) == "native_stochastic"
    assert evaluation_protocol(True, True) == "deterministic"
    with pytest.raises(ValueError, match="enable or disable"):
        evaluation_protocol(True, False)


class _BaseEnv:
    def __init__(self) -> None:
        self.num_envs = 2
        self.max_episode_length = 300
        self.episode_length_buf = torch.zeros(2, dtype=torch.long)
        self.state = torch.zeros(2)
        self.last_actions = torch.zeros(2, BFM_ACTION_DIM)

    def _compute_observations(self) -> None:
        pass

    def reset_envs_idx(self, env_ids: torch.Tensor, *args, **kwargs) -> None:
        del args, kwargs
        self.state[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0


class _SameStepEnv:
    def __init__(self) -> None:
        self._env = _BaseEnv()
        self.num_envs = 2
        self.device = torch.device("cpu")
        self.action_space = _ActionSpace()
        self._creation_config = SimpleNamespace(lafan_tail_path="fixture")
        self.last_input_actions: torch.Tensor | None = None

    def reset(self, to_numpy: bool = False):
        assert not to_numpy
        self._env.reset_envs_idx(torch.arange(2))
        return self._get_g1env_observation(to_numpy=False), {}

    def step(self, actions: torch.Tensor, to_numpy: bool = False):
        assert not to_numpy
        self.last_input_actions = actions.clone()
        self._env.last_actions.copy_(actions)
        self._env.state += 1.0
        self._env.episode_length_buf += 1
        truncated = torch.tensor([False, True])
        terminated = torch.tensor([False, False])
        self._env.reset_envs_idx(truncated.nonzero().flatten())
        info = {"aux_rewards": {name: torch.full((2,), float(index)) for index, name in enumerate(BFM_AUXILIARY_EVIDENCE_NAMES)}}
        return self._get_g1env_observation(to_numpy=False), torch.ones(2), terminated, truncated, info

    def _get_g1env_observation(self, to_numpy: bool = False):
        assert not to_numpy
        state = self._env.state.unsqueeze(-1)
        return {
            "state": state.repeat(1, BFM_FIELD_WIDTHS["state"]),
            "last_action": self._env.last_actions.clone(),
            "history_actor": state.repeat(1, BFM_FIELD_WIDTHS["history_actor"]),
            "privileged_state": state.repeat(1, BFM_FIELD_WIDTHS["privileged_state"]),
        }

    def _get_qpos_qvel(self, to_numpy: bool = False):
        assert not to_numpy
        state = self._env.state.unsqueeze(-1)
        return state.repeat(1, 36), state.repeat(1, 35)

    def close(self) -> None:
        pass


def test_correct_terminal_captures_pre_reset_state_and_keeps_action_identity(monkeypatch) -> None:
    """Done rows should carry exact pre-reset fields while returned rows remain post-reset."""
    monkeypatch.setattr(torch.random, "fork_rng", lambda **_kwargs: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    native = _SameStepEnv()
    env = BFMZeroVecEnv(native, terminal_profile="correct_terminal", device="cpu")
    actions = torch.arange(2 * BFM_ACTION_DIM, dtype=torch.float32).reshape(2, BFM_ACTION_DIM)

    observations, _rewards, done, extras = env.step(actions)

    assert done.tolist() == [False, True]
    assert torch.all(observations["state"][1] == 0.0)
    assert extras["final_obs_valid"].tolist() == [False, True]
    assert torch.all(extras["final_obs"]["state"][1] == 1.0)
    assert torch.all(extras["final_obs"]["last_action"][1] == actions[1])
    assert torch.all(extras["final_qpos"][1] == 1.0)
    assert torch.equal(native.last_input_actions, actions)
    assert extras["auxiliary_reward_evidence"][0].tolist() == list(map(float, range(len(BFM_AUXILIARY_EVIDENCE_NAMES))))


def test_native_reference_is_explicitly_separate_from_correct_terminal() -> None:
    """The historical profile should never masquerade as an exact-final stream."""
    env = BFMZeroVecEnv(_SameStepEnv(), terminal_profile="native_reference", device="cpu")
    _observations, _rewards, _done, extras = env.step(torch.zeros(2, BFM_ACTION_DIM))

    assert env.cfg["terminal_profile"] == "native_reference"
    assert "final_obs" not in extras
    assert "final_obs_valid" not in extras


def test_reset_refreshes_observations_after_shared_curriculum_evaluation() -> None:
    """The wrapper should discard evaluator state before behavior collection resumes."""
    env = BFMZeroVecEnv(_SameStepEnv(), terminal_profile="native_reference", device="cpu")
    env.env._env.state.fill_(7.0)

    observations = env.reset()

    assert torch.all(observations["state"] == 0.0)


def test_replay_terminal_capacity_covers_every_live_timeout() -> None:
    """A 5,000-step ring needs seventeen slots for deterministic 300-step episodes."""
    config = replay_config(seed=0)

    assert config["terminal_capacity_per_env"] == 17
    assert config["sampling"] == "episode_uniform"
    assert config["history_layout"]["last_action_field"] is None
    assert config["history_layout"]["sources"][0]["observation_name"] == "last_action"


def test_source_replay_exposes_released_batch_names_from_exact_edges() -> None:
    """The source learner should consume the same compact logical transitions as RSL-RL."""
    observations = TensorDict(
        {name: torch.zeros(2, width) for name, width in BFM_FIELD_WIDTHS.items()},
        batch_size=[2],
    )
    replay = _SourceReplay(
        observations,
        num_envs=2,
        action_dim=BFM_ACTION_DIM,
        context_dim=4,
        device="cpu",
        seed=7,
    )
    current = observations
    for step in range(10):
        reached = current.clone()
        reached["state"].fill_(step + 1)
        replay.add(
            ForwardBackwardTransitionBatch(
                observations=current,
                next_observations=reached,
                final_observations=reached,
                actions=torch.full((2, BFM_ACTION_DIM), float(step)),
                behavior_context=torch.full((2, 4), float(step)),
                environment_reward=torch.full((2, 1), float(step)),
                auxiliary_reward_evidence=torch.full((2, len(BFM_AUXILIARY_EVIDENCE_NAMES)), float(step)),
                terminated=torch.zeros(2, 1, dtype=torch.bool),
                truncated=torch.zeros(2, 1, dtype=torch.bool),
                context_changed=torch.zeros(2, 1, dtype=torch.bool),
                action_applied=torch.ones(2, 1, dtype=torch.bool),
                final_observation_valid=torch.zeros(2, 1, dtype=torch.bool),
            )
        )
        current = reached

    sample_random = replay.storage.sample_random
    requested_batch_sizes = []

    def record_sample(batch_size: int):
        requested_batch_sizes.append(batch_size)
        return sample_random(batch_size)

    replay.storage.sample_random = record_sample
    batch = replay.sample(16)

    assert requested_batch_sizes == [16]
    assert batch["action"].shape == (16, BFM_ACTION_DIM)
    assert batch["z"].shape == (16, 4)
    assert batch["reward"].shape == (16, 1)
    assert set(batch["observation"]) == set(BFM_FIELD_WIDTHS)
    assert set(batch["next"]["observation"]) == set(BFM_FIELD_WIDTHS)
    assert set(batch["aux_rewards"]) == set(BFM_AUXILIARY_EVIDENCE_NAMES)

    reset_observations = observations.clone()
    reset_observations["state"].fill_(100.0)
    replay.process_env_reset(reset_observations)
    boundary = replay.storage.sample(torch.full((2,), 9), torch.arange(2))
    assert torch.all(boundary.valid)
    assert torch.all(boundary.truncated)
    assert torch.all(boundary.next_observations["state"] == 10.0)

    reached_after_reset = reset_observations.clone()
    reached_after_reset["state"].fill_(101.0)
    replay.add(
        ForwardBackwardTransitionBatch(
            observations=reset_observations,
            next_observations=reached_after_reset,
            final_observations=reached_after_reset,
            actions=torch.zeros(2, BFM_ACTION_DIM),
            behavior_context=torch.zeros(2, 4),
            environment_reward=torch.zeros(2, 1),
            auxiliary_reward_evidence=torch.zeros(2, len(BFM_AUXILIARY_EVIDENCE_NAMES)),
            terminated=torch.zeros(2, 1, dtype=torch.bool),
            truncated=torch.zeros(2, 1, dtype=torch.bool),
            context_changed=torch.zeros(2, 1, dtype=torch.bool),
            action_applied=torch.ones(2, 1, dtype=torch.bool),
            final_observation_valid=torch.zeros(2, 1, dtype=torch.bool),
        )
    )
    after_reset = replay.storage.sample(torch.full((2,), 10), torch.arange(2))
    assert torch.all(after_reset.valid)
    assert torch.all(after_reset.observations["state"] == 100.0)
    assert torch.all(after_reset.next_observations["state"] == 101.0)
    replay.assert_no_errors()


def test_source_milestone_checkpoint_creates_one_immutable_parent(tmp_path: Path) -> None:
    """The adapter should own parent creation before the research model saves."""

    class Model:
        def save(self, path: str) -> None:
            destination = Path(path)
            assert destination.parent.is_dir()
            destination.mkdir(exist_ok=True)
            (destination / "model.safetensors").touch()

    _save_evaluation_checkpoint(Model(), tmp_path, 12_288)

    assert (tmp_path / "evaluation_checkpoints" / "12288" / "model" / "model.safetensors").is_file()
    with pytest.raises(FileExistsError):
        _save_evaluation_checkpoint(Model(), tmp_path, 12_288)


def test_tracking_normalization_requires_all_native_motions_and_scalars() -> None:
    """BFM normalization should preserve native values and reject incomplete output."""
    metrics = {
        "motion-a": {"motion_id": 3, "motion_file": "motion-a", "emd": 0.2, "distance": 0.4},
        "motion-b": {"motion_id": 4, "motion_file": "motion-b", "emd": 0.3, "distance": 0.5},
    }
    rows = normalize_tracking_metrics(
        metrics,
        implementation="released",
        training_seed=4728,
        evaluation_seed=1,
        checkpoint_transition=211_200_000,
        terminal_profile="native_reference",
        run_id="fixture",
        evaluator_hash="eval",
        dataset_hash="data",
        expected_motion_count=2,
    )

    assert len(rows) == 4
    assert {row["metric_value"] for row in rows if row["metric_name"] == "emd"} == {0.2, 0.3}
    with pytest.raises(ValueError, match="Expected 2"):
        normalize_tracking_metrics(
            {"motion-a": metrics["motion-a"]},
            implementation="released",
            training_seed=4728,
            evaluation_seed=1,
            checkpoint_transition=211_200_000,
            terminal_profile="native_reference",
            run_id="fixture",
            evaluator_hash="eval",
            dataset_hash="data",
            expected_motion_count=2,
        )
