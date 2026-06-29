"""Static checks for the BFM-Zero Phase 2 candidate entry point."""

from types import SimpleNamespace

import pytest
import torch
from rsl_rl.models.forward_backward_model import ForwardBackwardModel
from rsl_rl.runners.off_policy_runner import OffPolicyRunner
from tensordict import TensorDict

from phase2_adapter.candidate import (
    _BFMEvaluationCheckpointRunner,
    _configure_training_runtime,
    candidate_config,
)
from phase2_adapter.environment import BFM_AUXILIARY_EVIDENCE_NAMES, BFM_FIELD_WIDTHS
from phase2_adapter.policy import BFMCandidatePolicy
from phase2_adapter.specification import BFM_MODEL_PROFILES


def test_candidate_matches_released_matmul_precision() -> None:
    """Candidate training should use the source's TF32-enabled float32 policy."""
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        _configure_training_runtime()
        assert torch.get_float32_matmul_precision() == "high"
    finally:
        torch.set_float32_matmul_precision(previous)


def test_candidate_uses_released_cadence_routes_and_compact_history() -> None:
    """The candidate should differ from source at learner ownership and exact finals only."""
    config = candidate_config(lambda *_args, **_kwargs: None, seed=4728, save_interval=9_375)

    assert config["num_steps_per_env"] == 1
    assert config["num_updates_per_iteration"] == 16
    assert config["random_action_steps"] == 10_240
    assert config["save_interval"] == 9_375
    assert config["model"]["normalization_type"] == "exponential"
    assert config["model"]["normalization_eps"] == 1e-5
    assert config["model"]["normalization_momentum"] == 0.01
    assert config["model"]["distribution_cfg"]["noise_clip"] == 0.3
    assert config["algorithm"]["fb_pessimism"] == 0.0
    assert config["replay"]["capacity_steps"] == 5_000
    assert config["replay"]["terminal_capacity_per_env"] == 17
    assert config["replay"]["sampling"] == "episode_uniform"
    assert config["replay"]["autoreset_mode"] == "same_step"
    discriminator_channel = next(channel for channel in config["replay"]["reward_channels"] if channel["name"] == "discriminator")
    assert discriminator_channel["timing"] == "state"
    assert config["replay"]["auxiliary_evidence_names"] == list(BFM_AUXILIARY_EVIDENCE_NAMES)
    assert config["replay"]["history_layout"]["history_length"] == 4
    assert config["replay"]["history_layout"]["last_action_field"] is None
    assert config["replay"]["history_layout"]["sources"][0]["observation_name"] == "last_action"
    assert config["expert"]["window_lengths"] == (8, 257)
    assert config["algorithm"]["rollout_expert_fraction"] == 0.5
    assert config["algorithm"]["random_action_range"] == (-5.0, 5.0)
    assert config["model"]["value_heads"][1]["spec"]["reward_composition"] == "scalar"
    assert config["torch_compile_mode"] is None


def test_candidate_resolves_semantic_capacity_profiles() -> None:
    """Each Pareto profile should change network capacity without changing topology semantics."""
    for name, (hidden_dim, hidden_layers) in BFM_MODEL_PROFILES.items():
        config = candidate_config(lambda *_args, **_kwargs: None, seed=4728, save_interval=9_375, model_profile=name)
        actor = config["model"]["actor_cfg"]
        forward = config["model"]["forward_cfg"]

        assert actor == {
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
            "embedding_layers": 2,
            "residual": True,
        }
        assert forward["hidden_dim"] == hidden_dim
        assert forward["hidden_layers"] == hidden_layers
        assert forward["embedding_layers"] == hidden_layers
        assert all(head["network"] == forward for head in config["model"]["value_heads"])

    with pytest.raises(ValueError, match="Unknown BFM model profile"):
        candidate_config(lambda *_args, **_kwargs: None, seed=4728, save_interval=9_375, model_profile="residual_unknown")


def test_candidate_policy_reuses_named_model_routes() -> None:
    """The native wrapper should translate dictionaries without owning model math."""
    observations = TensorDict(
        {name: torch.zeros(2, width) for name, width in BFM_FIELD_WIDTHS.items()},
        batch_size=[2],
    )
    routes = {
        "actor": ("state", "last_action", "history_actor"),
        "forward": ("state", "privileged_state", "last_action", "history_actor"),
        "backward": ("state", "privileged_state"),
    }
    config = {
        "context_dim": 4,
        "actor_cfg": {"hidden_dim": 16, "hidden_layers": 1, "embedding_layers": 2},
        "forward_cfg": {"hidden_dim": 16, "hidden_layers": 1, "embedding_layers": 2},
        "backward_hidden_dims": [8],
        "normalization_type": "none",
    }
    model = ForwardBackwardModel.from_config(observations, routes, 29, config)
    policy = BFMCandidatePolicy(model)
    native = {name: value.clone() for name, value in observations.items()}
    context = torch.randn(2, 4)

    assert policy._model is policy
    assert policy.backward_map(native).shape == (2, 4)
    assert policy.act(native, context).shape == (2, 29)


def test_candidate_keeps_compact_milestones_and_one_full_checkpoint(tmp_path, monkeypatch) -> None:
    """Intermediate evaluation points should not duplicate replay and optimizer state."""
    full_saves: list[tuple[str, dict | None]] = []

    def record_full_save(_runner, path: str, infos: dict | None = None) -> None:
        full_saves.append((path, infos))

    monkeypatch.setattr(OffPolicyRunner, "save", record_full_save)
    runner = object.__new__(_BFMEvaluationCheckpointRunner)
    runner.logger = SimpleNamespace(log_dir=str(tmp_path))
    runner.alg = SimpleNamespace(get_policy=lambda: torch.nn.Linear(2, 1))
    runner._final_transitions = 19_200_000
    curriculum_events: list[int] = []
    runner.curriculum_event = lambda: curriculum_events.append(runner.collected_transitions)
    runner.collected_transitions = 0
    runner.save_evaluation_checkpoint(tmp_path / "evaluation_checkpoints")
    initial = tmp_path / "evaluation_checkpoints" / "0.pt"

    assert initial.is_file()
    assert full_saves == []
    assert curriculum_events == [0]

    runner.collected_transitions = 9_600_000
    runner.save(str(tmp_path / "full.pt"))
    first = tmp_path / "evaluation_checkpoints" / "9600000.pt"

    assert first.is_file()
    assert "model_state_dict" in torch.load(first, weights_only=True)
    assert full_saves == []
    assert curriculum_events == [0, 9_600_000]

    runner.collected_transitions = 19_200_000
    runner.save(str(tmp_path / "full.pt"), {"final": True})
    second = tmp_path / "evaluation_checkpoints" / "19200000.pt"

    assert second.is_file()
    assert full_saves == [(str(tmp_path / "full.pt"), {"final": True})]
    assert curriculum_events == [0, 9_600_000, 19_200_000]
