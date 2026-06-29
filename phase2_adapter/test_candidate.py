"""Static checks for the BFM-Zero Phase 2 candidate entry point."""

from types import SimpleNamespace

import pytest
import safetensors.torch
import torch
from rsl_rl.models.forward_backward_model import ForwardBackwardModel
from rsl_rl.runners.off_policy_runner import OffPolicyRunner
from tensordict import TensorDict

import phase2_adapter.candidate as candidate_module
from phase2_adapter.candidate import (
    BFMEvaluationCheckpointRunner,
    _configure_training_runtime,
    build_candidate_runner,
    candidate_config,
)
from phase2_adapter.environment import BFM_AUXILIARY_EVIDENCE_NAMES, BFM_FIELD_WIDTHS
from phase2_adapter.policy import BFMCandidatePolicy
from phase2_adapter.specification import BFM_MODEL_PROFILES


class _CandidateInferenceModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor_network = torch.nn.Linear(3, 2)
        self.backward_network = torch.nn.Linear(3, 2)
        self.observation_normalizers = torch.nn.ModuleDict({"state": torch.nn.Linear(3, 3)})
        self.action_distribution = torch.nn.Linear(2, 2)


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
    runner = object.__new__(BFMEvaluationCheckpointRunner)
    runner.logger = SimpleNamespace(log_dir=str(tmp_path))
    runner._artifact_dir = tmp_path
    runner.alg = SimpleNamespace(get_policy=_CandidateInferenceModel)
    runner._final_transitions = 19_200_000
    runner._milestone_sink = None
    runner._artifacts_enabled = True
    runner._last_published_transition = None
    curriculum_events: list[int] = []
    runner.curriculum_event = lambda: curriculum_events.append(runner.collected_transitions)
    runner.collected_transitions = 0
    runner.publish_evaluation_checkpoint(tmp_path / "evaluation_checkpoints")
    runner.curriculum_event()
    initial = tmp_path / "evaluation_checkpoints" / "0.pt"

    assert initial.is_file()
    assert full_saves == []
    assert curriculum_events == [0]

    runner.collected_transitions = 9_600_000
    runner.save(str(tmp_path / "full.pt"))
    first = tmp_path / "evaluation_checkpoints" / "9600000.pt"

    assert first.is_file()
    assert {name.split(".", 1)[0] for name in safetensors.torch.load_file(first)} == {
        "action_distribution",
        "actor",
        "backward",
        "normalizers",
    }
    assert full_saves == []
    assert curriculum_events == [0, 9_600_000]

    runner.collected_transitions = 19_200_000
    runner.save(str(tmp_path / "full.pt"), {"final": True})
    second = tmp_path / "evaluation_checkpoints" / "19200000.pt"

    assert second.is_file()
    assert full_saves == [(str(tmp_path / "full.pt"), {"final": True})]
    assert curriculum_events == [0, 9_600_000, 19_200_000]


def test_candidate_can_disable_all_artifacts_without_disabling_curriculum(tmp_path, monkeypatch) -> None:
    """Throughput observation should retain training semantics without tensor exports."""
    full_saves = []
    monkeypatch.setattr(OffPolicyRunner, "save", lambda *_args, **_kwargs: full_saves.append(True))
    runner = object.__new__(BFMEvaluationCheckpointRunner)
    runner.logger = SimpleNamespace(log_dir=str(tmp_path))
    runner._artifact_dir = tmp_path
    runner._final_transitions = 1
    runner._milestone_sink = None
    runner._artifacts_enabled = False
    runner._last_published_transition = None
    runner.collected_transitions = 1
    runner.alg = SimpleNamespace(get_policy=_CandidateInferenceModel)
    curriculum_events = []
    runner.curriculum_event = lambda: curriculum_events.append(runner.collected_transitions)
    runner.publish_evaluation_checkpoint = lambda _destination: (_ for _ in ()).throw(
        AssertionError("artifact publication was not suppressed")
    )

    runner.save(str(tmp_path / "full.pt"))

    assert curriculum_events == [1]
    assert full_saves == []
    assert runner._last_published_transition == 1
    assert tuple(tmp_path.iterdir()) == ()


def test_candidate_builder_constructs_declared_runner_subclass(tmp_path, monkeypatch) -> None:
    """Timing observation should subclass the canonical runner without reproducing setup."""
    env = object()
    seeds = []
    constructions = []
    monkeypatch.setattr(candidate_module, "_configure_training_runtime", lambda: None)
    monkeypatch.setattr(candidate_module.torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(candidate_module, "set_seed_everywhere", seeds.append)
    monkeypatch.setattr(candidate_module, "make_native_environment", lambda **_kwargs: env)

    class TimingRunner(BFMEvaluationCheckpointRunner):
        def __init__(self, *args, **kwargs) -> None:
            constructions.append((args, kwargs))

    args = SimpleNamespace(
        device="cuda:0",
        seed=4728,
        reference_config=tmp_path / "reference.json",
        data_path=tmp_path / "lafan.pkl",
        num_envs=1_024,
        output_dir=tmp_path / "output",
        transitions=211_200_000,
        model_profile="residual_6x1024",
    )
    schedule = SimpleNamespace(save_interval=9_375)
    built_env, runner = build_candidate_runner(
        args,
        schedule,
        None,
        runner_class=TimingRunner,
        artifacts_enabled=False,
        logging_enabled=False,
    )

    assert built_env is env
    assert isinstance(runner, TimingRunner)
    assert seeds == [4728, 4728]
    assert constructions[0][0][0] is env
    assert constructions[0][1]["final_transitions"] == 211_200_000
    assert constructions[0][1]["artifact_dir"] == tmp_path / "output"
    assert constructions[0][1]["log_dir"] is None
    assert constructions[0][1]["milestone_sink"] is None
    assert constructions[0][1]["artifacts_enabled"] is False
