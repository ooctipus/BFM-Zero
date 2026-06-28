"""Static checks for the BFM-Zero Phase 2 candidate entry point."""

import torch
from rsl_rl.models.forward_backward_model import ForwardBackwardModel
from tensordict import TensorDict

from phase2_adapter.candidate import candidate_config
from phase2_adapter.environment import BFM_AUXILIARY_EVIDENCE_NAMES, BFM_FIELD_WIDTHS
from phase2_adapter.policy import BFMCandidatePolicy


def test_candidate_uses_released_cadence_routes_and_compact_history() -> None:
    """The candidate should differ from source at learner ownership and exact finals only."""
    config = candidate_config(lambda *_args, **_kwargs: None, seed=4728)

    assert config["num_steps_per_env"] == 1
    assert config["num_updates_per_iteration"] == 16
    assert config["random_action_steps"] == 10_240
    assert config["model"]["normalization_type"] == "exponential"
    assert config["model"]["normalization_eps"] == 1e-5
    assert config["model"]["normalization_momentum"] == 0.01
    assert config["replay"]["capacity_steps"] == 5_000
    assert config["replay"]["terminal_capacity_per_env"] == 17
    assert config["replay"]["autoreset_mode"] == "same_step"
    assert config["replay"]["auxiliary_evidence_names"] == list(BFM_AUXILIARY_EVIDENCE_NAMES)
    assert config["replay"]["history_layout"]["history_length"] == 4
    assert config["expert"]["window_lengths"] == (8, 257)
    assert config["algorithm"]["rollout_expert_fraction"] == 0.5
    assert config["torch_compile_mode"] is None


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
