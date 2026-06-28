"""Static checks for the BFM-Zero Phase 2 candidate entry point."""

from phase2_adapter.candidate import candidate_config
from phase2_adapter.environment import BFM_AUXILIARY_EVIDENCE_NAMES


def test_candidate_uses_released_cadence_routes_and_compact_history() -> None:
    """The candidate should differ from source at learner ownership and exact finals only."""
    config = candidate_config(lambda *_args, **_kwargs: None, seed=4728)

    assert config["num_steps_per_env"] == 1
    assert config["num_updates_per_iteration"] == 16
    assert config["random_action_steps"] == 10_240
    assert config["replay"]["capacity_steps"] == 5_000
    assert config["replay"]["terminal_capacity_per_env"] == 16
    assert config["replay"]["autoreset_mode"] == "same_step"
    assert config["replay"]["auxiliary_evidence_names"] == list(BFM_AUXILIARY_EVIDENCE_NAMES)
    assert config["replay"]["history_layout"]["history_length"] == 4
    assert config["expert"]["window_lengths"] == (8, 257)
    assert config["algorithm"]["rollout_expert_fraction"] == 0.5
    assert config["torch_compile_mode"] is None
