"""Shared BFM transition representation at the source/candidate bridge."""

from __future__ import annotations

from dataclasses import dataclass

from .environment import BFM_AUXILIARY_EVIDENCE_NAMES

BFM_MODEL_PROFILE_DEFAULT = "residual_6x2048"


@dataclass(frozen=True, slots=True)
class BFMTrainingSchedule:
    """Exact vector-step schedule shared by source and candidate learners."""

    total_iterations: int
    save_interval: int
    save_initial_evaluation_checkpoint: bool


def resolve_training_schedule(
    *,
    transitions: int,
    num_envs: int,
    evaluation_checkpoint_every_transitions: int,
    save_initial_evaluation_checkpoint: bool,
) -> BFMTrainingSchedule:
    """Convert transition counts into one exact source/candidate iteration schedule."""
    if not isinstance(save_initial_evaluation_checkpoint, bool):
        raise ValueError("save_initial_evaluation_checkpoint must be boolean.")
    values = (transitions, num_envs, evaluation_checkpoint_every_transitions)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in values):
        raise ValueError("transitions, num_envs, and evaluation checkpoint cadence must be positive integers.")
    if transitions % num_envs:
        raise ValueError("transitions must be divisible by num_envs.")
    if evaluation_checkpoint_every_transitions % num_envs:
        raise ValueError("evaluation checkpoint cadence must be divisible by num_envs.")
    if transitions % evaluation_checkpoint_every_transitions:
        raise ValueError("final transition must align with evaluation checkpoint cadence.")
    return BFMTrainingSchedule(
        total_iterations=transitions // num_envs,
        save_interval=evaluation_checkpoint_every_transitions // num_envs,
        save_initial_evaluation_checkpoint=save_initial_evaluation_checkpoint,
    )


BFM_MODEL_PROFILES = {
    "residual_6x2048": (2048, 6),
    "residual_6x1024": (1024, 6),
    "residual_3x1024": (1024, 3),
}


def resolve_model_profile(name: str) -> tuple[int, int]:
    """Return hidden width and residual-block count for one BFM profile."""
    try:
        return BFM_MODEL_PROFILES[name]
    except KeyError as error:
        choices = ", ".join(BFM_MODEL_PROFILES)
        raise ValueError(f"Unknown BFM model profile {name!r}; expected one of: {choices}.") from error


def observation_routes() -> dict[str, list[str]]:
    """Return the released asymmetric observation routes."""
    return {
        "actor": ["state", "last_action", "history_actor"],
        "forward": ["state", "privileged_state", "last_action", "history_actor"],
        "backward": ["state", "privileged_state"],
        "discriminator": ["state", "privileged_state"],
        "critic_discriminator": ["state", "privileged_state", "last_action", "history_actor"],
        "critic_auxiliary": ["state", "privileged_state", "last_action", "history_actor"],
    }


def replay_config(seed: int) -> dict[str, object]:
    """Return one compact exact-terminal replay specification for both learners."""
    reward_channels = [
        {
            "name": "environment",
            "provider_name": "environment",
            "source": "environment",
            "timing": "transition",
            "context_dependent": False,
            "sign": 1,
        },
        {
            "name": "discriminator",
            "provider_name": "discriminator",
            "source": "recomputed",
            "timing": "state",
            "context_dependent": True,
            "sign": 1,
        },
    ]
    reward_channels.extend(
        {
            "name": name,
            "provider_name": name,
            "source": "stored_evidence",
            "timing": "transition",
            "context_dependent": False,
            "sign": -1,
        }
        for name in BFM_AUXILIARY_EVIDENCE_NAMES
    )
    return {
        "class_name": "rsl_rl.storage.forward_backward_replay:ForwardBackwardReplay",
        "capacity_steps": 5_000,
        "terminal_capacity_per_env": 17,
        "sampling": "episode_uniform",
        "autoreset_mode": "same_step",
        "environment_reward_name": "environment",
        "auxiliary_evidence_names": list(BFM_AUXILIARY_EVIDENCE_NAMES),
        "reward_channels": reward_channels,
        "history_layout": {
            "history_field": "history_actor",
            "history_length": 4,
            "last_action_field": None,
            "include_seed_observations": False,
            "sources": [
                {"observation_name": "last_action", "start": 0, "stop": 29},
                {"observation_name": "state", "start": 61, "stop": 64},
                {"observation_name": "state", "start": 0, "stop": 29},
                {"observation_name": "state", "start": 29, "stop": 58},
                {"observation_name": "state", "start": 58, "stop": 61},
            ],
        },
        "seed": seed,
    }
