"""Build the RSL-RL candidate on the unchanged native BFM-Zero environment."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from rsl_rl.runners.off_policy_runner import OffPolicyRunner

from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig

from .environment import BFM_AUXILIARY_EVIDENCE_NAMES, BFMZeroVecEnv
from .expert import BFMZeroExpertProvider


def make_native_environment(
    *,
    reference_config: str | Path,
    data_path: str | Path,
    num_envs: int,
    device: str,
) -> BFMZeroVecEnv:
    """Create the source BFM distribution with exact pre-reset final capture."""
    with Path(reference_config).open() as stream:
        config = json.load(stream)
    env_options = copy.deepcopy(config["env"])
    env_options["device"] = device
    env_options["lafan_tail_path"] = str(Path(data_path).resolve())
    native = HumanoidVerseIsaacConfig(**env_options).build(num_envs)[0]
    return BFMZeroVecEnv(native, terminal_profile="correct_terminal", device=device)


def candidate_config(expert_provider: BFMZeroExpertProvider, *, seed: int) -> dict[str, Any]:
    """Return the released-scale BFM configuration with exact-terminal collection."""
    actor = {"hidden_dim": 2048, "hidden_layers": 6, "embedding_layers": 2, "residual": True}
    value = {"hidden_dim": 2048, "hidden_layers": 6, "embedding_layers": 6, "residual": True}
    magnitudes = (0.0, 0.1, 10.0, 0.0, 1.0, 0.4, 4.0, 2.0)
    routes = {
        "actor": ["state", "last_action", "history_actor"],
        "forward": ["state", "privileged_state", "last_action", "history_actor"],
        "backward": ["state", "privileged_state"],
        "discriminator": ["state", "privileged_state"],
        "critic_discriminator": ["state", "privileged_state", "last_action", "history_actor"],
        "critic_auxiliary": ["state", "privileged_state", "last_action", "history_actor"],
    }
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
            "timing": "next_state",
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
        "num_steps_per_env": 1,
        "num_updates_per_iteration": 16,
        "random_action_steps": 10_240,
        "save_interval": 9_375,
        "check_for_nan": True,
        "obs_groups": routes,
        "model": {
            "class_name": "rsl_rl.models.forward_backward_model:ForwardBackwardModel",
            "context_dim": 256,
            "actor_cfg": actor,
            "forward_cfg": value,
            "backward_hidden_dims": [256],
            "discriminator_hidden_dims": [1024, 1024, 1024],
            "distribution_cfg": {"class_name": "ClippedGaussianDistribution", "init_std": 0.05},
            "value_heads": [
                {
                    "spec": {
                        "name": "discriminator",
                        "kind": "critic",
                        "route": "critic_discriminator",
                        "reward_channels": ["discriminator"],
                        "ensemble_size": 2,
                        "has_target": True,
                    },
                    "network": value,
                },
                {
                    "spec": {
                        "name": "auxiliary",
                        "kind": "critic",
                        "route": "critic_auxiliary",
                        "reward_channels": list(BFM_AUXILIARY_EVIDENCE_NAMES),
                        "ensemble_size": 2,
                        "has_target": True,
                    },
                    "network": value,
                },
            ],
        },
        "replay": {
            "class_name": "rsl_rl.storage.forward_backward_replay:ForwardBackwardReplay",
            "capacity_steps": 5_000,
            "terminal_capacity_per_env": 16,
            "autoreset_mode": "same_step",
            "environment_reward_name": "environment",
            "auxiliary_evidence_names": list(BFM_AUXILIARY_EVIDENCE_NAMES),
            "reward_channels": reward_channels,
            "history_layout": {
                "history_field": "history_actor",
                "history_length": 4,
                "last_action_field": "last_action",
                "sources": [
                    {"observation_name": None, "start": 0, "stop": 29},
                    {"observation_name": "state", "start": 61, "stop": 64},
                    {"observation_name": "state", "start": 0, "stop": 29},
                    {"observation_name": "state", "start": 29, "stop": 58},
                    {"observation_name": "state", "start": 58, "stop": 61},
                ],
            },
            "seed": seed,
        },
        "expert": {"provider": expert_provider, "window_lengths": (8, 257)},
        "algorithm": {
            "class_name": "rsl_rl.algorithms.forward_backward:ForwardBackward",
            "batch_size": 1024,
            "expert_sequence_length": 8,
            "gamma": 0.98,
            "learning_rate": 3e-4,
            "backward_learning_rate": 1e-5,
            "discriminator_learning_rate": 1e-5,
            "orthogonality_coefficient": 100.0,
            "discriminator_gradient_penalty_coefficient": 10.0,
            "context_goal_fraction": 0.2,
            "context_expert_fraction": 0.6,
            "relabel_fraction": 0.8,
            "context_buffer_capacity": 8_192,
            "rollout_context_refresh_steps": 100,
            "rollout_expert_fraction": 0.5,
            "rollout_expert_steps": 250,
            "rollout_expert_context_steps": 8,
            "value_cfg": {
                "discriminator": {"learning_rate": 3e-4, "actor_coefficient": 0.05},
                "auxiliary": {
                    "learning_rate": 3e-4,
                    "actor_coefficient": 0.02,
                    "reward_coefficients": magnitudes,
                    "normalize_rewards": True,
                },
            },
            "seed": seed,
        },
        "torch_compile_mode": None,
    }


def main() -> None:
    """Run the candidate for an exact number of native environment transitions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference_config", type=Path, required=True)
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--transitions", type=int, required=True)
    parser.add_argument("--seed", type=int, default=4728)
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.transitions % args.num_envs:
        raise ValueError(f"transitions must be divisible by {args.num_envs}.")
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    env = make_native_environment(
        reference_config=args.reference_config,
        data_path=args.data_path,
        num_envs=args.num_envs,
        device=args.device,
    )
    runner = OffPolicyRunner(
        env,
        candidate_config(BFMZeroExpertProvider(seed=args.seed), seed=args.seed),
        log_dir=str(args.output_dir),
        device=args.device,
    )
    runner.learn(args.transitions // args.num_envs)


if __name__ == "__main__":
    main()
