"""Build the RSL-RL candidate on the unchanged native BFM-Zero environment."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch
from rsl_rl.runners.off_policy_runner import OffPolicyRunner

from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig
from humanoidverse.agents.utils import set_seed_everywhere

from .curriculum import run_curriculum_event
from .environment import BFM_AUXILIARY_EVIDENCE_NAMES, BFMZeroVecEnv
from .expert import BFMZeroExpertProvider
from .specification import (
    BFM_MODEL_PROFILE_DEFAULT,
    BFM_MODEL_PROFILES,
    observation_routes,
    replay_config,
    resolve_model_profile,
)


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


def candidate_config(
    expert_provider: BFMZeroExpertProvider,
    *,
    seed: int,
    model_profile: str = BFM_MODEL_PROFILE_DEFAULT,
) -> dict[str, Any]:
    """Return the released-scale BFM configuration with exact-terminal collection."""
    hidden_dim, hidden_layers = resolve_model_profile(model_profile)
    actor = {"hidden_dim": hidden_dim, "hidden_layers": hidden_layers, "embedding_layers": 2, "residual": True}
    value = {
        "hidden_dim": hidden_dim,
        "hidden_layers": hidden_layers,
        "embedding_layers": hidden_layers,
        "residual": True,
    }
    magnitudes = (0.0, 0.1, 10.0, 0.0, 1.0, 0.4, 4.0, 2.0)
    return {
        "num_steps_per_env": 1,
        "num_updates_per_iteration": 16,
        "random_action_steps": 10_240,
        "save_interval": 9_375,
        "check_for_nan": True,
        "obs_groups": observation_routes(),
        "model": {
            "class_name": "rsl_rl.models.forward_backward_model:ForwardBackwardModel",
            "context_dim": 256,
            "actor_cfg": actor,
            "forward_cfg": value,
            "backward_hidden_dims": [256],
            "discriminator_hidden_dims": [1024, 1024, 1024],
            "distribution_cfg": {"class_name": "ClippedGaussianDistribution", "init_std": 0.05, "noise_clip": 0.3},
            "initialization_type": "orthogonal",
            "normalization_type": "exponential",
            "normalization_eps": 1e-5,
            "normalization_momentum": 0.01,
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
                        "reward_composition": "scalar",
                        "ensemble_size": 2,
                        "has_target": True,
                    },
                    "network": value,
                },
            ],
        },
        "replay": replay_config(seed),
        "expert": {"provider": expert_provider, "window_lengths": (8, 257)},
        "algorithm": {
            "class_name": "rsl_rl.algorithms.forward_backward:ForwardBackward",
            "batch_size": 1024,
            "expert_sequence_length": 8,
            "gamma": 0.98,
            "learning_rate": 3e-4,
            "backward_learning_rate": 1e-5,
            "discriminator_learning_rate": 1e-5,
            "fb_pessimism": 0.0,
            "orthogonality_coefficient": 100.0,
            "discriminator_gradient_penalty_coefficient": 10.0,
            "context_goal_fraction": 0.2,
            "context_expert_fraction": 0.6,
            "relabel_fraction": 0.8,
            "context_buffer_capacity": 8_192,
            "rollout_context_refresh_steps": 100,
            "rollout_expert_fraction": 0.5,
            "random_action_range": (-5.0, 5.0),
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


class _BFMEvaluationCheckpointRunner(OffPolicyRunner):
    """Write compact milestone policies and one final recovery checkpoint."""

    def __init__(self, *args, final_transitions: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._final_transitions = final_transitions

    def save(self, path: str, infos: dict | None = None) -> None:
        if self.logger.log_dir is None:
            raise RuntimeError("BFM milestone checkpoints require a log directory.")
        destination = Path(self.logger.log_dir) / "evaluation_checkpoints"
        destination.mkdir(exist_ok=True)
        policy = destination / f"{self.collected_transitions}.pt"
        temporary = policy.with_suffix(".tmp")
        torch.save({"model_state_dict": self.alg.get_policy().state_dict()}, temporary)
        temporary.replace(policy)
        self.curriculum_event()
        if self.collected_transitions == self._final_transitions:
            super().save(path, infos)

    def curriculum_event(self) -> None:
        """Apply one bridge-owned tracking curriculum update."""
        from .policy import BFMCandidatePolicy

        model = self.alg.model
        was_training = model.training
        model.eval()
        try:
            run_curriculum_event(
                BFMCandidatePolicy(model),
                env=self.env.env,
                num_envs=self.env.num_envs,
                transition=self.collected_transitions,
                output_dir=Path(self.logger.log_dir) / "curriculum_events",
                device=self.device,
                update_expert_priorities=lambda values: self.alg.expert.set_priorities(values.to(self.alg.expert.device)),
            )
        finally:
            model.train(was_training)
        observations = self.env.reset()
        if self.collected_transitions:
            self.alg.process_env_reset(
                observations,
                torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device),
            )


def _configure_training_runtime() -> None:
    """Match the released BFM float32 matrix-multiplication policy."""
    torch.set_float32_matmul_precision("high")


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
    parser.add_argument(
        "--model_profile",
        choices=tuple(BFM_MODEL_PROFILES),
        default=BFM_MODEL_PROFILE_DEFAULT,
    )
    args = parser.parse_args()
    if args.transitions % args.num_envs:
        raise ValueError(f"transitions must be divisible by {args.num_envs}.")
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    _configure_training_runtime()
    torch.cuda.set_device(torch.device(args.device))
    set_seed_everywhere(args.seed)
    env = make_native_environment(
        reference_config=args.reference_config,
        data_path=args.data_path,
        num_envs=args.num_envs,
        device=args.device,
    )
    set_seed_everywhere(args.seed)
    runner = _BFMEvaluationCheckpointRunner(
        env,
        candidate_config(
            BFMZeroExpertProvider(seed=args.seed),
            seed=args.seed,
            model_profile=args.model_profile,
        ),
        log_dir=str(args.output_dir),
        device=args.device,
        final_transitions=args.transitions,
    )
    runner.curriculum_event()
    runner.learn(args.transitions // args.num_envs)
    torch.save({"model_state_dict": runner.alg.get_policy().state_dict()}, args.output_dir / "policy.pt")
    env.close()


if __name__ == "__main__":
    main()
