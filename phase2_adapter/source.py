"""Train the released BFM learner through the exact shared transition bridge."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
from rsl_rl.models.forward_backward_model import ForwardBackwardObservationSchema
from rsl_rl.modules.reward_channels import ForwardBackwardRewardChannel, ForwardBackwardRewardSchema
from rsl_rl.storage.forward_backward_replay import (
    ForwardBackwardAutoresetMode,
    ForwardBackwardHistoryLayout,
    ForwardBackwardReplay,
    ForwardBackwardReplayBatch,
    ForwardBackwardTransitionBatch,
    ForwardBackwardTransitionSchema,
)
from tensordict import TensorDict

from humanoidverse.agents.envs.humanoidverse_isaac import load_expert_trajectories_from_motion_lib
from humanoidverse.agents.utils import set_seed_everywhere
from humanoidverse.train import TrainConfig, Workspace

from .curriculum import run_curriculum_event
from .environment import BFM_AUXILIARY_EVIDENCE_NAMES, BFMZeroVecEnv
from .specification import (
    BFM_MODEL_PROFILE_DEFAULT,
    BFM_MODEL_PROFILES,
    BFMTrainingSchedule,
    observation_routes,
    replay_config,
    resolve_model_profile,
    resolve_training_schedule,
)


class _SourceReplay:
    """Present exact unified replay samples through the released learner's batch shape."""

    def __init__(
        self,
        observations: TensorDict,
        *,
        num_envs: int,
        action_dim: int,
        context_dim: int,
        device: str,
        seed: int,
    ) -> None:
        options = replay_config(seed)
        observation_schema = ForwardBackwardObservationSchema.from_observations(
            observations,
            observation_routes(),
        )
        reward_schema = ForwardBackwardRewardSchema(
            tuple(ForwardBackwardRewardChannel(**channel) for channel in options["reward_channels"])
        )
        transition_schema = ForwardBackwardTransitionSchema(
            observation_schema_hash=observation_schema.schema_hash,
            reward_schema_hash=reward_schema.schema_hash,
            action_width=action_dim,
            context_width=context_dim,
            environment_reward_name=str(options["environment_reward_name"]),
            auxiliary_evidence_names=tuple(options["auxiliary_evidence_names"]),
            autoreset_mode=ForwardBackwardAutoresetMode(str(options["autoreset_mode"])),
        )
        history_options = options["history_layout"]
        history_layout = ForwardBackwardHistoryLayout(
            history_field=str(history_options["history_field"]),
            history_length=int(history_options["history_length"]),
            last_action_field=history_options["last_action_field"],
            include_seed_observations=bool(history_options["include_seed_observations"]),
            sources=tuple(ForwardBackwardHistoryLayout.Source(**source) for source in history_options["sources"]),
        )
        self.storage = ForwardBackwardReplay(
            capacity_steps=int(options["capacity_steps"]),
            num_envs=num_envs,
            terminal_capacity_per_env=int(options["terminal_capacity_per_env"]),
            observation_schema=observation_schema,
            transition_schema=transition_schema,
            reward_schema=reward_schema,
            device=device,
            history_layout=history_layout,
            sampling=str(options["sampling"]),
            seed=seed,
        )

    def add(self, transition: ForwardBackwardTransitionBatch) -> None:
        """Append one exact same-step vector transition."""
        self.storage.add(transition)

    def sample(self, batch_size: int) -> dict[str, Any]:
        """Sample one strict logical batch and expose released field names."""
        return self._source_batch(self.storage.sample_random(batch_size))

    def assert_no_errors(self) -> None:
        """Fail at an update boundary if collection violated the shared contract."""
        self.storage.assert_no_errors()

    def process_env_reset(self, observations: TensorDict) -> None:
        """Close every latest edge and seed the externally reset stream."""
        self.storage.process_env_reset(
            observations,
            torch.ones(self.storage.num_envs, dtype=torch.bool, device=self.storage.device),
        )

    def __len__(self) -> int:
        return self.storage.num_transitions

    def empty(self) -> bool:
        return len(self) == 0

    @staticmethod
    def _source_batch(batch: ForwardBackwardReplayBatch) -> dict[str, Any]:
        observations = dict(batch.observations.items())
        next_observations = dict(batch.next_observations.items())
        auxiliary = {
            name: batch.auxiliary_reward_evidence[:, column : column + 1] for column, name in enumerate(BFM_AUXILIARY_EVIDENCE_NAMES)
        }
        return {
            "observation": observations,
            "action": batch.actions,
            "z": batch.behavior_context,
            "reward": batch.environment_reward,
            "aux_rewards": auxiliary,
            "next": {
                "observation": next_observations,
                "terminated": batch.terminated,
                "truncated": batch.truncated,
            },
        }


def _save_evaluation_checkpoint(model: Any, work_dir: Path, transition: int) -> None:
    """Save one immutable source policy at an evaluation transition."""
    checkpoint_root = work_dir / "evaluation_checkpoints" / str(transition)
    checkpoint_root.mkdir(parents=True, exist_ok=False)
    model.save(str(checkpoint_root / "model"))


def _source_curriculum_event(
    workspace: Workspace,
    expert: Any,
    *,
    transition: int,
    env: BFMZeroVecEnv | None = None,
) -> TensorDict | None:
    """Apply the same native tracking curriculum to the released learner."""
    model = workspace.agent._model
    motion_ids = torch.as_tensor(expert.motion_ids, dtype=torch.long, device=workspace.agent.device)
    expected_motion_ids = torch.arange(len(expert.priorities), device=motion_ids.device)
    if not torch.equal(torch.sort(motion_ids).values, expected_motion_ids):
        raise ValueError("BFM source expert motion ids must be a complete permutation.")

    def update_expert_priorities(priorities: torch.Tensor) -> None:
        expert.update_priorities(
            priorities=priorities.index_select(0, motion_ids).to(workspace.cfg.buffer_device),
            idxs=torch.arange(motion_ids.numel(), device=workspace.cfg.buffer_device),
        )

    was_training = model.training
    model.eval()
    try:
        run_curriculum_event(
            workspace.agent,
            env=workspace.train_env,
            num_envs=workspace.cfg.online_parallel_envs,
            transition=transition,
            output_dir=workspace.work_dir / "curriculum_events",
            device=workspace.agent.device,
            update_expert_priorities=update_expert_priorities,
        )
    finally:
        model.train(was_training)
    if env is not None:
        return env.reset()
    return None


def _train(workspace: Workspace) -> None:
    """Run the released update equations over exact bridge transitions."""
    cfg = workspace.cfg
    agent = workspace.agent
    expert = load_expert_trajectories_from_motion_lib(
        workspace.train_env._env,
        cfg.agent,
        device=cfg.buffer_device,
    )
    _source_curriculum_event(workspace, expert, transition=0)
    env = BFMZeroVecEnv(workspace.train_env, terminal_profile="correct_terminal", device=cfg.env.device)
    observations = env.get_observations()
    replay = _SourceReplay(
        observations,
        num_envs=cfg.online_parallel_envs,
        action_dim=workspace.action_dim,
        context_dim=cfg.agent.model.archi.z_dim,
        device=cfg.buffer_device,
        seed=cfg.seed,
    )
    replay_buffer = {"train": replay, "expert_slicer": expert}
    context = None
    totals: dict[str, torch.Tensor] | None = None
    num_metric_updates = 0
    interval_start = time.perf_counter()
    start = interval_start

    try:
        for vector_step in range(cfg.num_env_steps // cfg.online_parallel_envs):
            transition_count = vector_step * cfg.online_parallel_envs
            with torch.no_grad():
                step_count = env.episode_length_buf.reshape(cfg.online_parallel_envs, 1)
                context = agent.maybe_update_rollout_context(
                    z=context,
                    step_count=step_count,
                    replay_buffer=replay_buffer,
                )
                if transition_count < cfg.num_seed_steps:
                    actions = torch.as_tensor(
                        workspace.train_env.action_space.sample(),
                        dtype=torch.float32,
                        device=agent.device,
                    )
                else:
                    actions = agent.act(obs=dict(observations.items()), z=context, mean=False)

            next_observations, rewards, done, extras = env.step(actions)
            truncated = extras["time_outs"].bool().reshape(cfg.online_parallel_envs, 1)
            terminated = done.reshape(cfg.online_parallel_envs, 1) & ~truncated
            replay.add(
                ForwardBackwardTransitionBatch(
                    observations=observations,
                    next_observations=next_observations,
                    final_observations=extras["final_obs"],
                    actions=actions,
                    behavior_context=context,
                    environment_reward=rewards.reshape(cfg.online_parallel_envs, 1),
                    auxiliary_reward_evidence=extras["auxiliary_reward_evidence"],
                    terminated=terminated,
                    truncated=truncated,
                    context_changed=torch.zeros_like(terminated),
                    action_applied=torch.ones_like(terminated),
                    final_observation_valid=extras["final_obs_valid"].reshape(cfg.online_parallel_envs, 1),
                )
            )
            observations = next_observations

            if transition_count > cfg.num_seed_steps:
                replay.assert_no_errors()
                for _ in range(cfg.num_agent_updates):
                    metrics = agent.update(replay_buffer, transition_count)
                    if totals is None:
                        totals = {name: value.float().clone() for name, value in metrics.items()}
                    else:
                        for name, value in metrics.items():
                            totals[name].add_(value.float())
                    num_metric_updates += 1

            completed = transition_count + cfg.online_parallel_envs
            if totals is not None and completed % cfg.log_every_updates == 0:
                summary = {name: round((value / num_metric_updates).mean().item(), 6) for name, value in sorted(totals.items())}
                summary["duration [minutes]"] = (time.perf_counter() - start) / 60
                summary["FPS"] = cfg.log_every_updates / (time.perf_counter() - interval_start)
                print(summary, flush=True)
                totals = None
                num_metric_updates = 0
                interval_start = time.perf_counter()

            if completed % cfg.checkpoint_every_steps == 0:
                _save_evaluation_checkpoint(agent._model, workspace.work_dir, completed)
                reset_observations = _source_curriculum_event(workspace, expert, transition=completed, env=env)
                if reset_observations is None:
                    raise RuntimeError("A post-training curriculum event must reset the behavior environment.")
                replay.process_env_reset(reset_observations)
                observations = reset_observations

        replay.assert_no_errors()
        agent.save(str(workspace.work_dir / "checkpoint"))
    finally:
        env.close()


def _load_config(args: argparse.Namespace, schedule: BFMTrainingSchedule) -> TrainConfig:
    config = TrainConfig.model_validate_json(args.reference_config.read_text())
    hidden_dim, hidden_layers = resolve_model_profile(args.model_profile)
    env = config.env.model_copy(
        update={
            "device": args.device,
            "lafan_tail_path": str(args.data_path.resolve()),
        }
    )
    architecture = config.agent.model.archi
    architecture = architecture.model_copy(
        update={
            "f": architecture.f.model_copy(update={"hidden_dim": hidden_dim, "hidden_layers": hidden_layers}),
            "actor": architecture.actor.model_copy(update={"hidden_dim": hidden_dim, "hidden_layers": hidden_layers}),
            "critic": architecture.critic.model_copy(update={"hidden_dim": hidden_dim, "hidden_layers": hidden_layers}),
            "aux_critic": architecture.aux_critic.model_copy(update={"hidden_dim": hidden_dim, "hidden_layers": hidden_layers}),
        }
    )
    model = config.agent.model.model_copy(update={"device": args.device, "archi": architecture})
    agent = config.agent.model_copy(update={"model": model, "compile": args.compile})
    return config.model_copy(
        update={
            "agent": agent,
            "env": env,
            "work_dir": str(args.output_dir),
            "seed": args.seed,
            "online_parallel_envs": args.num_envs,
            "num_env_steps": args.transitions,
            "num_seed_steps": 10_240,
            "num_agent_updates": 16,
            "update_agent_every": args.num_envs,
            "log_every_updates": args.log_every_transitions,
            "checkpoint_every_steps": schedule.save_interval * args.num_envs,
            "checkpoint_buffer": False,
            "prioritization": False,
            "use_trajectory_buffer": False,
            "buffer_size": 5_120_000,
            "buffer_device": args.device,
            "use_wandb": False,
            "evaluations": [],
        }
    )


def main() -> None:
    """Run the released learner for an exact number of correct-terminal transitions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference_config", type=Path, required=True)
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--transitions", type=int, required=True)
    parser.add_argument("--seed", type=int, default=4728)
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log_every_transitions", type=int, default=384_000)
    parser.add_argument("--evaluation_checkpoint_every_transitions", type=int, default=9_600_000)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--model_profile",
        choices=tuple(BFM_MODEL_PROFILES),
        default=BFM_MODEL_PROFILE_DEFAULT,
    )
    args = parser.parse_args()
    schedule = resolve_training_schedule(
        transitions=args.transitions,
        num_envs=args.num_envs,
        evaluation_checkpoint_every_transitions=args.evaluation_checkpoint_every_transitions,
    )
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    torch.cuda.set_device(torch.device(args.device))
    set_seed_everywhere(args.seed)
    workspace = Workspace(_load_config(args, schedule))
    _train(workspace)


if __name__ == "__main__":
    main()
