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
from .specification import observation_routes, replay_config


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
            last_action_field=str(history_options["last_action_field"]),
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
            seed=seed,
        )

    def add(self, transition: ForwardBackwardTransitionBatch) -> None:
        """Append one exact same-step vector transition."""
        self.storage.add(transition)

    def sample(self, batch_size: int) -> dict[str, Any]:
        """Sample uniformly from logical transitions and expose released field names."""
        while True:
            batch = self.storage.sample_random(2 * batch_size)
            indices = batch.valid.squeeze(-1).nonzero(as_tuple=False).squeeze(-1)
            if indices.numel() >= batch_size:
                return self._source_batch(batch, indices[:batch_size])

    def assert_no_errors(self) -> None:
        """Fail at an update boundary if collection violated the shared contract."""
        self.storage.assert_no_errors()

    def __len__(self) -> int:
        return self.storage.num_transitions

    def empty(self) -> bool:
        return len(self) == 0

    @staticmethod
    def _source_batch(batch: ForwardBackwardReplayBatch, indices: torch.Tensor) -> dict[str, Any]:
        observations = {name: value[indices] for name, value in batch.observations.items()}
        next_observations = {name: value[indices] for name, value in batch.next_observations.items()}
        auxiliary = {
            name: batch.auxiliary_reward_evidence[indices, column : column + 1] for column, name in enumerate(BFM_AUXILIARY_EVIDENCE_NAMES)
        }
        return {
            "observation": observations,
            "action": batch.actions[indices],
            "z": batch.behavior_context[indices],
            "reward": batch.environment_reward[indices],
            "aux_rewards": auxiliary,
            "next": {
                "observation": next_observations,
                "terminated": batch.terminated[indices],
                "truncated": batch.truncated[indices],
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
) -> None:
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
        env.reset()


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
                _source_curriculum_event(workspace, expert, transition=completed, env=env)

        replay.assert_no_errors()
        agent.save(str(workspace.work_dir / "checkpoint"))
    finally:
        env.close()


def _load_config(args: argparse.Namespace) -> TrainConfig:
    config = TrainConfig.model_validate_json(args.reference_config.read_text())
    env = config.env.model_copy(
        update={
            "device": args.device,
            "lafan_tail_path": str(args.data_path.resolve()),
        }
    )
    model = config.agent.model.model_copy(update={"device": args.device})
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
            "checkpoint_every_steps": args.evaluation_checkpoint_every_transitions,
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
    args = parser.parse_args()
    if args.transitions % args.num_envs:
        raise ValueError("transitions must be divisible by num_envs.")
    if args.evaluation_checkpoint_every_transitions < 1:
        raise ValueError("evaluation_checkpoint_every_transitions must be positive.")
    if args.evaluation_checkpoint_every_transitions % args.num_envs:
        raise ValueError("evaluation checkpoint cadence must be divisible by num_envs.")
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    torch.cuda.set_device(torch.device(args.device))
    set_seed_everywhere(args.seed)
    workspace = Workspace(_load_config(args))
    _train(workspace)


if __name__ == "__main__":
    main()
