"""Evaluate broad reward tasks and raw safety evidence in native BFM dynamics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import mujoco
import torch

from humanoidverse.agents.utils import set_seed_everywhere
from humanoidverse.envs.g1_env_helper.bench.reward_eval_hv import relabel
from humanoidverse.envs.g1_env_helper.robot import make_from_name
from humanoidverse.utils.g1_env_config import get_g1_robot_xml_root

from .candidate import make_native_environment
from .policy import load_evaluation_policy
from .reward_evaluation import BFM_REWARD_TASKS, infer_reward_contexts, normalize_reward_rollouts
from .specification import BFM_MODEL_PROFILE_DEFAULT, BFM_MODEL_PROFILES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relabel(
    model: mujoco.MjModel,
    task: str,
    qpos: torch.Tensor,
    qvel: torch.Tensor,
    action: torch.Tensor,
    workers: int,
) -> torch.Tensor:
    values = relabel(
        model,
        qpos.numpy(),
        qvel.numpy(),
        action.numpy(),
        make_from_name(task),
        max_workers=workers,
        process_executor=False,
    )
    return torch.as_tensor(values, dtype=torch.float32).reshape(-1)


def main() -> None:
    """Infer all task contexts once, run vectorized episodes, and write evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_folder", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint_type", choices=("source", "candidate"), default="source")
    parser.add_argument("--reference_config", type=Path, required=True)
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--inference_dataset", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--implementation", required=True)
    parser.add_argument("--training_seed", type=int, required=True)
    parser.add_argument("--evaluation_seed", type=int, required=True)
    parser.add_argument("--checkpoint_transition", type=int, required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--evaluator_hash", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episodes_per_task", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--inference_batch_size", type=int, default=1024)
    parser.add_argument("--reward_workers", type=int, default=12)
    parser.add_argument(
        "--model_profile",
        choices=tuple(BFM_MODEL_PROFILES),
        default=BFM_MODEL_PROFILE_DEFAULT,
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    if args.episodes_per_task < 1 or args.horizon < 1 or args.reward_workers < 1:
        raise ValueError("Evaluation counts and reward_workers must be positive.")
    args.output_dir.mkdir(parents=True)

    torch.cuda.set_device(torch.device(args.device))
    set_seed_everywhere(args.evaluation_seed)
    policy = load_evaluation_policy(
        args.model_folder,
        args.checkpoint,
        args.checkpoint_type,
        args.device,
        args.model_profile,
    )
    dataset = torch.load(args.inference_dataset, map_location="cpu", weights_only=True)
    reward_model_path = get_g1_robot_xml_root() / "scene_29dof_freebase_noadditional_actuators.xml"
    reward_model = mujoco.MjModel.from_xml_path(str(reward_model_path))
    inference_rewards = torch.stack(
        [
            _relabel(
                reward_model,
                task,
                dataset["qpos"],
                dataset["qvel"],
                dataset["action"],
                args.reward_workers,
            )
            for task in BFM_REWARD_TASKS
        ],
        dim=-1,
    )
    contexts = infer_reward_contexts(
        policy,
        dataset["observation"],
        inference_rewards,
        batch_size=args.inference_batch_size,
    )

    task_count = len(BFM_REWARD_TASKS)
    num_envs = task_count * args.episodes_per_task
    env = make_native_environment(
        reference_config=args.reference_config,
        data_path=args.data_path,
        num_envs=num_envs,
        device=args.device,
    )
    rollout_contexts = contexts.repeat_interleave(args.episodes_per_task, dim=0)
    observations = env.get_observations()
    qpos_steps = []
    qvel_steps = []
    action_steps = []
    evidence_steps = []
    done_steps = []
    timeout_steps = []
    start = time.perf_counter()
    try:
        for _step in range(args.horizon):
            with torch.no_grad():
                actions = policy.act(observations, rollout_contexts, mean=True)
            observations, _environment_reward, done, extras = env.step(actions)
            qpos, qvel = env.env._get_qpos_qvel(to_numpy=False)
            final = done & extras["final_obs_valid"]
            qpos = torch.where(final.unsqueeze(-1), extras["final_qpos"], qpos)
            qvel = torch.where(final.unsqueeze(-1), extras["final_qvel"], qvel)
            qpos_steps.append(qpos.cpu())
            qvel_steps.append(qvel.cpu())
            action_steps.append(actions.cpu())
            evidence_steps.append(extras["auxiliary_reward_evidence"].cpu())
            done_steps.append(done.cpu())
            timeout_steps.append(extras["time_outs"].cpu())
    finally:
        env.close()
    rollout_seconds = time.perf_counter() - start

    qpos = torch.stack(qpos_steps).reshape(args.horizon, task_count, args.episodes_per_task, -1)
    qvel = torch.stack(qvel_steps).reshape(args.horizon, task_count, args.episodes_per_task, -1)
    actions = torch.stack(action_steps).reshape(args.horizon, task_count, args.episodes_per_task, -1)
    evidence = torch.stack(evidence_steps).reshape(args.horizon, task_count, args.episodes_per_task, -1)
    done = torch.stack(done_steps).reshape(args.horizon, task_count, args.episodes_per_task)
    timeouts = torch.stack(timeout_steps).reshape(args.horizon, task_count, args.episodes_per_task)
    task_returns = []
    for task_index, task in enumerate(BFM_REWARD_TASKS):
        rewards = _relabel(
            reward_model,
            task,
            qpos[:, task_index].reshape(-1, qpos.shape[-1]),
            qvel[:, task_index].reshape(-1, qvel.shape[-1]),
            actions[:, task_index].reshape(-1, actions.shape[-1]),
            args.reward_workers,
        )
        task_returns.append(rewards.reshape(args.horizon, args.episodes_per_task).sum(dim=0))
    task_returns = torch.stack(task_returns)

    normalized = normalize_reward_rollouts(
        BFM_REWARD_TASKS,
        task_returns,
        evidence,
        done,
        timeouts,
        actions,
    )
    rows = [
        {
            "implementation": args.implementation,
            "training_seed": args.training_seed,
            "evaluation_seed": args.evaluation_seed,
            "checkpoint_transition": args.checkpoint_transition,
            "run_id": args.run_id,
            **row,
        }
        for row in normalized
    ]
    with (args.output_dir / "metrics.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    torch.save(
        {
            "tasks": BFM_REWARD_TASKS,
            "contexts": contexts.cpu(),
            "inference_reward_mean": inference_rewards.mean(dim=0),
            "returns": task_returns,
        },
        args.output_dir / "contexts_and_returns.pt",
    )
    manifest = {
        "schema": "forward_backward_phase2_reward_evaluation_v1",
        "implementation": args.implementation,
        "training_seed": args.training_seed,
        "evaluation_seed": args.evaluation_seed,
        "checkpoint_transition": args.checkpoint_transition,
        "run_id": args.run_id,
        "evaluator_hash": args.evaluator_hash,
        "inference_dataset_sha256": _sha256(args.inference_dataset),
        "reward_model_sha256": _sha256(reward_model_path),
        "model_profile": args.model_profile,
        "task_count": task_count,
        "episodes_per_task": args.episodes_per_task,
        "horizon": args.horizon,
        "record_count": len(rows),
        "rollout_seconds": rollout_seconds,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
