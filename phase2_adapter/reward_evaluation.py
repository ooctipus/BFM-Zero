"""Frozen broad-reward and safety evaluation operators."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, cast

import torch
from rsl_rl.modules.forward_backward import reward_context

from humanoidverse.envs.g1_env_helper.bench.reward_eval_hv import relabel as _native_relabel
from humanoidverse.envs.g1_env_helper.robot import make_from_name as _make_reward

from .environment import (
    BFM_ACTION_DIM,
    BFM_AUXILIARY_EVIDENCE_NAMES,
    BFM_FIELD_WIDTHS,
    BFM_QPOS_DIM,
    BFM_QVEL_DIM,
)

BFM_REWARD_TASKS = (
    "move-ego-0-0",
    "move-ego-low0.5-0-0",
    "move-ego-0-0.7",
    "move-ego-0-0.3",
    "move-ego-90-0.3",
    "move-ego-180-0.3",
    "move-ego--90-0.3",
    "rotate-z-5-0.5",
    "rotate-z--5-0.5",
    "raisearms-l-l",
    "raisearms-l-m",
    "raisearms-m-l",
    "raisearms-m-m",
    "move-arms-0-0.7-m-m",
    "move-arms-90-0.7-m-m",
    "move-arms-180-0.4-m-m",
    "move-arms--90-0.7-m-m",
    "move-arms-0-0.7-l-m",
    "move-arms-90-0.7-l-m",
    "move-arms-180-0.4-l-m",
    "move-arms--90-0.7-l-m",
    "move-arms-0-0.7-m-l",
    "move-arms-90-0.7-m-l",
    "move-arms-180-0.4-m-l",
    "move-arms--90-0.7-m-l",
    "move-arms-0-0.7-l-l",
    "move-arms-90-0.7-l-l",
    "move-arms-180-0.4-l-l",
    "move-arms--90-0.7-l-l",
    "spin-arms-5-l-l",
    "spin-arms--5-l-l",
    "spin-arms-5-l-m",
    "spin-arms--5-l-m",
    "spin-arms-5-m-l",
    "spin-arms--5-m-l",
    "crouch-0",
    "crouch-0.25",
    "sitonground",
)
BFM_REWARD_TASKS_SHA256 = hashlib.sha256("\0".join(BFM_REWARD_TASKS).encode()).hexdigest()
BFM_AUXILIARY_COST_COEFFICIENTS = torch.tensor((0.0, 0.1, 10.0, 0.0, 1.0, 0.4, 4.0, 2.0))
BFM_HARD_SAFETY_NAMES = ("limits_dof_pos", "limits_torque", "penalty_undesired_contact")
BFM_REWARD_INFERENCE_DATASET_SCHEMA = "bfm_reward_inference_dataset_v2"
BFM_REWARD_OBSERVATION_NAMES = ("state", "privileged_state")


def _validate_relabel_inputs(qpos: torch.Tensor, qvel: torch.Tensor, action: torch.Tensor) -> int:
    """Validate the exact native G1 reward-state contract."""
    expected = (
        ("qpos", qpos, BFM_QPOS_DIM),
        ("qvel", qvel, BFM_QVEL_DIM),
        ("action", action, BFM_ACTION_DIM),
    )
    sample_count = qpos.shape[0] if qpos.ndim == 2 else -1
    for name, value, width in expected:
        if value.ndim != 2 or value.shape != (sample_count, width):
            raise ValueError(f"{name} must have shape [samples, {width}].")
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain finite floating-point values.")
    if sample_count < 1:
        raise ValueError("Reward relabeling requires at least one sample.")
    return sample_count


def relabel_reward_tasks(
    model: Any,
    task_names: Sequence[str],
    qpos: torch.Tensor,
    qvel: torch.Tensor,
    action: torch.Tensor,
    *,
    workers: int,
) -> torch.Tensor:
    """Evaluate frozen BFM reward tasks on one batch of native G1 states."""
    tasks = tuple(task_names)
    if not tasks:
        raise ValueError("At least one reward task is required.")
    if len(set(tasks)) != len(tasks):
        raise ValueError("Reward tasks must be unique.")
    unknown = tuple(task for task in tasks if task not in BFM_REWARD_TASKS)
    if unknown:
        raise ValueError(f"Unknown BFM reward tasks: {unknown}.")
    if workers < 1:
        raise ValueError("workers must be positive.")
    sample_count = _validate_relabel_inputs(qpos, qvel, action)
    qpos_numpy = qpos.detach().cpu().numpy()
    qvel_numpy = qvel.detach().cpu().numpy()
    action_numpy = action.detach().cpu().numpy()
    labels = []
    for task in tasks:
        values = _native_relabel(
            model,
            qpos_numpy,
            qvel_numpy,
            action_numpy,
            _make_reward(task),
            max_workers=workers,
            process_executor=False,
        )
        values = torch.as_tensor(values, dtype=torch.float32)
        if values.shape != (sample_count, 1) or not torch.isfinite(values).all():
            raise RuntimeError(f"Reward task {task!r} returned invalid labels with shape {tuple(values.shape)}.")
        labels.append(values[:, 0])
    return torch.stack(labels, dim=-1).contiguous()


def _validate_sha256(name: str, value: object) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")


def validate_reward_inference_dataset(dataset: Mapping[str, object]) -> None:
    """Validate the immutable policy-independent reward-inference dataset."""
    required_keys = {
        "schema",
        "reward_tasks",
        "reference_config_sha256",
        "data_sha256",
        "reward_model_sha256",
        "observation",
        "reward_labels",
        "motion_id",
    }
    if set(dataset) != required_keys:
        raise ValueError(f"Reward-inference dataset fields must be exactly {sorted(required_keys)}.")
    if dataset["schema"] != BFM_REWARD_INFERENCE_DATASET_SCHEMA:
        raise ValueError(f"Reward-inference dataset schema must be {BFM_REWARD_INFERENCE_DATASET_SCHEMA!r}.")
    reward_tasks = dataset["reward_tasks"]
    if not isinstance(reward_tasks, tuple) or reward_tasks != BFM_REWARD_TASKS:
        raise ValueError("Reward-inference dataset tasks must exactly match the frozen BFM task order.")
    for name in ("reference_config_sha256", "data_sha256", "reward_model_sha256"):
        _validate_sha256(name, dataset[name])

    observations = dataset["observation"]
    if not isinstance(observations, Mapping) or set(observations) != set(BFM_REWARD_OBSERVATION_NAMES):
        raise ValueError("Reward-inference observations must contain exactly state and privileged_state.")
    labels = dataset["reward_labels"]
    motion_id = dataset["motion_id"]
    if not isinstance(labels, torch.Tensor) or labels.ndim != 2:
        raise ValueError("reward_labels must be a tensor with shape [samples, tasks].")
    sample_count = labels.shape[0]
    if labels.shape != (sample_count, len(BFM_REWARD_TASKS)) or sample_count < 1:
        raise ValueError(f"reward_labels must have shape [samples, {len(BFM_REWARD_TASKS)}].")
    if labels.device.type != "cpu" or labels.dtype != torch.float32 or not labels.is_contiguous() or not torch.isfinite(labels).all():
        raise ValueError("reward_labels must be finite, contiguous CPU float32 values.")
    if not isinstance(motion_id, torch.Tensor) or motion_id.shape != (sample_count,):
        raise ValueError("motion_id must have shape [samples].")
    if motion_id.device.type != "cpu" or motion_id.dtype != torch.long or not motion_id.is_contiguous() or torch.any(motion_id < 0):
        raise ValueError("motion_id must be a contiguous non-negative CPU int64 tensor.")
    for name in BFM_REWARD_OBSERVATION_NAMES:
        width = BFM_FIELD_WIDTHS[name]
        value = observations[name]
        if not isinstance(value, torch.Tensor) or value.shape != (sample_count, width):
            raise ValueError(f"observation[{name!r}] must have shape [samples, {width}].")
        if value.device.type != "cpu" or value.dtype != torch.float32 or not value.is_contiguous() or not torch.isfinite(value).all():
            raise ValueError(f"observation[{name!r}] must be finite, contiguous CPU float32 values.")


def build_reward_inference_dataset(
    observations: Mapping[str, torch.Tensor],
    reward_labels: torch.Tensor,
    motion_id: torch.Tensor,
    *,
    reference_config_sha256: str,
    data_sha256: str,
    reward_model_sha256: str,
) -> dict[str, object]:
    """Build the strict payload consumed by every reward evaluator."""
    dataset: dict[str, object] = {
        "schema": BFM_REWARD_INFERENCE_DATASET_SCHEMA,
        "reward_tasks": BFM_REWARD_TASKS,
        "reference_config_sha256": reference_config_sha256,
        "data_sha256": data_sha256,
        "reward_model_sha256": reward_model_sha256,
        "observation": dict(observations),
        "reward_labels": reward_labels,
        "motion_id": motion_id,
    }
    validate_reward_inference_dataset(dataset)
    return dataset


def motion_sample_indices(length: int, samples: int, *, device: torch.device | str) -> torch.Tensor:
    """Select deterministic, uniformly spaced reached states from one motion."""
    if length < 2:
        raise ValueError("A reward-inference motion must contain at least two states.")
    if samples < 1:
        raise ValueError("samples must be positive.")
    count = min(samples, length - 1)
    return torch.linspace(1, length - 1, count, device=device).round().long()


@torch.no_grad()
def infer_reward_contexts(
    policy: Any,
    observations: Mapping[str, torch.Tensor],
    rewards: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    """Compute B once and infer all reward-task contexts together."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    sample_count = rewards.shape[0]
    if rewards.ndim != 2 or sample_count < 1:
        raise ValueError("rewards must have shape [samples, tasks].")
    if any(value.shape[0] != sample_count for value in observations.values()):
        raise ValueError("Every inference observation must align with rewards.")

    device = torch.device(policy.device)
    backward_chunks = []
    for start in range(0, sample_count, batch_size):
        stop = min(start + batch_size, sample_count)
        batch = {name: value[start:stop].to(device) for name, value in observations.items()}
        backward_chunks.append(policy.backward_map(batch).float())
    backward = torch.cat(backward_chunks)
    task_rewards = rewards.to(device=device, dtype=backward.dtype)
    weights = torch.softmax(10.0 * task_rewards, dim=0)
    return policy.project_z(reward_context(backward, task_rewards, weights))


def infer_reward_contexts_from_dataset(
    policy: Any,
    dataset: Mapping[str, object],
    *,
    batch_size: int,
    reference_config_sha256: str,
    data_sha256: str,
    reward_model_sha256: str,
) -> torch.Tensor:
    """Infer contexts directly from cached policy-independent reward labels."""
    validate_reward_inference_dataset(dataset)
    expected_hashes = {
        "reference_config_sha256": reference_config_sha256,
        "data_sha256": data_sha256,
        "reward_model_sha256": reward_model_sha256,
    }
    for name, expected in expected_hashes.items():
        if dataset[name] != expected:
            raise ValueError(f"Reward-inference labels have a different {name.removesuffix('_sha256')} identity.")
    observations = cast(Mapping[str, torch.Tensor], dataset["observation"])
    rewards = cast(torch.Tensor, dataset["reward_labels"])
    return infer_reward_contexts(policy, observations, rewards, batch_size=batch_size)


def normalize_reward_rollouts(
    task_names: Sequence[str],
    task_returns: torch.Tensor,
    auxiliary_evidence: torch.Tensor,
    done: torch.Tensor,
    timeouts: torch.Tensor,
    actions: torch.Tensor,
) -> list[dict[str, object]]:
    """Convert vectorized broad-task rollouts into per-episode scalar rows."""
    steps, task_count, episode_count, evidence_count = auxiliary_evidence.shape
    if task_count != len(task_names) or evidence_count != len(BFM_AUXILIARY_EVIDENCE_NAMES):
        raise ValueError("Auxiliary evidence does not match the task or reward schema.")
    if task_returns.shape != (task_count, episode_count):
        raise ValueError("task_returns must have shape [tasks, episodes].")
    if done.shape != (steps, task_count, episode_count) or timeouts.shape != done.shape:
        raise ValueError("done and timeouts must align with auxiliary evidence.")
    if actions.shape[:3] != (steps, task_count, episode_count):
        raise ValueError("actions must align with auxiliary evidence.")

    coefficients = BFM_AUXILIARY_COST_COEFFICIENTS.to(auxiliary_evidence)
    hard_columns = [BFM_AUXILIARY_EVIDENCE_NAMES.index(name) for name in BFM_HARD_SAFETY_NAMES]
    rows: list[dict[str, object]] = []
    for task_index, task_name in enumerate(task_names):
        for episode in range(episode_count):
            evidence = auxiliary_evidence[:, task_index, episode]
            active = evidence.abs().gt(0.0)
            metrics = {
                "return": task_returns[task_index, episode],
                "auxiliary_cost": (evidence * coefficients).sum(dim=-1).mean(),
                "safety_violation_rate": active[:, hard_columns].any(dim=-1).float().mean(),
                "termination_rate": (done[:, task_index, episode] & ~timeouts[:, task_index, episode]).float().mean(),
                "action_l2": actions[:, task_index, episode].square().sum(dim=-1).sqrt().mean(),
            }
            for column, name in enumerate(BFM_AUXILIARY_EVIDENCE_NAMES):
                metrics[f"{name}_mean"] = evidence[:, column].mean()
                metrics[f"{name}_active_fraction"] = active[:, column].float().mean()
            rows.extend(
                {
                    "task": task_name,
                    "episode": episode,
                    "metric_name": name,
                    "metric_value": float(value),
                }
                for name, value in metrics.items()
            )
    return rows
