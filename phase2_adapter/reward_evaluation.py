"""Frozen broad-reward and safety evaluation operators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from rsl_rl.modules.forward_backward import reward_context

from .environment import BFM_AUXILIARY_EVIDENCE_NAMES

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
BFM_AUXILIARY_COST_COEFFICIENTS = torch.tensor((0.0, 0.1, 10.0, 0.0, 1.0, 0.4, 4.0, 2.0))
BFM_HARD_SAFETY_NAMES = ("limits_dof_pos", "limits_torque", "penalty_undesired_contact")


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
