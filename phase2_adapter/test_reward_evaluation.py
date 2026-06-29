"""Tests for frozen broad-reward and safety evaluation operators."""

from __future__ import annotations

import math

import torch

from phase2_adapter.environment import BFM_AUXILIARY_EVIDENCE_NAMES
from phase2_adapter.reward_evaluation import (
    BFM_REWARD_TASKS,
    infer_reward_contexts,
    motion_sample_indices,
    normalize_reward_rollouts,
)


def test_reward_task_suite_is_unique_and_broad() -> None:
    """The released locomotion, arm, spin, and posture suite should stay frozen."""
    assert len(BFM_REWARD_TASKS) == 38
    assert len(set(BFM_REWARD_TASKS)) == len(BFM_REWARD_TASKS)
    assert {"move-ego-0-0.7", "spin-arms-5-l-l", "sitonground"} <= set(BFM_REWARD_TASKS)


def test_motion_samples_are_deterministic_reached_states() -> None:
    """Inference data should cover a motion without sampling its initial node."""
    first = motion_sample_indices(301, 64, device="cpu")
    second = motion_sample_indices(301, 64, device="cpu")

    assert torch.equal(first, second)
    assert first.shape == (64,)
    assert first[0].item() == 1
    assert first[-1].item() == 300
    assert torch.all(first[1:] > first[:-1])


def test_reward_contexts_compute_backward_features_once_for_all_tasks() -> None:
    """All reward tasks should share one batched B pass."""
    calls: list[int] = []

    class Policy:
        device = torch.device("cpu")

        def backward_map(self, observations):
            calls.append(observations["state"].shape[0])
            return observations["state"][:, :2]

        def project_z(self, context):
            return context

    observations = {
        "state": torch.arange(30, dtype=torch.float32).reshape(10, 3),
        "privileged_state": torch.zeros(10, 4),
    }
    rewards = torch.stack((torch.linspace(-1.0, 1.0, 10), torch.linspace(1.0, -1.0, 10)), dim=-1)

    contexts = infer_reward_contexts(Policy(), observations, rewards, batch_size=4)

    assert calls == [4, 4, 2]
    backward = observations["state"][:, :2]
    expected = (rewards * torch.softmax(10.0 * rewards, dim=0)).mT @ backward
    torch.testing.assert_close(contexts, expected)


def test_reward_rollout_normalization_preserves_raw_safety_evidence() -> None:
    """Per-task records should expose return, hard violations, and every raw term."""
    evidence_count = len(BFM_AUXILIARY_EVIDENCE_NAMES)
    evidence = torch.zeros(3, 1, 2, evidence_count)
    hard = BFM_AUXILIARY_EVIDENCE_NAMES.index("limits_dof_pos")
    action_rate = BFM_AUXILIARY_EVIDENCE_NAMES.index("penalty_action_rate")
    evidence[0, 0, 0, hard] = 2.0
    evidence[:, 0, 0, action_rate] = torch.tensor([1.0, 2.0, 3.0])
    done = torch.zeros(3, 1, 2, dtype=torch.bool)
    timeouts = torch.zeros_like(done)
    done[1, 0, 1] = True
    actions = torch.ones(3, 1, 2, 2)

    rows = normalize_reward_rollouts(
        ("task",),
        torch.tensor([[12.0, 8.0]]),
        evidence,
        done,
        timeouts,
        actions,
    )
    episode_zero = {row["metric_name"]: row["metric_value"] for row in rows if row["episode"] == 0}
    episode_one = {row["metric_name"]: row["metric_value"] for row in rows if row["episode"] == 1}

    assert episode_zero["return"] == 12.0
    assert math.isclose(episode_zero["safety_violation_rate"], 1.0 / 3.0, rel_tol=1e-6)
    assert math.isclose(episode_zero["limits_dof_pos_mean"], 2.0 / 3.0, rel_tol=1e-6)
    assert episode_zero["penalty_action_rate_mean"] == 2.0
    assert math.isclose(episode_one["termination_rate"], 1.0 / 3.0, rel_tol=1e-6)
    assert len(rows) == 2 * (5 + 2 * evidence_count)
