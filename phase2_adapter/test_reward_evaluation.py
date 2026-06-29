"""Tests for frozen broad-reward and safety evaluation operators."""

from __future__ import annotations

import hashlib
import math

import mujoco
import pytest
import torch

from humanoidverse.envs.g1_env_helper.bench.reward_eval_hv import relabel as native_relabel
from humanoidverse.envs.g1_env_helper.robot import make_from_name
from humanoidverse.utils.g1_env_config import get_g1_robot_xml_root
from phase2_adapter import reward_evaluation
from phase2_adapter.environment import (
    BFM_ACTION_DIM,
    BFM_AUXILIARY_EVIDENCE_NAMES,
    BFM_FIELD_WIDTHS,
    BFM_QPOS_DIM,
    BFM_QVEL_DIM,
)
from phase2_adapter.reward_evaluation import (
    BFM_REWARD_OBSERVATION_NAMES,
    BFM_REWARD_TASKS,
    BFM_REWARD_TASKS_SHA256,
    build_reward_inference_dataset,
    infer_reward_contexts,
    infer_reward_contexts_from_dataset,
    motion_sample_indices,
    normalize_reward_rollouts,
    relabel_reward_tasks,
)

_REFERENCE_CONFIG_SHA256 = "1" * 64
_DATA_SHA256 = "2" * 64
_REWARD_MODEL_SHA256 = "0" * 64


def test_reward_task_suite_is_unique_and_broad() -> None:
    """The released locomotion, arm, spin, and posture suite should stay frozen."""
    assert len(BFM_REWARD_TASKS) == 38
    assert len(set(BFM_REWARD_TASKS)) == len(BFM_REWARD_TASKS)
    assert BFM_REWARD_TASKS_SHA256 == hashlib.sha256("\0".join(BFM_REWARD_TASKS).encode()).hexdigest()
    assert {"move-ego-0-0.7", "spin-arms-5-l-l", "sitonground"} <= set(BFM_REWARD_TASKS)


def test_cached_reward_labels_exactly_match_native_direct_relabel(tmp_path) -> None:
    """The persisted cache should be bit-exact with the released MuJoCo reward backend."""
    model_path = get_g1_robot_xml_root() / "scene_29dof_freebase_noadditional_actuators.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    qpos = torch.from_numpy(model.qpos0.copy()).float().repeat(2, 1)
    qpos[1, 0] = 0.1
    qvel = torch.zeros(2, model.nv)
    qvel[1, 0] = 0.2
    action = torch.zeros(2, model.nu)
    direct = torch.stack(
        [
            torch.as_tensor(
                native_relabel(
                    model,
                    qpos.numpy(),
                    qvel.numpy(),
                    action.numpy(),
                    make_from_name(task),
                    max_workers=1,
                    process_executor=False,
                ),
                dtype=torch.float32,
            )[:, 0]
            for task in BFM_REWARD_TASKS
        ],
        dim=-1,
    )
    cached = relabel_reward_tasks(model, BFM_REWARD_TASKS, qpos, qvel, action, workers=1)
    observations = {name: torch.zeros(qpos.shape[0], BFM_FIELD_WIDTHS[name]) for name in BFM_REWARD_OBSERVATION_NAMES}
    reward_model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    dataset = build_reward_inference_dataset(
        observations,
        cached,
        torch.arange(qpos.shape[0]),
        reference_config_sha256=_REFERENCE_CONFIG_SHA256,
        data_sha256=_DATA_SHA256,
        reward_model_sha256=reward_model_sha256,
    )
    dataset_path = tmp_path / "native_reward_dataset.pt"
    torch.save(dataset, dataset_path)
    persisted = torch.load(dataset_path, map_location="cpu", weights_only=True)

    assert torch.equal(cached, direct)
    assert torch.equal(persisted["reward_labels"], direct)


def test_reward_dataset_caches_direct_labels_exactly(monkeypatch, tmp_path) -> None:
    """Cached policy-independent labels should exactly replace evaluator-time relabeling."""
    sample_count = 4
    qpos = torch.arange(sample_count * BFM_QPOS_DIM, dtype=torch.float32).reshape(sample_count, -1)
    qvel = torch.arange(sample_count * BFM_QVEL_DIM, dtype=torch.float32).reshape(sample_count, -1)
    action = torch.arange(sample_count * BFM_ACTION_DIM, dtype=torch.float32).reshape(sample_count, -1)
    task_indices = {task: index for index, task in enumerate(BFM_REWARD_TASKS)}

    def make_reward(task):
        return task_indices[task]

    def native_relabel(model, qpos_numpy, qvel_numpy, action_numpy, reward, **kwargs):
        assert model == "model"
        assert kwargs == {"max_workers": 3, "process_executor": False}
        return (qpos_numpy[:, :1] + qvel_numpy[:, :1] + action_numpy[:, :1] + reward).astype("float32")

    monkeypatch.setattr(reward_evaluation, "_make_reward", make_reward)
    monkeypatch.setattr(reward_evaluation, "_native_relabel", native_relabel)
    direct = relabel_reward_tasks("model", BFM_REWARD_TASKS, qpos, qvel, action, workers=3)
    observations = {
        name: torch.arange(sample_count * BFM_FIELD_WIDTHS[name], dtype=torch.float32).reshape(sample_count, BFM_FIELD_WIDTHS[name])
        for name in BFM_REWARD_OBSERVATION_NAMES
    }
    dataset = build_reward_inference_dataset(
        observations,
        direct,
        torch.arange(sample_count),
        reference_config_sha256=_REFERENCE_CONFIG_SHA256,
        data_sha256=_DATA_SHA256,
        reward_model_sha256=_REWARD_MODEL_SHA256,
    )
    dataset_path = tmp_path / "reward_dataset.pt"
    torch.save(dataset, dataset_path)
    cached = torch.load(dataset_path, map_location="cpu", weights_only=True)

    assert torch.equal(cached["reward_labels"], direct)

    class Policy:
        device = torch.device("cpu")

        def backward_map(self, batch):
            return batch["state"][:, :4]

        def project_z(self, context):
            return context

    expected_contexts = infer_reward_contexts(Policy(), observations, direct, batch_size=2)
    monkeypatch.setattr(
        reward_evaluation,
        "_native_relabel",
        lambda *args, **kwargs: pytest.fail("cached inference must not relabel rewards"),
    )
    cached_contexts = infer_reward_contexts_from_dataset(
        Policy(),
        cached,
        batch_size=2,
        reference_config_sha256=_REFERENCE_CONFIG_SHA256,
        data_sha256=_DATA_SHA256,
        reward_model_sha256=_REWARD_MODEL_SHA256,
    )
    torch.testing.assert_close(cached_contexts, expected_contexts)


def test_reward_relabel_rejects_unknown_tasks_and_native_shape_mismatches() -> None:
    """Reward labels should fail before native evaluation on an ambiguous task or state layout."""
    qpos = torch.zeros(2, BFM_QPOS_DIM)
    qvel = torch.zeros(2, BFM_QVEL_DIM)
    action = torch.zeros(2, BFM_ACTION_DIM)

    with pytest.raises(ValueError, match="Unknown BFM reward tasks"):
        relabel_reward_tasks(object(), ("not-a-bfm-task",), qpos, qvel, action, workers=1)
    with pytest.raises(ValueError, match=r"qpos must have shape \[samples, 36\]"):
        relabel_reward_tasks(object(), BFM_REWARD_TASKS[:1], qpos[:, :-1], qvel, action, workers=1)
    with pytest.raises(ValueError, match=r"qvel must have shape \[samples, 35\]"):
        relabel_reward_tasks(object(), BFM_REWARD_TASKS[:1], qpos, qvel[:1], action, workers=1)


def test_reward_dataset_rejects_legacy_or_misaligned_labels() -> None:
    """The evaluator should accept only the exact v2 task-label contract without fallback."""
    sample_count = 2
    observations = {name: torch.zeros(sample_count, BFM_FIELD_WIDTHS[name]) for name in BFM_REWARD_OBSERVATION_NAMES}
    labels = torch.zeros(sample_count, len(BFM_REWARD_TASKS))
    dataset = build_reward_inference_dataset(
        observations,
        labels,
        torch.arange(sample_count),
        reference_config_sha256=_REFERENCE_CONFIG_SHA256,
        data_sha256=_DATA_SHA256,
        reward_model_sha256=_REWARD_MODEL_SHA256,
    )

    with pytest.raises(ValueError, match="different reward_model identity"):
        infer_reward_contexts_from_dataset(
            object(),
            dataset,
            batch_size=1,
            reference_config_sha256=_REFERENCE_CONFIG_SHA256,
            data_sha256=_DATA_SHA256,
            reward_model_sha256="3" * 64,
        )
    with pytest.raises(ValueError, match="different data identity"):
        infer_reward_contexts_from_dataset(
            object(),
            dataset,
            batch_size=1,
            reference_config_sha256=_REFERENCE_CONFIG_SHA256,
            data_sha256="4" * 64,
            reward_model_sha256=_REWARD_MODEL_SHA256,
        )
    legacy = dict(dataset, schema="bfm_reward_inference_dataset_v1")
    with pytest.raises(ValueError, match="schema must be"):
        infer_reward_contexts_from_dataset(
            object(),
            legacy,
            batch_size=1,
            reference_config_sha256=_REFERENCE_CONFIG_SHA256,
            data_sha256=_DATA_SHA256,
            reward_model_sha256=_REWARD_MODEL_SHA256,
        )
    wrong_shape = dict(dataset, reward_labels=labels[:, :-1].contiguous())
    with pytest.raises(ValueError, match=r"reward_labels must have shape \[samples, 38\]"):
        infer_reward_contexts_from_dataset(
            object(),
            wrong_shape,
            batch_size=1,
            reference_config_sha256=_REFERENCE_CONFIG_SHA256,
            data_sha256=_DATA_SHA256,
            reward_model_sha256=_REWARD_MODEL_SHA256,
        )
    nonportable = dict(dataset, reward_labels=torch.empty(labels.shape, device="meta"))
    with pytest.raises(ValueError, match="CPU float32"):
        infer_reward_contexts_from_dataset(
            object(),
            nonportable,
            batch_size=1,
            reference_config_sha256=_REFERENCE_CONFIG_SHA256,
            data_sha256=_DATA_SHA256,
            reward_model_sha256=_REWARD_MODEL_SHA256,
        )


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
