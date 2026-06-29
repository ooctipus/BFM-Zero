"""Materialize immutable observations and broad-reward labels for reward inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from humanoidverse.envs.legged_robot_motions.legged_robot_motions import compute_humanoid_observations_max
from humanoidverse.utils.g1_env_config import get_g1_robot_xml_root
from humanoidverse.utils.torch_utils import quat_rotate_inverse

from .candidate import make_native_environment
from .environment import BFM_ACTION_DIM
from .evaluation import artifact_sha256
from .reward_evaluation import (
    BFM_REWARD_INFERENCE_DATASET_SCHEMA,
    BFM_REWARD_TASKS,
    BFM_REWARD_TASKS_SHA256,
    build_reward_inference_dataset,
    motion_sample_indices,
    relabel_reward_tasks,
)


def main() -> None:
    """Write fixed reached observations and policy-independent reward labels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference_config", type=Path, required=True)
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples_per_motion", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reward_workers", type=int, default=12)
    args = parser.parse_args()
    manifest_path = args.output.with_suffix(".json")
    if args.output.exists() or manifest_path.exists():
        raise FileExistsError(f"Reward inference dataset already exists: {args.output}")
    if args.samples_per_motion < 1 or args.reward_workers < 1:
        raise ValueError("samples_per_motion and reward_workers must be positive.")

    env = make_native_environment(
        reference_config=args.reference_config,
        data_path=args.data_path,
        num_envs=1,
        device=args.device,
    )
    base = env.env._env
    motion_lib = base._motion_lib
    motion_lib.load_motions_for_training()
    fields: dict[str, list[torch.Tensor]] = {
        "state": [],
        "privileged_state": [],
        "qpos": [],
        "qvel": [],
        "action": [],
        "motion_id": [],
    }
    try:
        for motion_index in range(motion_lib._num_unique_motions):
            length = int(np.ceil((motion_lib._motion_lengths[motion_index] / base.dt).cpu()))
            times = torch.arange(length, device=base.device) * base.dt
            motion_ids = torch.full((length,), motion_index, dtype=torch.long, device=base.device)
            motion = motion_lib.get_motion_state(motion_ids, times)
            indices = motion_sample_indices(length, args.samples_per_motion, device=base.device)

            body_positions = motion["rg_pos_t"]
            body_rotations = motion["rg_rot_t"]
            body_velocities = motion["body_vel_t"]
            body_angular_velocities = motion["body_ang_vel_t"]
            observation = compute_humanoid_observations_max(
                body_positions,
                body_rotations,
                body_velocities,
                body_angular_velocities,
                local_root_obs=True,
                root_height_obs=base.config.obs.root_height_obs,
            )
            privileged_state = torch.cat(tuple(observation.values()), dim=-1)
            dof_position = motion["dof_pos"]
            dof_velocity = motion["dof_vel"]
            relative_dof_position = dof_position - base.default_dof_pos[0]
            root_rotation = body_rotations[:, 0]
            gravity = base.gravity_vec[0:1].repeat(length, 1)
            projected_gravity = quat_rotate_inverse(root_rotation, gravity, w_last=True)
            root_angular_velocity = body_angular_velocities[:, 0]
            state = torch.cat(
                (relative_dof_position, dof_velocity, projected_gravity, root_angular_velocity),
                dim=-1,
            )
            qpos = torch.cat((motion["root_pos"], motion["root_rot"], dof_position), dim=-1)
            qvel = torch.cat((motion["root_vel"], motion["root_ang_vel"], dof_velocity), dim=-1)
            fields["state"].append(state.index_select(0, indices).cpu())
            fields["privileged_state"].append(privileged_state.index_select(0, indices).cpu())
            fields["qpos"].append(qpos.index_select(0, indices).cpu())
            fields["qvel"].append(qvel.index_select(0, indices).cpu())
            fields["action"].append(torch.zeros(indices.numel(), BFM_ACTION_DIM))
            fields["motion_id"].append(torch.full((indices.numel(),), motion_index, dtype=torch.long))
    finally:
        env.close()

    reference_config_sha256 = artifact_sha256(args.reference_config)
    data_sha256 = artifact_sha256(args.data_path)
    observations = {
        "state": torch.cat(fields["state"]).contiguous(),
        "privileged_state": torch.cat(fields["privileged_state"]).contiguous(),
    }
    qpos = torch.cat(fields["qpos"]).contiguous()
    qvel = torch.cat(fields["qvel"]).contiguous()
    action = torch.cat(fields["action"]).contiguous()
    reward_model_path = get_g1_robot_xml_root() / "scene_29dof_freebase_noadditional_actuators.xml"
    reward_model_sha256 = artifact_sha256(reward_model_path)
    reward_model = mujoco.MjModel.from_xml_path(str(reward_model_path))
    reward_labels = relabel_reward_tasks(
        reward_model,
        BFM_REWARD_TASKS,
        qpos,
        qvel,
        action,
        workers=args.reward_workers,
    )
    payload = build_reward_inference_dataset(
        observations,
        reward_labels,
        torch.cat(fields["motion_id"]).contiguous(),
        reference_config_sha256=reference_config_sha256,
        data_sha256=data_sha256,
        reward_model_sha256=reward_model_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    manifest = {
        "schema": BFM_REWARD_INFERENCE_DATASET_SCHEMA,
        "source": "deterministic_uniform_lafan_expert_reached_states",
        "reference_config_path": str(args.reference_config.resolve()),
        "reference_config_sha256": reference_config_sha256,
        "data_path": str(args.data_path.resolve()),
        "data_sha256": data_sha256,
        "reward_model_path": str(reward_model_path.resolve()),
        "reward_model_sha256": reward_model_sha256,
        "reward_tasks": BFM_REWARD_TASKS,
        "reward_tasks_sha256": BFM_REWARD_TASKS_SHA256,
        "reward_task_count": len(BFM_REWARD_TASKS),
        "motion_count": int(payload["motion_id"].unique().numel()),
        "samples_per_motion": args.samples_per_motion,
        "sample_count": int(payload["motion_id"].numel()),
        "field_shapes": {
            "state": list(payload["observation"]["state"].shape),
            "privileged_state": list(payload["observation"]["privileged_state"].shape),
            "reward_labels": list(payload["reward_labels"].shape),
            "motion_id": list(payload["motion_id"].shape),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
