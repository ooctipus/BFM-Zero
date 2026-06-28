"""Expose native BFM-Zero facts through the RSL-RL vector environment contract."""

from __future__ import annotations

import types
from typing import Any, Literal

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

BFM_ACTION_DIM = 29
BFM_CONTROL_HZ = 50
BFM_FIELD_WIDTHS = {
    "state": 64,
    "last_action": 29,
    "history_actor": 372,
    "privileged_state": 463,
}
BFM_AUXILIARY_EVIDENCE_NAMES = (
    "penalty_torques",
    "penalty_action_rate",
    "limits_dof_pos",
    "limits_torque",
    "penalty_undesired_contact",
    "penalty_feet_ori",
    "penalty_ankle_roll",
    "penalty_slippage",
)
TerminalProfile = Literal["native_reference", "correct_terminal"]


class ExactFinalObservationCapture:
    """Capture the emitted BFM observation immediately before an internal reset."""

    def __init__(self, env: Any, example_observation: dict[str, torch.Tensor]) -> None:
        self.env = env
        self.base_env = env._env
        self.device = torch.device(env.device)
        self.valid = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)
        self.observations = {name: torch.empty_like(example_observation[name]) for name in BFM_FIELD_WIDTHS}
        self.qpos = torch.empty(env.num_envs, 36, dtype=torch.float32, device=self.device)
        self.qvel = torch.empty(env.num_envs, 35, dtype=torch.float32, device=self.device)
        self._original_reset = self.base_env.reset_envs_idx

        def reset_with_capture(_instance: Any, env_ids: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
            if len(env_ids) > 0:
                self._capture(env_ids)
            return self._original_reset(env_ids, *args, **kwargs)

        self.base_env.reset_envs_idx = types.MethodType(reset_with_capture, self.base_env)

    def begin_step(self) -> None:
        """Clear validity without touching dense payload storage."""
        self.valid.zero_()

    def close(self) -> None:
        """Restore the native reset method."""
        self.base_env.reset_envs_idx = self._original_reset

    def _capture(self, env_ids: torch.Tensor) -> None:
        # Observation noise must not advance the behavior RNG stream. The extra
        # observation compute is therefore isolated in a forked RNG context.
        devices = []
        if self.device.type == "cuda":
            device_index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            devices = [device_index]
        with torch.random.fork_rng(devices=devices):
            self.base_env._compute_observations()
            observation = self.env._get_g1env_observation(to_numpy=False)
        qpos, qvel = self.env._get_qpos_qvel(to_numpy=False)
        for name in BFM_FIELD_WIDTHS:
            self.observations[name].index_copy_(0, env_ids, observation[name][env_ids])
        self.qpos.index_copy_(0, env_ids, qpos[env_ids])
        self.qvel.index_copy_(0, env_ids, qvel[env_ids])
        self.valid[env_ids] = True


class BFMZeroVecEnv(VecEnv):
    """Translate BFM asymmetric observations, reward evidence, and same-step resets."""

    def __init__(
        self,
        env: Any,
        *,
        terminal_profile: TerminalProfile,
        device: str | torch.device | None = None,
    ) -> None:
        if terminal_profile not in ("native_reference", "correct_terminal"):
            raise ValueError(f"Unsupported BFM terminal profile: {terminal_profile!r}.")
        self.env = env
        self.device = torch.device(device or env.device)
        self.num_envs = int(env.num_envs)
        self.num_actions = int(env.action_space.shape[-1])
        if self.num_actions != BFM_ACTION_DIM:
            raise ValueError(f"BFM-Zero requires {BFM_ACTION_DIM} actions, got {self.num_actions}.")
        self.max_episode_length = int(env._env.max_episode_length)
        self.episode_length_buf = env._env.episode_length_buf
        self.terminal_profile = terminal_profile
        self.cfg = {
            "adapter": "bfm_zero_phase2_v1",
            "autoreset_mode": "same_step",
            "terminal_profile": terminal_profile,
            "control_hz": BFM_CONTROL_HZ,
            "auxiliary_evidence_names": BFM_AUXILIARY_EVIDENCE_NAMES,
        }
        observation, _info = env.reset(to_numpy=False)
        self._observations = self._convert_observation(observation)
        self._capture = ExactFinalObservationCapture(env, observation) if terminal_profile == "correct_terminal" else None

    def get_observations(self) -> TensorDict:
        """Return the current named asymmetric BFM observations."""
        return self._observations

    def reset(self) -> TensorDict:
        """Reset the shared native environment after a curriculum event."""
        observation, _info = self.env.reset(to_numpy=False)
        self._observations = self._convert_observation(observation)
        if self._capture is not None:
            self._capture.begin_step()
        return self._observations

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Apply actions unchanged and expose native transition evidence."""
        if self._capture is not None:
            self._capture.begin_step()
        observation, reward, terminated, truncated, info = self.env.step(actions, to_numpy=False)
        terminated = terminated.to(self.device).bool().reshape(self.num_envs)
        truncated = truncated.to(self.device).bool().reshape(self.num_envs)
        done = terminated | truncated
        self._observations = self._convert_observation(observation)
        extras: dict[str, Any] = {
            "time_outs": truncated,
            "auxiliary_reward_evidence": self._auxiliary_evidence(info),
            "episode_steps": self.episode_length_buf,
        }
        if self._capture is not None:
            extras["final_obs"] = TensorDict(
                self._capture.observations,
                batch_size=[self.num_envs],
                device=self.device,
            )
            extras["final_obs_valid"] = self._capture.valid
            extras["final_qpos"] = self._capture.qpos
            extras["final_qvel"] = self._capture.qvel
        return self._observations, reward.to(self.device), done, extras

    def close(self) -> None:
        """Restore the reset hook and close the native simulator."""
        if self._capture is not None:
            self._capture.close()
        self.env.close()

    def _convert_observation(self, observation: dict[str, torch.Tensor]) -> TensorDict:
        fields: dict[str, torch.Tensor] = {}
        for name, width in BFM_FIELD_WIDTHS.items():
            value = observation[name].to(self.device)
            if value.shape != (self.num_envs, width):
                raise ValueError(f"BFM observation {name!r} must have shape {(self.num_envs, width)}, got {tuple(value.shape)}.")
            fields[name] = value
        return TensorDict(fields, batch_size=[self.num_envs], device=self.device)

    def _auxiliary_evidence(self, info: dict[str, Any]) -> torch.Tensor:
        try:
            evidence = info["aux_rewards"]
            values = [evidence[name].to(self.device).reshape(self.num_envs) for name in BFM_AUXILIARY_EVIDENCE_NAMES]
        except KeyError as error:
            raise KeyError(f"BFM transition is missing auxiliary evidence {error.args[0]!r}.") from error
        return torch.stack(values, dim=-1)
