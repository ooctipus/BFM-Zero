"""Translate the native BFM motion corpus into the immutable RSL-RL corpus."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
from rsl_rl.models.forward_backward_model import ForwardBackwardObservationSchema
from rsl_rl.storage.forward_backward_expert import ForwardBackwardExpertBuffer, ForwardBackwardExpertSchema

from humanoidverse.agents.envs.humanoidverse_isaac import load_expert_trajectories_from_motion_lib


@dataclass(frozen=True, slots=True)
class BFMZeroExpertProvider:
    """Load the same motion-library expert states used by released BFM-Zero."""

    dataset_id: str = "bfm-zero-lafan-862"
    seed: int = 0

    def __call__(
        self,
        env: object,
        observation_schema: ForwardBackwardObservationSchema,
        device: str,
        *,
        window_lengths: tuple[int, ...],
    ) -> ForwardBackwardExpertBuffer:
        """Translate source expert state without carrying its sampling buffer."""
        source = load_expert_trajectories_from_motion_lib(
            env._env,
            SimpleNamespace(model=SimpleNamespace(seq_length=max(window_lengths))),
            device=device,
            add_history_noaction=False,
        )
        observations = source.storage["observation"]
        frames = torch.cat((observations["state"], observations["privileged_state"]), dim=-1)
        expected_width = observation_schema.route_width("backward")
        if frames.shape[1] != expected_width:
            raise ValueError(f"BFM expert width must be {expected_width}, got {frames.shape[1]}.")
        lengths = source.lengths.to(device=device, dtype=torch.long)
        clip_offsets = torch.cat((torch.zeros(1, device=device, dtype=torch.long), lengths.cumsum(dim=0)))
        if clip_offsets[-1] != frames.shape[0]:
            raise ValueError("BFM expert clip lengths do not span the frame tensor.")
        priorities = source.priorities.to(device=device, dtype=torch.float32)
        data_path = Path(env._creation_config.lafan_tail_path)
        data_hash = _file_hash(data_path)
        offsets_hash = hashlib.sha256(clip_offsets.cpu().numpy().tobytes()).hexdigest()
        schema = ForwardBackwardExpertSchema(
            dataset_id=self.dataset_id,
            data_hash=data_hash,
            feature_schema_hash=observation_schema.schema_hash,
            clip_offsets_hash=offsets_hash,
            expert_feature_width=expected_width,
            num_frames=frames.shape[0],
            num_clips=lengths.shape[0],
            window_lengths=window_lengths,
        )
        return ForwardBackwardExpertBuffer(frames, clip_offsets, priorities, schema, seed=self.seed)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
