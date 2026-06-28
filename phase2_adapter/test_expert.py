"""Contract tests for the BFM-Zero expert bridge."""

from types import SimpleNamespace

import torch
from rsl_rl.models.forward_backward_model import ForwardBackwardObservationSchema
from tensordict import TensorDict

from phase2_adapter.environment import BFM_FIELD_WIDTHS
from phase2_adapter.expert import BFMZeroExpertProvider
from phase2_adapter.specification import observation_routes


def test_expert_provider_reads_the_wrapped_native_motion_environment(tmp_path, monkeypatch) -> None:
    """Expert loading should cross the BFM adapter once and never require an RSL special case."""
    data_path = tmp_path / "motions.pkl"
    data_path.write_bytes(b"fixture")
    native_base = object()
    native = SimpleNamespace(
        _env=native_base,
        _creation_config=SimpleNamespace(lafan_tail_path=str(data_path)),
    )
    env = SimpleNamespace(env=native)
    source = SimpleNamespace(
        storage={
            "observation": {
                "state": torch.zeros(4, BFM_FIELD_WIDTHS["state"]),
                "privileged_state": torch.zeros(4, BFM_FIELD_WIDTHS["privileged_state"]),
            }
        },
        lengths=torch.tensor([4]),
        priorities=torch.tensor([1.0]),
        motion_ids=[0],
    )

    def load_expert(base_env, *_args, **_kwargs):
        assert base_env is native_base
        return source

    monkeypatch.setattr("phase2_adapter.expert.load_expert_trajectories_from_motion_lib", load_expert)
    observations = TensorDict(
        {name: torch.zeros(1, width) for name, width in BFM_FIELD_WIDTHS.items()},
        batch_size=[1],
    )
    schema = ForwardBackwardObservationSchema.from_observations(observations, observation_routes())

    expert = BFMZeroExpertProvider(seed=3)(env, schema, "cpu", window_lengths=(2,))

    assert expert.frames.shape == (
        4,
        BFM_FIELD_WIDTHS["state"] + BFM_FIELD_WIDTHS["privileged_state"],
    )
    assert expert.schema.num_clips == 1
