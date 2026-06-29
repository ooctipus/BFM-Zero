"""Role-specific single-file inference states behind the shared milestone contract."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import safetensors.torch
import torch

from .specification import source_model_config


def _source_inference_modules(model: torch.nn.Module) -> torch.nn.ModuleDict:
    return torch.nn.ModuleDict(
        {
            "actor": model._actor,
            "backward": model._backward_map,
            "normalizer": model._obs_normalizer,
        }
    )


def _candidate_inference_modules(model: torch.nn.Module) -> torch.nn.ModuleDict:
    return torch.nn.ModuleDict(
        {
            "actor": model.actor_network,
            "backward": model.backward_network,
            "normalizers": model.observation_normalizers,
            "action_distribution": model.action_distribution,
        }
    )


def export_source_state(model: torch.nn.Module, destination: Path) -> None:
    """Write only source actor, backward-map, and observation-normalizer tensors."""
    safetensors.torch.save_model(_source_inference_modules(model), destination)


def export_candidate_state(model: torch.nn.Module, destination: Path) -> None:
    """Write only candidate actor, backward-map, normalizer, and distribution tensors."""
    safetensors.torch.save_model(_candidate_inference_modules(model), destination)


def load_source_state(
    reference_config: str | Path,
    checkpoint: str | Path,
    observation_space: Any,
    action_dim: int,
    device: str | torch.device,
    model_profile: str,
) -> Any:
    """Rebuild the declared source topology and load one compact inference state."""
    from humanoidverse.train import TrainConfig

    config = TrainConfig.model_validate_json(Path(reference_config).read_text())
    model_config = source_model_config(config, model_profile, str(device))
    model_observation_space = copy.deepcopy(observation_space)
    if "time" not in model_observation_space.spaces:
        raise ValueError("Source observation space must include the training time field.")
    del model_observation_space.spaces["time"]
    model = model_config.build(model_observation_space, action_dim)
    safetensors.torch.load_model(
        _source_inference_modules(model),
        checkpoint,
        device=str(device),
        strict=True,
    )
    model.to(device)
    model.eval()
    return model


def load_candidate_state(
    model: torch.nn.Module,
    checkpoint: str | Path,
    device: str | torch.device,
) -> None:
    """Load one compact candidate inference state into its declared topology."""
    safetensors.torch.load_model(
        _candidate_inference_modules(model),
        checkpoint,
        device=str(device),
        strict=True,
    )
