"""Load the unified candidate policy behind the native BFM model contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from rsl_rl.models.forward_backward_model import ForwardBackwardModel
from tensordict import TensorDict

from .candidate import candidate_config
from .environment import BFM_ACTION_DIM, BFM_FIELD_WIDTHS
from .specification import BFM_MODEL_PROFILE_DEFAULT


@dataclass
class BFMCandidatePolicy:
    """Translate native observation dictionaries to unified named routes."""

    model: ForwardBackwardModel

    @property
    def _model(self) -> BFMCandidatePolicy:
        """Expose the model boundary expected by the unchanged native evaluator."""
        return self

    @torch.no_grad()
    def backward_map(self, observations: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Encode a native tracking target through the candidate backward map."""
        return self.model.backward_map(self._observations(observations, ("state", "privileged_state")))

    def project_z(self, context: torch.Tensor) -> torch.Tensor:
        """Project contexts with the candidate model's configured geometry."""
        return self.model.context_project(context)

    def context_infer_reward(
        self,
        backward_features: torch.Tensor,
        rewards: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Infer reward contexts through the shared FB integration operator."""
        return self.model.context_infer_reward(backward_features, rewards, weights)

    @torch.no_grad()
    def act(
        self,
        observations: Mapping[str, torch.Tensor],
        context: torch.Tensor,
        mean: bool = True,
    ) -> torch.Tensor:
        """Return candidate actions through the native evaluator boundary."""
        fields = self._observations(observations, ("state", "last_action", "history_actor"))
        return self.model.action_sample(fields, context.to(self.device), deterministic=mean)

    @property
    def device(self) -> torch.device:
        """Return the policy parameter device."""
        return next(self.model.parameters()).device

    def _observations(self, values: Mapping[str, Any], names: tuple[str, ...]) -> TensorDict:
        fields = {name: torch.as_tensor(values[name], dtype=torch.float32, device=self.device) for name in names}
        batch_size = next(iter(fields.values())).shape[0]
        return TensorDict(fields, batch_size=[batch_size], device=self.device)


def resolve_evaluation_checkpoint(
    model_folder: str | Path,
    checkpoint: str | Path | None,
    checkpoint_type: str,
) -> Path:
    """Resolve the exact source directory or candidate file loaded for evaluation."""
    if checkpoint_type == "source":
        return Path(checkpoint) if checkpoint is not None else Path(model_folder) / "checkpoint"
    if checkpoint_type != "candidate":
        raise ValueError(f"Unknown checkpoint type: {checkpoint_type!r}.")
    if checkpoint is None:
        raise ValueError("Candidate evaluation requires a checkpoint.")
    return Path(checkpoint)


def load_evaluation_policy(
    model_folder: str | Path,
    checkpoint: str | Path | None,
    checkpoint_type: str,
    device: str | torch.device,
    model_profile: str = BFM_MODEL_PROFILE_DEFAULT,
) -> Any:
    """Load a source or unified policy behind one native evaluation contract."""
    checkpoint_path = resolve_evaluation_checkpoint(model_folder, checkpoint, checkpoint_type)
    if checkpoint_type == "source":
        from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir

        model = load_model_from_checkpoint_dir(checkpoint_path, device="cuda")
        model.to(device)
        model.eval()
        return model
    return load_candidate_policy(checkpoint_path, device, model_profile)


def load_candidate_policy(
    checkpoint: str | Path,
    device: str | torch.device,
    model_profile: str = BFM_MODEL_PROFILE_DEFAULT,
) -> BFMCandidatePolicy:
    """Construct the released-scale inference topology and load candidate state."""
    device = torch.device(device)
    observations = TensorDict(
        {name: torch.zeros(1, width, device=device) for name, width in BFM_FIELD_WIDTHS.items()},
        batch_size=[1],
        device=device,
    )
    config = candidate_config(lambda *_args, **_kwargs: None, seed=4728, model_profile=model_profile)
    model = ForwardBackwardModel.from_config(
        observations,
        config["obs_groups"],
        BFM_ACTION_DIM,
        config["model"],
    ).to(device)
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    return BFMCandidatePolicy(model)
