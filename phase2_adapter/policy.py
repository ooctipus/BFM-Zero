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


def load_candidate_policy(checkpoint: str | Path, device: str | torch.device) -> BFMCandidatePolicy:
    """Construct the released-scale inference topology and load candidate state."""
    device = torch.device(device)
    observations = TensorDict(
        {name: torch.zeros(1, width, device=device) for name, width in BFM_FIELD_WIDTHS.items()},
        batch_size=[1],
        device=device,
    )
    config = candidate_config(lambda *_args, **_kwargs: None, seed=4728)
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
