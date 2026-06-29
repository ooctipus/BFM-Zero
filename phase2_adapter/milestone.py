"""Stream one compact policy state at a time to an external evaluator."""

from __future__ import annotations

import hashlib
import importlib
import random
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import numpy as np
import torch


def _rng_fingerprint() -> str:
    digest = hashlib.sha256()
    digest.update(repr(random.getstate()).encode())
    numpy_state = np.random.get_state()
    digest.update(numpy_state[0].encode())
    digest.update(numpy_state[1].tobytes())
    digest.update(repr(numpy_state[2:]).encode())
    digest.update(torch.get_rng_state().cpu().numpy().tobytes())
    if torch.cuda.is_available():
        for state in torch.cuda.get_rng_state_all():
            digest.update(state.cpu().numpy().tobytes())
    return digest.hexdigest()


def _lifecycle_module() -> ModuleType:
    return importlib.import_module("bfm_phase2f_supervisor")


@dataclass(slots=True)
class MilestoneSink:
    """Apply one-slot backpressure before exporting each compact tensor state."""

    run_root: Path
    run_id: str
    transitions: tuple[int, ...]
    poll_seconds: float = 1.0
    lifecycle: ModuleType | None = field(default=None, repr=False)
    sleep: Callable[[float], None] = field(default=time.sleep, repr=False)

    def __post_init__(self) -> None:
        if not self.run_root.is_absolute() or not self.run_root.is_dir() or not self.run_id:
            raise ValueError("Milestone sink run identity differs.")
        if (
            not self.transitions
            or tuple(sorted(set(self.transitions))) != self.transitions
            or any(type(value) is not int or value < 0 for value in self.transitions)
        ):
            raise ValueError("Milestone sink transitions must be ordered unique nonnegative integers.")
        if not isinstance(self.poll_seconds, (int, float)) or isinstance(self.poll_seconds, bool) or self.poll_seconds <= 0:
            raise ValueError("Milestone sink poll interval must be positive.")
        if self.lifecycle is None:
            self.lifecycle = _lifecycle_module()

    def publish(self, transition: int, export: Callable[[Path], None]) -> Path:
        """Wait before export, prove RNG immutability, and publish one compact file."""
        try:
            index = self.transitions.index(transition)
        except ValueError as error:
            raise ValueError(f"Transition {transition} is outside the milestone schedule.") from error
        previous = self.transitions[index - 1] if index else None
        destination = self.run_root / "compact_states" / f"{transition}.pt"
        handoff = self.run_root / "checkpoint_handoffs" / f"transition_{transition}"
        rng_changed = destination.parent / f"{destination.name}.RNG_CHANGED"
        assert self.lifecycle is not None
        if rng_changed.exists():
            raise RuntimeError("Compact-state export previously changed training RNG state.")
        if handoff.is_dir():
            state = self.lifecycle.load_checkpoint_handoff(self.run_root, transition)
            producer = state["producer"]
            if (
                producer["run_id"] != self.run_id
                or producer["transition"] != transition
                or producer["previous_transition"] != previous
                or producer["final_transition"] != self.transitions[-1]
            ):
                raise ValueError("Existing milestone handoff identity differs.")
            return destination
        while not self.lifecycle.checkpoint_slot_ready(self.run_root, previous):
            self.sleep(float(self.poll_seconds))
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".tmp")
            if temporary.exists():
                raise FileExistsError(f"Stale compact-state export exists: {temporary}.")
            rng_before = _rng_fingerprint()
            export(temporary)
            if not temporary.is_file() or temporary.is_symlink():
                raise ValueError("Compact-state exporter must create exactly one regular file.")
            temporary.replace(destination)
            if _rng_fingerprint() != rng_before:
                rng_changed.touch()
                raise RuntimeError("Compact-state export changed training RNG state.")
        elif not destination.is_file() or destination.is_symlink():
            raise ValueError("Existing compact state is not one regular file.")
        self.lifecycle.publish_checkpoint(
            self.run_root,
            run_id=self.run_id,
            transition=transition,
            checkpoint=destination,
            previous_transition=previous,
            final_transition=self.transitions[-1],
        )
        return destination


def optional_milestone_sink(
    run_root: Path | None,
    run_id: str | None,
    transitions: tuple[int, ...],
    poll_seconds: float,
) -> MilestoneSink | None:
    """Construct the opt-in Phase 2F sink or require both ownership inputs."""
    if run_root is None and run_id is None:
        return None
    if run_root is None or run_id is None:
        raise ValueError("Phase 2F milestone sink requires both run root and run id.")
    return MilestoneSink(run_root.resolve(), run_id, transitions, poll_seconds)


def replace_recovery_file(destination: Path, export: Callable[[Path], None]) -> None:
    """Atomically replace the one rolling file recovery state."""
    temporary = destination.with_suffix(".tmp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if temporary.exists():
        raise FileExistsError(f"Stale recovery export exists: {temporary}.")
    export(temporary)
    if not temporary.is_file() or temporary.is_symlink():
        raise ValueError("Recovery exporter must create one regular file.")
    temporary.replace(destination)


def replace_recovery_directory(destination: Path, export: Callable[[Path], None]) -> None:
    """Replace the one retained directory recovery state through a crash-visible rotation."""
    staging = destination.with_name(f".{destination.name}.staging")
    previous = destination.with_name(f".{destination.name}.previous")
    if staging.exists() or previous.exists():
        raise FileExistsError("Stale recovery-directory rotation requires operator review.")
    export(staging)
    if not staging.is_dir():
        raise ValueError("Recovery exporter must create one directory.")
    if destination.exists():
        destination.rename(previous)
    staging.rename(destination)
    if previous.exists():
        shutil.rmtree(previous)
