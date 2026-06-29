# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure the canonical compiled BFM source learner without copying its loop."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from humanoidverse.agents.utils import set_seed_everywhere
from humanoidverse.train import Workspace

from .source import SourceTrainingObserver, _load_config, _train
from .specification import BFMTrainingSchedule, resolve_model_profile

SOURCE_THROUGHPUT_CONTRACT_SCHEMA = "forward_backward_bfm_source_capacity_throughput_contract_v1"
SOURCE_THROUGHPUT_TIMING_SCHEMA = "bfm_source_throughput_timing_v1"
TRAINING_BFM_COMMIT = "f0495e864ffcf332f346bd7b55e9aa108cb8b38f"
TRAINING_RSL_COMMIT = "087a6051b6603e9543fdeaf209703261801902d5"
MODEL_PROFILE = "residual_6x1024"
TRAINING_SEED = 4728
NUM_ENVS = 1_024
UPDATES_PER_ITERATION = 16
RANDOM_ACTION_TRANSITIONS = 10_240
POLICY_COLLECTION_WITHOUT_UPDATE_ITERATIONS = 1
PRE_UPDATE_ITERATIONS = 11
UPDATE_STABILIZATION_ITERATIONS = 32
MEASUREMENT_ITERATION_FIRST = 43
MEASUREMENT_ITERATION_LAST = 170
MEASUREMENT_ITERATIONS = 128
TOTAL_ITERATIONS = 171
TOTAL_TRANSITIONS = 175_104
MEASUREMENT_TRANSITIONS = 131_072
CHECKPOINT_INTERVAL_TRANSITIONS = 9_600_000
SAVE_INTERVAL_ITERATIONS = 9_375
TIMING_CSV_NAME = "iteration_timing.csv"
TIMING_SUMMARY_NAME = "timing.json"
TIMING_FIELDS = (
    "iteration",
    "start_transition",
    "end_transition",
    "collection_seconds",
    "learning_seconds",
    "iteration_seconds",
)
BOUNDARY_HOOKS = {
    "start": "SourceTrainingObserver.observe_iteration_start",
    "end": "SourceTrainingObserver.observe_iteration_learning_complete",
    "decomposition": "SourceTrainingObserver.observe_iteration_complete",
}
_REQUIRED_SHARED_VALUES = {
    "implementation": "bfm_zero_source",
    "training_seed": TRAINING_SEED,
    "terminal_profile": "correct_terminal",
    "num_envs": NUM_ENVS,
    "updates_per_iteration": UPDATES_PER_ITERATION,
    "random_action_transitions": RANDOM_ACTION_TRANSITIONS,
    "policy_collection_without_update_iterations": POLICY_COLLECTION_WITHOUT_UPDATE_ITERATIONS,
    "pre_update_iterations": PRE_UPDATE_ITERATIONS,
    "update_stabilization_iterations": UPDATE_STABILIZATION_ITERATIONS,
    "measurement_iterations": MEASUREMENT_ITERATIONS,
    "measurement_iteration_first": MEASUREMENT_ITERATION_FIRST,
    "measurement_iteration_last": MEASUREMENT_ITERATION_LAST,
    "total_iterations": TOTAL_ITERATIONS,
    "total_transitions": TOTAL_TRANSITIONS,
    "measurement_transitions": MEASUREMENT_TRANSITIONS,
    "checkpoint_interval_transitions": CHECKPOINT_INTERVAL_TRANSITIONS,
    "save_interval_iterations": SAVE_INTERVAL_ITERATIONS,
    "compile": True,
    "compile_mode": "reduce-overhead",
    "save_initial_evaluation_checkpoint": False,
    "save_final_checkpoint": False,
}
_TIMING_SUMMARY_FIELDS = {
    "schema",
    "contract",
    "contract_sha256",
    "model_profile",
    "measurement_iteration_first",
    "measurement_iteration_last",
    "measurement_iterations",
    "measurement_transitions",
    "boundary_hooks",
    "boundary_synchronizations",
    "measurement_wall_seconds",
    "total_transitions_per_second",
    "iteration_timing",
    "iteration_timing_sha256",
}


@dataclass(frozen=True, slots=True)
class SourceThroughputSpec:
    """Exact source-profile inputs consumed by one capacity probe."""

    contract: Path
    contract_sha256: str
    reference_config: Path
    data_path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source_throughput_spec(contract: Path) -> SourceThroughputSpec:
    """Load one immutable short source-run contract."""
    if not contract.is_absolute() or not contract.is_file():
        raise ValueError(f"Source throughput contract must be an existing absolute file: {contract}.")
    payload = json.loads(contract.read_text())
    if not isinstance(payload, dict) or payload.get("schema") != SOURCE_THROUGHPUT_CONTRACT_SCHEMA:
        raise ValueError("Source throughput contract schema differs.")
    shared = payload.get("shared_run_spec")
    if not isinstance(shared, dict):
        raise ValueError("Source throughput shared run specification differs.")
    for name, expected in _REQUIRED_SHARED_VALUES.items():
        if shared.get(name) != expected:
            raise ValueError(f"Source throughput shared run field differs: {name}.")
    timing = payload.get("timing_contract")
    if not isinstance(timing, dict) or timing.get("boundary_hooks") != BOUNDARY_HOOKS or timing.get("boundary_synchronizations") != 2:
        raise ValueError("Source throughput timing boundary differs.")
    if (
        PRE_UPDATE_ITERATIONS != RANDOM_ACTION_TRANSITIONS // NUM_ENVS + POLICY_COLLECTION_WITHOUT_UPDATE_ITERATIONS
        or MEASUREMENT_ITERATION_FIRST != PRE_UPDATE_ITERATIONS + UPDATE_STABILIZATION_ITERATIONS
        or MEASUREMENT_ITERATION_LAST - MEASUREMENT_ITERATION_FIRST + 1 != MEASUREMENT_ITERATIONS
        or TOTAL_ITERATIONS != PRE_UPDATE_ITERATIONS + UPDATE_STABILIZATION_ITERATIONS + MEASUREMENT_ITERATIONS
        or TOTAL_TRANSITIONS != TOTAL_ITERATIONS * NUM_ENVS
        or MEASUREMENT_TRANSITIONS != MEASUREMENT_ITERATIONS * NUM_ENVS
        or CHECKPOINT_INTERVAL_TRANSITIONS // NUM_ENVS != SAVE_INTERVAL_ITERATIONS
    ):
        raise ValueError("Source throughput iteration algebra differs.")
    profile = payload.get("profile")
    hidden_dim, hidden_layers = resolve_model_profile(MODEL_PROFILE)
    if profile != {
        "model_profile": MODEL_PROFILE,
        "hidden_dim": hidden_dim,
        "hidden_layers": hidden_layers,
    }:
        raise ValueError("Source throughput profile differs.")
    training = payload.get("training_inputs")
    if not isinstance(training, dict):
        raise ValueError("Source throughput training inputs differ.")
    if training.get("base_bfm_commit") != TRAINING_BFM_COMMIT or training.get("base_rsl_rl_commit") != TRAINING_RSL_COMMIT:
        raise ValueError("Source throughput training commit differs.")
    reference = Path(str(training.get("reference_config")))
    data = Path(str(training.get("data")))
    if not reference.is_absolute() or not reference.is_file() or not data.is_absolute() or not data.is_file():
        raise ValueError("Source throughput training input is unavailable.")
    if training.get("reference_config_sha256") != _sha256(reference) or training.get("data_sha256") != _sha256(data):
        raise ValueError("Source throughput training input identity differs.")
    return SourceThroughputSpec(contract, _sha256(contract), reference, data)


class SourceThroughputObserver(SourceTrainingObserver):
    """Record the exact CUDA-synchronized source timing window."""

    def __init__(self, device: str) -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("Source throughput requires a CUDA device.")
        self._rows: list[dict[str, int | float]] = []
        self._measurement_started: float | None = None
        self._measurement_finished: float | None = None
        self._next_iteration = 0

    def observe_iteration_start(self, iteration: int, start_transitions: int) -> None:
        if iteration != self._next_iteration or start_transitions != iteration * NUM_ENVS:
            raise RuntimeError("Source throughput iteration start differs.")
        if iteration == MEASUREMENT_ITERATION_FIRST:
            torch.cuda.synchronize(self.device)
            self._measurement_started = time.monotonic()

    def observe_iteration_learning_complete(self, iteration: int, end_transitions: int) -> None:
        if iteration != self._next_iteration or end_transitions != (iteration + 1) * NUM_ENVS:
            raise RuntimeError("Source throughput learning boundary differs.")
        if iteration == MEASUREMENT_ITERATION_LAST:
            torch.cuda.synchronize(self.device)
            self._measurement_finished = time.monotonic()

    def observe_iteration_complete(
        self,
        iteration: int,
        end_transitions: int,
        collection_seconds: float,
        learning_seconds: float,
    ) -> None:
        if iteration != self._next_iteration or end_transitions != (iteration + 1) * NUM_ENVS:
            raise RuntimeError("Source throughput iteration completion differs.")
        values = (collection_seconds, learning_seconds)
        if any(
            not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0.0 for value in values
        ):
            raise ValueError("Source throughput decomposition differs.")
        if MEASUREMENT_ITERATION_FIRST <= iteration <= MEASUREMENT_ITERATION_LAST:
            self._rows.append(
                {
                    "iteration": iteration,
                    "start_transition": iteration * NUM_ENVS,
                    "end_transition": end_transitions,
                    "collection_seconds": float(collection_seconds),
                    "learning_seconds": float(learning_seconds),
                    "iteration_seconds": float(collection_seconds + learning_seconds),
                }
            )
        self._next_iteration += 1

    def write(self, output: Path, spec: SourceThroughputSpec) -> dict[str, object]:
        """Publish the complete timing evidence atomically."""
        if (
            not output.is_absolute()
            or not output.is_dir()
            or self._next_iteration != TOTAL_ITERATIONS
            or len(self._rows) != MEASUREMENT_ITERATIONS
            or self._measurement_started is None
            or self._measurement_finished is None
        ):
            raise ValueError("Source throughput timing window is incomplete.")
        wall = self._measurement_finished - self._measurement_started
        if not math.isfinite(wall) or wall <= 0.0:
            raise ValueError("Source throughput wall time differs.")
        csv_path = output / TIMING_CSV_NAME
        summary_path = output / TIMING_SUMMARY_NAME
        csv_staging = output / "iteration_timing.tmp"
        summary_staging = output / "timing.tmp"
        if any(path.exists() for path in (csv_path, summary_path, csv_staging, summary_staging)):
            raise FileExistsError("Source throughput timing output is not fresh.")
        with csv_staging.open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=TIMING_FIELDS)
            writer.writeheader()
            writer.writerows(self._rows)
        summary = {
            "schema": SOURCE_THROUGHPUT_TIMING_SCHEMA,
            "contract": str(spec.contract),
            "contract_sha256": spec.contract_sha256,
            "model_profile": MODEL_PROFILE,
            "measurement_iteration_first": MEASUREMENT_ITERATION_FIRST,
            "measurement_iteration_last": MEASUREMENT_ITERATION_LAST,
            "measurement_iterations": MEASUREMENT_ITERATIONS,
            "measurement_transitions": MEASUREMENT_TRANSITIONS,
            "boundary_hooks": BOUNDARY_HOOKS,
            "boundary_synchronizations": 2,
            "measurement_wall_seconds": wall,
            "total_transitions_per_second": MEASUREMENT_TRANSITIONS / wall,
            "iteration_timing": TIMING_CSV_NAME,
            "iteration_timing_sha256": _sha256(csv_staging),
        }
        with summary_staging.open("x") as stream:
            json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        csv_staging.replace(csv_path)
        summary_staging.replace(summary_path)
        return summary


def load_source_throughput_timing(output: Path, spec: SourceThroughputSpec) -> dict[str, object]:
    """Revalidate one source timing artifact and its no-checkpoint law."""
    csv_path = output / TIMING_CSV_NAME
    summary_path = output / TIMING_SUMMARY_NAME
    if not output.is_absolute() or not csv_path.is_file() or not summary_path.is_file():
        raise ValueError("Source throughput timing artifact is incomplete.")
    summary = json.loads(summary_path.read_text())
    if not isinstance(summary, dict) or set(summary) != _TIMING_SUMMARY_FIELDS:
        raise ValueError("Source throughput timing summary differs.")
    identity = {
        "schema": SOURCE_THROUGHPUT_TIMING_SCHEMA,
        "contract": str(spec.contract),
        "contract_sha256": spec.contract_sha256,
        "model_profile": MODEL_PROFILE,
        "measurement_iteration_first": MEASUREMENT_ITERATION_FIRST,
        "measurement_iteration_last": MEASUREMENT_ITERATION_LAST,
        "measurement_iterations": MEASUREMENT_ITERATIONS,
        "measurement_transitions": MEASUREMENT_TRANSITIONS,
        "boundary_hooks": BOUNDARY_HOOKS,
        "boundary_synchronizations": 2,
        "iteration_timing": TIMING_CSV_NAME,
        "iteration_timing_sha256": _sha256(csv_path),
    }
    if any(summary.get(name) != value for name, value in identity.items()):
        raise ValueError("Source throughput timing identity differs.")
    with csv_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != TIMING_FIELDS:
            raise ValueError("Source throughput timing CSV differs.")
        rows = list(reader)
    if len(rows) != MEASUREMENT_ITERATIONS:
        raise ValueError("Source throughput timing row count differs.")
    for offset, row in enumerate(rows):
        iteration = MEASUREMENT_ITERATION_FIRST + offset
        collection = float(row["collection_seconds"])
        learning = float(row["learning_seconds"])
        total = float(row["iteration_seconds"])
        if (
            int(row["iteration"]) != iteration
            or int(row["start_transition"]) != iteration * NUM_ENVS
            or int(row["end_transition"]) != (iteration + 1) * NUM_ENVS
            or not all(math.isfinite(value) and value >= 0.0 for value in (collection, learning, total))
            or total != collection + learning
        ):
            raise ValueError("Source throughput timing row differs.")
    wall = summary.get("measurement_wall_seconds")
    rate = summary.get("total_transitions_per_second")
    if (
        not isinstance(wall, (int, float))
        or isinstance(wall, bool)
        or not math.isfinite(wall)
        or wall <= 0.0
        or not isinstance(rate, (int, float))
        or isinstance(rate, bool)
        or rate != MEASUREMENT_TRANSITIONS / wall
    ):
        raise ValueError("Source throughput synchronized total differs.")
    curriculum = output / "curriculum_events/0.json"
    if not curriculum.is_file():
        raise ValueError("Source throughput lacks transition-zero curriculum evidence.")
    forbidden = [
        path
        for path in output.rglob("*")
        if path.is_file()
        and (
            path.suffix in {".pt", ".safetensors"}
            or path.name.startswith("events.out.tfevents")
            or "checkpoint" in path.parts
            or "evaluation_checkpoints" in path.parts
        )
    ]
    if forbidden:
        raise ValueError("Source throughput contains forbidden training artifacts.")
    return summary


def run_source_throughput(
    contract: Path,
    output: Path,
    *,
    device: str = "cuda:0",
) -> dict[str, object]:
    """Construct and run the canonical compiled source learner once."""
    spec = load_source_throughput_spec(contract)
    if not output.is_absolute() or output.exists():
        raise ValueError(f"Source throughput output must be a fresh absolute path: {output}.")
    args = argparse.Namespace(
        reference_config=spec.reference_config,
        data_path=spec.data_path,
        output_dir=output,
        transitions=TOTAL_TRANSITIONS,
        seed=TRAINING_SEED,
        num_envs=NUM_ENVS,
        device=device,
        log_every_transitions=CHECKPOINT_INTERVAL_TRANSITIONS,
        evaluation_checkpoint_every_transitions=CHECKPOINT_INTERVAL_TRANSITIONS,
        save_initial_evaluation_checkpoint=False,
        compile=True,
        model_profile=MODEL_PROFILE,
    )
    schedule = BFMTrainingSchedule(
        total_iterations=TOTAL_ITERATIONS,
        save_interval=SAVE_INTERVAL_ITERATIONS,
        save_initial_evaluation_checkpoint=False,
    )
    torch.cuda.set_device(torch.device(device))
    set_seed_everywhere(TRAINING_SEED)
    workspace = Workspace(_load_config(args, schedule))
    observer = SourceThroughputObserver(device)
    _train(
        workspace,
        schedule,
        observer=observer,
        save_final_checkpoint=False,
    )
    observer.write(output, spec)
    return load_source_throughput_timing(output, spec)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    """Run one immutable source throughput profile."""
    args = _parse_args()
    result = run_source_throughput(
        args.contract.resolve(),
        args.output_dir.resolve(),
        device=args.device,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
