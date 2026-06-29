# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure one exact BFM candidate window without training artifacts or a copied loop."""

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

from .candidate import (
    BFMEvaluationCheckpointRunner,
    build_candidate_runner,
    initialize_candidate_runner,
)
from .specification import BFM_MODEL_PROFILES, BFMTrainingSchedule, resolve_model_profile

THROUGHPUT_CONTRACT_SCHEMA = "forward_backward_bfm_capacity_throughput_contract_v2"
THROUGHPUT_TIMING_SCHEMA = "bfm_candidate_throughput_timing_v1"
TRAINING_SEED = 4728
NUM_ENVS = 1_024
MOTION_COUNT = 862
NUM_STEPS_PER_ENV = 1
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
    "start": "OffPolicyRunner._observe_iteration_start",
    "end": "OffPolicyRunner._observe_iteration_learning_complete",
    "decomposition": "OffPolicyRunner._observe_iteration_complete",
}
_REQUIRED_SHARED_VALUES = {
    "implementation": "rsl_rl_candidate",
    "training_seed": TRAINING_SEED,
    "terminal_profile": "correct_terminal",
    "num_envs": NUM_ENVS,
    "num_steps_per_env": NUM_STEPS_PER_ENV,
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
    "float32_matmul_precision": "high",
    "initialization_type": "orthogonal",
    "normalization_type": "exponential",
    "normalization_momentum": 0.01,
    "only_varied_field": "model_profile",
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
class ThroughputSpec:
    """Exact contract fields consumed by one no-checkpoint candidate run."""

    contract: Path
    contract_sha256: str
    model_profile: str
    reference_config: Path
    data_path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_throughput_spec(contract: Path, model_profile: str) -> ThroughputSpec:
    """Load the immutable contract and require the exact short-run semantics."""
    if not contract.is_absolute() or not contract.is_file():
        raise ValueError(f"Throughput contract must be an existing absolute file: {contract}.")
    payload = json.loads(contract.read_text())
    if not isinstance(payload, dict) or payload.get("schema") != THROUGHPUT_CONTRACT_SCHEMA:
        raise ValueError("Throughput contract schema differs.")
    shared = payload.get("shared_run_spec")
    if not isinstance(shared, dict):
        raise ValueError("Throughput shared run specification differs.")
    for name, expected in _REQUIRED_SHARED_VALUES.items():
        if shared.get(name) != expected:
            raise ValueError(f"Throughput shared run field differs: {name}.")
    timing = payload.get("timing_contract")
    if not isinstance(timing, dict) or timing.get("boundary_hooks") != BOUNDARY_HOOKS or timing.get("boundary_synchronizations") != 2:
        raise ValueError("Throughput timing-boundary contract differs.")
    if (
        PRE_UPDATE_ITERATIONS != RANDOM_ACTION_TRANSITIONS // NUM_ENVS + POLICY_COLLECTION_WITHOUT_UPDATE_ITERATIONS
        or MEASUREMENT_ITERATION_FIRST != PRE_UPDATE_ITERATIONS + UPDATE_STABILIZATION_ITERATIONS
        or MEASUREMENT_ITERATION_LAST - MEASUREMENT_ITERATION_FIRST + 1 != MEASUREMENT_ITERATIONS
        or TOTAL_ITERATIONS != PRE_UPDATE_ITERATIONS + UPDATE_STABILIZATION_ITERATIONS + MEASUREMENT_ITERATIONS
        or TOTAL_TRANSITIONS != TOTAL_ITERATIONS * NUM_ENVS
        or MEASUREMENT_TRANSITIONS != MEASUREMENT_ITERATIONS * NUM_ENVS
        or CHECKPOINT_INTERVAL_TRANSITIONS // NUM_ENVS != SAVE_INTERVAL_ITERATIONS
    ):
        raise ValueError("Throughput iteration algebra differs.")
    profiles = payload.get("profiles")
    if not isinstance(profiles, list):
        raise ValueError("Throughput profile table differs.")
    selected = [profile for profile in profiles if isinstance(profile, dict) and profile.get("model_profile") == model_profile]
    if len(selected) != 1:
        raise ValueError(f"Throughput contract must contain one profile {model_profile!r}.")
    hidden_dim, hidden_layers = resolve_model_profile(model_profile)
    if selected[0].get("hidden_dim") != hidden_dim or selected[0].get("hidden_layers") != hidden_layers:
        raise ValueError("Throughput profile topology differs.")
    training = payload.get("training_inputs")
    if not isinstance(training, dict):
        raise ValueError("Throughput training inputs differ.")
    reference_config = Path(str(training.get("reference_config")))
    data_path = Path(str(training.get("data")))
    if not reference_config.is_absolute() or not reference_config.is_file() or not data_path.is_absolute() or not data_path.is_file():
        raise ValueError("Throughput reference config or motion data is unavailable.")
    if training.get("data_sha256") != _sha256(data_path):
        raise ValueError("Throughput motion-data identity differs.")
    return ThroughputSpec(
        contract=contract,
        contract_sha256=_sha256(contract),
        model_profile=model_profile,
        reference_config=reference_config,
        data_path=data_path,
    )


class BFMThroughputRunner(BFMEvaluationCheckpointRunner):
    """Observe one fixed CUDA-synchronized timing window in the canonical loop."""

    _timing_rows: list[dict[str, int | float]]
    _measurement_started: float | None
    _measurement_finished: float | None
    _next_iteration: int

    def prepare_timing(self) -> None:
        """Initialize timing state at the only accepted fresh-run boundary."""
        if self.current_learning_iteration != 0 or self.collected_transitions != 0:
            raise ValueError("Throughput timing requires a fresh candidate runner.")
        if self.env.num_envs != NUM_ENVS or torch.device(self.device).type != "cuda":
            raise ValueError("Throughput timing requires the frozen CUDA vector shape.")
        if hasattr(self, "_timing_rows"):
            raise ValueError("Throughput timing was already initialized.")
        self._timing_rows = []
        self._measurement_started = None
        self._measurement_finished = None
        self._next_iteration = 0

    def _observe_iteration_start(self, iteration: int, start_transitions: int) -> None:
        if not hasattr(self, "_timing_rows") or iteration != self._next_iteration:
            raise RuntimeError("Throughput iteration start order differs.")
        if start_transitions != iteration * NUM_ENVS:
            raise RuntimeError("Throughput iteration start transition differs.")
        if iteration == MEASUREMENT_ITERATION_FIRST:
            torch.cuda.synchronize(torch.device(self.device))
            self._measurement_started = time.monotonic()

    def _observe_iteration_learning_complete(self, iteration: int, end_transitions: int) -> None:
        if not hasattr(self, "_timing_rows") or iteration != self._next_iteration:
            raise RuntimeError("Throughput learning boundary order differs.")
        if end_transitions != (iteration + 1) * NUM_ENVS:
            raise RuntimeError("Throughput iteration end transition differs.")
        if iteration == MEASUREMENT_ITERATION_LAST:
            torch.cuda.synchronize(torch.device(self.device))
            self._measurement_finished = time.monotonic()

    def _observe_iteration_complete(
        self,
        iteration: int,
        end_transitions: int,
        collect_time: float,
        learn_time: float,
    ) -> None:
        if not hasattr(self, "_timing_rows") or iteration != self._next_iteration:
            raise RuntimeError("Throughput iteration completion order differs.")
        if end_transitions != (iteration + 1) * NUM_ENVS:
            raise RuntimeError("Throughput completed transition differs.")
        if any(
            not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0.0
            for value in (collect_time, learn_time)
        ):
            raise ValueError("Throughput decomposition time must be finite and nonnegative.")
        if MEASUREMENT_ITERATION_FIRST <= iteration <= MEASUREMENT_ITERATION_LAST:
            self._timing_rows.append(
                {
                    "iteration": iteration,
                    "start_transition": iteration * NUM_ENVS,
                    "end_transition": end_transitions,
                    "collection_seconds": float(collect_time),
                    "learning_seconds": float(learn_time),
                    "iteration_seconds": float(collect_time + learn_time),
                }
            )
        self._next_iteration += 1

    def write_timing(self, output_dir: Path, spec: ThroughputSpec) -> dict[str, object]:
        """Atomically publish the exact timing rows and synchronized total."""
        if not output_dir.is_absolute() or not output_dir.is_dir():
            raise ValueError(f"Throughput output must be an existing absolute directory: {output_dir}.")
        if (
            self._next_iteration != TOTAL_ITERATIONS
            or len(self._timing_rows) != MEASUREMENT_ITERATIONS
            or self._measurement_started is None
            or self._measurement_finished is None
        ):
            raise ValueError("Throughput timing window is incomplete.")
        wall_seconds = self._measurement_finished - self._measurement_started
        if not math.isfinite(wall_seconds) or wall_seconds <= 0.0:
            raise ValueError("Throughput synchronized wall time must be finite and positive.")
        csv_path = output_dir / TIMING_CSV_NAME
        summary_path = output_dir / TIMING_SUMMARY_NAME
        csv_temporary = csv_path.with_suffix(".tmp")
        summary_temporary = summary_path.with_suffix(".tmp")
        if any(path.exists() for path in (csv_path, summary_path, csv_temporary, summary_temporary)):
            raise FileExistsError("Throughput timing output or staging already exists.")
        with csv_temporary.open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=TIMING_FIELDS)
            writer.writeheader()
            writer.writerows(self._timing_rows)
        summary = {
            "schema": THROUGHPUT_TIMING_SCHEMA,
            "contract": str(spec.contract),
            "contract_sha256": spec.contract_sha256,
            "model_profile": spec.model_profile,
            "measurement_iteration_first": MEASUREMENT_ITERATION_FIRST,
            "measurement_iteration_last": MEASUREMENT_ITERATION_LAST,
            "measurement_iterations": MEASUREMENT_ITERATIONS,
            "measurement_transitions": MEASUREMENT_TRANSITIONS,
            "boundary_hooks": BOUNDARY_HOOKS,
            "boundary_synchronizations": 2,
            "measurement_wall_seconds": wall_seconds,
            "total_transitions_per_second": MEASUREMENT_TRANSITIONS / wall_seconds,
            "iteration_timing": TIMING_CSV_NAME,
            "iteration_timing_sha256": _sha256(csv_temporary),
        }
        with summary_temporary.open("x") as stream:
            json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        csv_temporary.replace(csv_path)
        summary_temporary.replace(summary_path)
        return summary


def load_throughput_timing(output_dir: Path, spec: ThroughputSpec) -> dict[str, object]:
    """Recompute one bounded no-checkpoint timing artifact."""
    csv_path = output_dir / TIMING_CSV_NAME
    summary_path = output_dir / TIMING_SUMMARY_NAME
    if (
        not output_dir.is_absolute()
        or not output_dir.is_dir()
        or not csv_path.is_file()
        or not summary_path.is_file()
        or (output_dir / "iteration_timing.tmp").exists()
        or (output_dir / "timing.tmp").exists()
    ):
        raise ValueError("Throughput timing artifact is incomplete.")
    summary = json.loads(summary_path.read_text())
    if not isinstance(summary, dict) or set(summary) != _TIMING_SUMMARY_FIELDS:
        raise ValueError("Throughput timing summary fields differ.")
    if not spec.contract.is_file() or _sha256(spec.contract) != spec.contract_sha256:
        raise ValueError("Throughput contract changed after preparation.")
    expected_identity = {
        "schema": THROUGHPUT_TIMING_SCHEMA,
        "contract": str(spec.contract),
        "contract_sha256": spec.contract_sha256,
        "model_profile": spec.model_profile,
        "measurement_iteration_first": MEASUREMENT_ITERATION_FIRST,
        "measurement_iteration_last": MEASUREMENT_ITERATION_LAST,
        "measurement_iterations": MEASUREMENT_ITERATIONS,
        "measurement_transitions": MEASUREMENT_TRANSITIONS,
        "boundary_hooks": BOUNDARY_HOOKS,
        "boundary_synchronizations": 2,
        "iteration_timing": TIMING_CSV_NAME,
        "iteration_timing_sha256": _sha256(csv_path),
    }
    if any(summary.get(name) != value for name, value in expected_identity.items()):
        raise ValueError("Throughput timing identity differs.")
    with csv_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != TIMING_FIELDS:
            raise ValueError("Throughput timing CSV fields differ.")
        rows = list(reader)
    if len(rows) != MEASUREMENT_ITERATIONS:
        raise ValueError("Throughput timing row count differs.")
    for offset, row in enumerate(rows):
        iteration = MEASUREMENT_ITERATION_FIRST + offset
        if (
            int(row["iteration"]) != iteration
            or int(row["start_transition"]) != iteration * NUM_ENVS
            or int(row["end_transition"]) != (iteration + 1) * NUM_ENVS
        ):
            raise ValueError("Throughput timing row grid differs.")
        collection = float(row["collection_seconds"])
        learning = float(row["learning_seconds"])
        total = float(row["iteration_seconds"])
        if not all(math.isfinite(value) and value >= 0.0 for value in (collection, learning, total)) or total != collection + learning:
            raise ValueError("Throughput timing row values differ.")
    wall_seconds = summary["measurement_wall_seconds"]
    transitions_per_second = summary["total_transitions_per_second"]
    if (
        not isinstance(wall_seconds, (int, float))
        or isinstance(wall_seconds, bool)
        or not math.isfinite(wall_seconds)
        or wall_seconds <= 0.0
        or not isinstance(transitions_per_second, (int, float))
        or isinstance(transitions_per_second, bool)
        or transitions_per_second != MEASUREMENT_TRANSITIONS / wall_seconds
    ):
        raise ValueError("Throughput synchronized total differs.")
    curriculum_dir = output_dir / "curriculum_events"
    curriculum = curriculum_dir / "0.json"
    if not curriculum_dir.is_dir() or {path.name for path in curriculum_dir.iterdir()} != {"0.json"} or not curriculum.is_file():
        raise ValueError("Throughput run lacks its canonical transition-zero curriculum evidence.")
    curriculum_payload = json.loads(curriculum.read_text())
    if (
        not isinstance(curriculum_payload, dict)
        or set(curriculum_payload) != {"schema", "transition", "duration_seconds", "priorities", "metrics"}
        or curriculum_payload.get("schema") != "bfm_zero_curriculum_event_v1"
        or curriculum_payload.get("transition") != 0
        or not isinstance(curriculum_payload.get("duration_seconds"), (int, float))
        or isinstance(curriculum_payload.get("duration_seconds"), bool)
        or not math.isfinite(curriculum_payload["duration_seconds"])
        or curriculum_payload["duration_seconds"] < 0.0
        or not isinstance(curriculum_payload.get("priorities"), list)
        or len(curriculum_payload["priorities"]) != MOTION_COUNT
        or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            for value in curriculum_payload["priorities"]
        )
        or not isinstance(curriculum_payload.get("metrics"), dict)
        or len(curriculum_payload["metrics"]) != MOTION_COUNT
    ):
        raise ValueError("Throughput transition-zero curriculum evidence differs.")
    if any(path.suffix == ".pt" for path in output_dir.rglob("*") if path.is_file()):
        raise ValueError("Throughput run contains a forbidden tensor artifact.")
    if any(path.name.startswith("events.out.tfevents") for path in output_dir.rglob("*") if path.is_file()):
        raise ValueError("Throughput run contains forbidden logger output.")
    return summary


def run_throughput(
    contract: Path,
    output_dir: Path,
    model_profile: str,
    *,
    device: str = "cuda:0",
) -> dict[str, object]:
    """Run the canonical candidate and publish only curriculum and timing evidence."""
    spec = load_throughput_spec(contract, model_profile)
    if not output_dir.is_absolute() or output_dir.exists():
        raise ValueError(f"Throughput output must be a new absolute directory: {output_dir}.")
    output_dir.mkdir(parents=True)
    args = argparse.Namespace(
        reference_config=spec.reference_config,
        data_path=spec.data_path,
        output_dir=output_dir,
        transitions=TOTAL_TRANSITIONS,
        seed=TRAINING_SEED,
        num_envs=NUM_ENVS,
        device=device,
        model_profile=model_profile,
    )
    schedule = BFMTrainingSchedule(
        total_iterations=TOTAL_ITERATIONS,
        save_interval=SAVE_INTERVAL_ITERATIONS,
        save_initial_evaluation_checkpoint=False,
    )
    env, runner = build_candidate_runner(
        args,
        schedule,
        None,
        runner_class=BFMThroughputRunner,
        artifacts_enabled=False,
        logging_enabled=False,
    )
    try:
        if not isinstance(runner, BFMThroughputRunner):
            raise TypeError("Canonical builder returned the wrong throughput runner type.")
        initialize_candidate_runner(runner, output_dir / "evaluation_checkpoints", schedule)
        runner.prepare_timing()
        runner.learn(schedule.total_iterations)
        runner.write_timing(output_dir, spec)
        return load_throughput_timing(output_dir, spec)
    finally:
        env.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_profile", choices=tuple(BFM_MODEL_PROFILES), required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    """Run one immutable candidate throughput profile."""
    args = _parse_args()
    summary = run_throughput(
        args.contract.resolve(),
        args.output_dir.resolve(),
        args.model_profile,
        device=args.device,
    )
    print(json.dumps(summary, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
