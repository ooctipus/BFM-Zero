"""Call the unchanged BFM evaluator and translate only its numeric records."""

from __future__ import annotations

import hashlib
import math
import numbers
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Mapping, TypeVar

from humanoidverse.agents.utils import set_seed_everywhere

EXPECTED_BFM_MOTIONS = 862
EVALUATION_RNG_PROTOCOL = "seed_immediately_before_environment_construction_v1"
_EnvironmentT = TypeVar("_EnvironmentT")


def artifact_sha256(path: str | Path) -> str:
    """Hash one evaluation file or directory tree with stable relative paths."""
    path = Path(path)
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if not path.is_dir():
        raise FileNotFoundError(f"Evaluation artifact does not exist: {path}")
    digest = hashlib.sha256(b"directory\0")
    for child in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(child.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(artifact_sha256(child)))
    return digest.hexdigest()


def build_evaluation_environment(
    factory: Callable[[], _EnvironmentT],
    *,
    seed: int,
) -> _EnvironmentT:
    """Give environment construction sole ownership of evaluation RNG state."""
    set_seed_everywhere(seed)
    return factory()


def run_native_tracking(
    agent_or_model: Any,
    *,
    env: Any,
    num_envs: int,
) -> tuple[dict[str, Any], float]:
    """Run the unchanged native BFM tracking evaluator."""
    from humanoidverse.agents.evaluations.humanoidverse_isaac import (
        HumanoidVerseIsaacTrackingEvaluation,
        HumanoidVerseIsaacTrackingEvaluationConfig,
    )

    evaluator = HumanoidVerseIsaacTrackingEvaluation(
        HumanoidVerseIsaacTrackingEvaluationConfig(
            env=None,
            num_envs=num_envs,
            n_episodes_per_motion=1,
            include_results_from_all_envs=False,
            disable_tqdm=True,
        )
    )
    start = time.perf_counter()
    metrics, _aggregate = evaluator.run(
        timestep=0,
        agent_or_model=agent_or_model,
        logger=None,
        env=env,
    )
    return metrics, time.perf_counter() - start


def normalize_tracking_metrics(
    metrics: Mapping[str, Mapping[str, Any]],
    *,
    implementation: str,
    training_seed: int,
    evaluation_seed: int,
    checkpoint_transition: int,
    terminal_profile: str,
    run_id: str,
    evaluator_hash: str,
    dataset_hash: str,
    expected_motion_count: int = EXPECTED_BFM_MOTIONS,
) -> list[dict[str, object]]:
    """Convert native scalar metrics into the frozen long-form columns."""
    if len(metrics) != expected_motion_count:
        raise ValueError(f"Expected {expected_motion_count} BFM motions, got {len(metrics)}.")
    if terminal_profile not in ("native_reference", "correct_terminal"):
        raise ValueError(f"Unsupported BFM terminal profile: {terminal_profile!r}.")
    rows: list[dict[str, object]] = []
    metric_names: set[str] | None = None
    motion_ids: set[str] = set()
    for motion_name, values in metrics.items():
        motion_id = str(values["motion_id"])
        if motion_id in motion_ids:
            raise ValueError(f"Duplicate BFM motion id: {motion_id}")
        motion_ids.add(motion_id)
        numeric = {
            name: float(value)
            for name, value in values.items()
            if name not in ("motion_id", "motion_file") and isinstance(value, numbers.Number)
        }
        if not numeric or any(not math.isfinite(value) for value in numeric.values()):
            raise ValueError(f"Motion {motion_name!r} has empty or nonfinite metrics.")
        if metric_names is None:
            metric_names = set(numeric)
        elif set(numeric) != metric_names:
            raise ValueError("Every BFM motion must emit the same scalar metric names.")
        rows.extend(
            {
                "implementation": implementation,
                "training_seed": training_seed,
                "evaluation_seed": evaluation_seed,
                "motion_id": motion_id,
                "checkpoint_transition": checkpoint_transition,
                "terminal_profile": terminal_profile,
                "metric_name": name,
                "metric_value": value,
                "run_id": run_id,
                "evaluator_hash": evaluator_hash,
                "dataset_hash": dataset_hash,
            }
            for name, value in sorted(numeric.items())
        )
    return rows
