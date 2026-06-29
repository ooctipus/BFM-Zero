"""Evaluate a released BFM-Zero checkpoint on all 862 native motion clips."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.agents.utils import set_seed_everywhere

from .evaluation import EXPECTED_BFM_MOTIONS, normalize_tracking_metrics, run_native_tracking
from .specification import BFM_MODEL_PROFILE_DEFAULT, BFM_MODEL_PROFILES


def main() -> None:
    """Load the native model/environment and write complete immutable evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_folder", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint_type", choices=("source", "candidate"), default="source")
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--implementation", required=True)
    parser.add_argument("--training_seed", type=int, required=True)
    parser.add_argument("--evaluation_seed", type=int, required=True)
    parser.add_argument("--checkpoint_transition", type=int, required=True)
    parser.add_argument("--terminal_profile", choices=("native_reference", "correct_terminal"), required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--evaluator_hash", required=True)
    parser.add_argument("--dataset_hash", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--disable_domain_randomization", action="store_true")
    parser.add_argument("--disable_obs_noise", action="store_true")
    parser.add_argument(
        "--model_profile",
        choices=tuple(BFM_MODEL_PROFILES),
        default=BFM_MODEL_PROFILE_DEFAULT,
    )
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    torch.cuda.set_device(torch.device(args.device))
    set_seed_everywhere(args.evaluation_seed)
    if args.checkpoint_type == "source":
        checkpoint = args.checkpoint if args.checkpoint is not None else args.model_folder / "checkpoint"
        model = load_model_from_checkpoint_dir(checkpoint, device="cuda")
        model.to(args.device)
        model.eval()
    else:
        if args.checkpoint is None:
            raise ValueError("Candidate evaluation requires --checkpoint.")
        from .policy import load_candidate_policy

        model = load_candidate_policy(args.checkpoint, args.device, args.model_profile)
    with (args.model_folder / "config.json").open() as stream:
        config = json.load(stream)
    env_options = config["env"]
    env_options["device"] = args.device
    env_options["lafan_tail_path"] = str(args.data_path.resolve())
    env_options["disable_domain_randomization"] = args.disable_domain_randomization
    env_options["disable_obs_noise"] = args.disable_obs_noise
    env_options["hydra_overrides"].append("env.config.headless=True")
    env = HumanoidVerseIsaacConfig(**env_options).build(args.num_envs)[0]
    metrics, duration = run_native_tracking(model, env=env, num_envs=args.num_envs)
    rows = normalize_tracking_metrics(
        metrics,
        implementation=args.implementation,
        training_seed=args.training_seed,
        evaluation_seed=args.evaluation_seed,
        checkpoint_transition=args.checkpoint_transition,
        terminal_profile=args.terminal_profile,
        run_id=args.run_id,
        evaluator_hash=args.evaluator_hash,
        dataset_hash=args.dataset_hash,
    )
    with (args.output_dir / "native_metrics.json").open("x") as stream:
        json.dump({"duration_seconds": duration, "metrics": metrics}, stream, indent=2, default=_json_default)
    with (args.output_dir / "metrics.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema": "forward_backward_phase2_manifest_v1",
        "run_id": args.run_id,
        "implementation": args.implementation,
        "training_seed": args.training_seed,
        "evaluation_seed": args.evaluation_seed,
        "checkpoint_transition": args.checkpoint_transition,
        "terminal_profile": args.terminal_profile,
        "evaluator_hash": args.evaluator_hash,
        "dataset_hash": args.dataset_hash,
        "expected_motion_count": EXPECTED_BFM_MOTIONS,
        "record_count": len(rows),
        "duration_seconds": duration,
        "model_profile": args.model_profile,
    }
    with (args.output_dir / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


if __name__ == "__main__":
    main()
