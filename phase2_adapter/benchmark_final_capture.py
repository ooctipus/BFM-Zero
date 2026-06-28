"""Benchmark BFM environment stepping with and without exact final capture."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch

from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig

from .environment import BFM_ACTION_DIM, BFMZeroVecEnv


def main() -> None:
    """Measure full-vector throughput and terminal evidence for one profile."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference_config", type=Path, required=True)
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--terminal_profile", choices=("native_reference", "correct_terminal"), required=True)
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--warmup_steps", type=int, default=25)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    with args.reference_config.open() as stream:
        config = json.load(stream)
    env_options = copy.deepcopy(config["env"])
    env_options["device"] = args.device
    env_options["lafan_tail_path"] = str(args.data_path.resolve())
    native = HumanoidVerseIsaacConfig(**env_options).build(args.num_envs)[0]
    env = BFMZeroVecEnv(native, terminal_profile=args.terminal_profile, device=args.device)
    actions = torch.zeros(args.num_envs, BFM_ACTION_DIM, device=args.device)

    for _ in range(args.warmup_steps):
        env.step(actions)
    torch.cuda.synchronize()
    done_count = torch.zeros((), dtype=torch.long, device=args.device)
    final_count = torch.zeros_like(done_count)
    missing_final_count = torch.zeros_like(done_count)
    final_post_reset_linf = torch.zeros((), device=args.device)
    start = time.perf_counter()
    for _ in range(args.steps):
        observations, _reward, done, extras = env.step(actions)
        done_count.add_(done.sum())
        if args.terminal_profile == "correct_terminal":
            valid = extras["final_obs_valid"]
            final_count.add_(valid.sum())
            missing_final_count.add_((done & ~valid).sum())
            delta = (extras["final_obs"]["state"] - observations["state"]).abs()
            torch.maximum(final_post_reset_linf, (delta * done.unsqueeze(-1)).max(), out=final_post_reset_linf)
    torch.cuda.synchronize()
    duration = time.perf_counter() - start
    result = {
        "schema": "bfm_final_capture_benchmark_v1",
        "terminal_profile": args.terminal_profile,
        "num_envs": args.num_envs,
        "warmup_steps": args.warmup_steps,
        "steps": args.steps,
        "duration_seconds": duration,
        "vector_steps_per_second": args.steps / duration,
        "edges_per_second": args.steps * args.num_envs / duration,
        "done_count": int(done_count.item()),
        "final_count": int(final_count.item()),
        "missing_final_count": int(missing_final_count.item()),
        "final_post_reset_state_linf": float(final_post_reset_linf.item()),
        "cuda_allocated_gib": torch.cuda.memory_allocated() / 2**30,
        "cuda_reserved_gib": torch.cuda.memory_reserved() / 2**30,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
