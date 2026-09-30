from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .policy import HeuristicPolicy, TypeSafePolicy
from .runner import run_episode
from .state import MarioStateParser
from .train.config import BACKEND_HELP, BACKENDS, RewardConfig

LEVELS_HELP = (
    "Comma-separated levels drawn at random each episode, e.g. 1-1,2-1,3-1 "
    "(retro backend only; it ships save-states for 1-1, 1-4, 2-1 and 3-1 to 8-1)"
)


def _levels(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def _demo_ram() -> bytearray:
    ram = bytearray(0x0800)
    # Mario position and one goomba ahead.
    ram[0x006D] = 0
    ram[0x0086] = 172
    ram[0x000F] = 1
    ram[0x0016] = 0x06
    ram[0x006E] = 0
    ram[0x0087] = 214
    ram[0x00CF] = 79
    # Fill the tile-map row immediately below Mario with solid ground.
    ground_row = (79 + 16 - 32) // 16
    for column in range(16):
        ram[0x0500 + ground_row * 16 + column] = 1
    return ram


def state_demo() -> int:
    info = {
        "world": 1,
        "stage": 1,
        "area": 1,
        "x_pos": 172,
        "y_pos": 79,
        "y_pixel": 79,
        "left_x_pos": 60,
        "progress": 172,
        "progress_max": 172,
        "status": "small",
        "player_state": 8,
        "life": 2,
        "coins": 0,
        "score": 200,
        "time": 387,
        "death": False,
        "clear": False,
    }
    snapshot = MarioStateParser().parse(info, _demo_ram(), previous_action="right")
    print(json.dumps(snapshot.to_state(), indent=2))
    print("\n--- TEXT VIEW ---\n")
    print(snapshot.to_text())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="typesafe-mario")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("state-demo", help="Show structured and text state without API calls")

    play = subparsers.add_parser("play", help="Run a Mario episode")
    play.add_argument("--env", default="SuperMarioBros-1-1-v0")
    play.add_argument(
        "--frames-per-decision",
        type=int,
        default=8,
        help="Minimum macro duration in emulator frames",
    )
    play.add_argument("--max-decisions", type=int, default=2000)
    play.add_argument("--seed", type=int, default=123)
    play.add_argument("--policy", choices=("typesafe", "heuristic"), default="typesafe")
    play.add_argument(
        "--display",
        choices=("dashboard", "game", "none"),
        default="dashboard",
        help="Combined telemetry dashboard, plain game window, or headless mode",
    )
    play.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    play.add_argument(
        "--screenshot",
        type=Path,
        help="Save the first populated dashboard frame as a PNG",
    )
    train = subparsers.add_parser("train", help="Train the MLP policy with PPO")
    train.add_argument("--steps", type=int, default=2_000_000, help="Total decisions to train")
    train.add_argument("--envs", type=int, default=8)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--backend", choices=BACKENDS, default="nes-py", help=BACKEND_HELP)
    train.add_argument("--levels", type=_levels, default=(), help=LEVELS_HELP)
    train.add_argument("--out", type=Path, default=Path("artifacts/train"))
    defaults = RewardConfig()
    train.add_argument(
        "--progress-scale",
        type=float,
        default=defaults.progress_scale,
        help="Reward = pixels moved right / this value",
    )
    train.add_argument(
        "--time-penalty",
        type=float,
        default=defaults.time_penalty,
        help="Penalty charged per 4 emulator frames",
    )
    train.add_argument(
        "--stall-penalty",
        type=float,
        default=defaults.stall_penalty,
        help="Penalty per 4 frames without new progress",
    )
    train.add_argument("--death-penalty", type=float, default=defaults.death_penalty)
    train.add_argument("--flag-bonus", type=float, default=defaults.flag_bonus)
    train.add_argument(
        "--stall-limit",
        type=int,
        default=defaults.stall_limit,
        help="4-frame steps without new progress before the episode is cut",
    )

    evaluate = subparsers.add_parser("eval", help="Run a trained policy without the API")
    evaluate.add_argument("--checkpoint", type=Path, default=Path("artifacts/train/policy.pt"))
    evaluate.add_argument("--episodes", type=int, default=5)
    evaluate.add_argument("--render", action="store_true", help="Open a live game window")
    evaluate.add_argument("--backend", choices=BACKENDS, default="nes-py", help=BACKEND_HELP)
    evaluate.add_argument("--levels", type=_levels, default=(), help=LEVELS_HELP)

    watch = subparsers.add_parser("watch", help="Live window replaying the latest checkpoint")
    watch.add_argument("--checkpoint", type=Path, default=Path("artifacts/train/policy.pt"))
    watch.add_argument("--greedy", action="store_true", help="Pick the top action, not sample")
    watch.add_argument("--backend", choices=BACKENDS, default="nes-py", help=BACKEND_HELP)

    plot = subparsers.add_parser("plot", help="Serve training curves in the browser")
    plot.add_argument("--root", type=Path, default=Path("artifacts"))
    plot.add_argument("--port", type=int, default=8765)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        from .train.ppo import train

        checkpoint = train(
            total_steps=args.steps,
            out_dir=args.out,
            reward=RewardConfig(
                progress_scale=args.progress_scale,
                time_penalty=args.time_penalty,
                stall_penalty=args.stall_penalty,
                death_penalty=args.death_penalty,
                flag_bonus=args.flag_bonus,
                stall_limit=args.stall_limit,
            ),
            n_envs=args.envs,
            seed=args.seed,
            backend=args.backend,
            levels=args.levels,
        )
        print(f"Checkpoint: {checkpoint.resolve()}")
        return 0
    if args.command == "watch":
        from .train.watch import watch

        watch(args.checkpoint, greedy=args.greedy, backend=args.backend)
        return 0
    if args.command == "plot":
        from .train.plot import serve

        serve(args.root, args.port)
        return 0
    if args.command == "eval":
        from .train.ppo import evaluate

        evaluate(args.checkpoint, args.episodes, args.render, args.backend, args.levels)
        return 0
    if args.command == "state-demo":
        return state_demo()
    if args.command == "play":
        policy = TypeSafePolicy() if args.policy == "typesafe" else HeuristicPolicy()
        log_path = run_episode(
            env_id=args.env,
            policy=policy,
            frames_per_decision=args.frames_per_decision,
            max_decisions=args.max_decisions,
            seed=args.seed,
            artifacts_dir=args.artifacts_dir,
            display=args.display,
            screenshot_path=args.screenshot,
        )
        print(f"Run log: {log_path.resolve()}")
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
