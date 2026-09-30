"""Mario environment wrapper (frame skip, reward shaping) and a multiprocess vector env."""

from __future__ import annotations

import multiprocessing as mp
from collections import deque
from typing import Any

import numpy as np

from ..actions import JUMP_ACTIONS, JUMP_RELEASE_ACTION
from .backend import make_backend, read_status
from .config import RewardConfig
from .features import NUM_ACTIONS, OBS_DIM, STACK_FRAMES, decode_action, extract_features

# Per-decision penalties are defined for a 4-frame decision and scaled by actual duration,
# so choosing a long hold is not a way to dodge them.
REFERENCE_FRAMES = 4


class MarioEnv:
    """One Mario episode stream: 1-1, variable-length actions, shaped reward, auto-reset."""

    def __init__(
        self,
        env_id: str = "SuperMarioBros-1-1-v0",
        reward: RewardConfig | None = None,
        render: bool = False,
        backend: str = "nes-py",
        levels: tuple[str, ...] = (),
    ) -> None:
        self.backend = make_backend(backend, env_id, render, levels)
        self.cfg = reward or RewardConfig()
        self.ram = self.backend.reset()
        self.frames: deque[np.ndarray] = deque(maxlen=STACK_FRAMES)
        self._reset_state()

    def _reset_state(self) -> None:
        self.best_x = 0
        self.stall = 0
        self.prev_action = 0
        self.x = 0

    def reset(self) -> np.ndarray:
        self.ram = self.backend.reset()
        self._reset_state()
        self.x = read_status(self.ram)[0]
        self.best_x = self.x
        first = extract_features(self.ram, self.x, 0)
        self.frames.extend([first] * STACK_FRAMES)
        return np.concatenate(self.frames)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Returns (obs, reward, done, info). Resets automatically when done."""
        button, hold = decode_action(action)
        # A jump only triggers on a fresh A press, so release A for the first frame
        # when starting a jump from the ground.
        first = button
        if button in JUMP_ACTIONS and self.ram[0x001D] == 0:
            first = JUMP_RELEASE_ACTION[button]

        frames = 0
        died = won = False
        x = self.x
        for i in range(hold):
            self.ram = self.backend.step(first if i == 0 else button)
            frames += 1
            x, died, won = read_status(self.ram)
            if died or won:
                break

        cfg = self.cfg
        scale = frames / REFERENCE_FRAMES
        reward = (x - self.x) / cfg.progress_scale - cfg.time_penalty * scale
        self.x = x
        if x > self.best_x:
            self.best_x = x
            self.stall = 0
        else:
            self.stall += frames
            reward -= cfg.stall_penalty * scale

        stalled = self.stall >= cfg.stall_limit * REFERENCE_FRAMES
        done = won or died or stalled
        if won:
            reward += cfg.flag_bonus
        if died:
            reward += cfg.death_penalty

        # The level must be read before reset() below draws the next one.
        level = getattr(self.backend, "level", "1-1")
        episode = {"x": self.best_x, "won": won, "level": level} if done else {}
        self.prev_action = action
        if done:
            obs = self.reset()
        else:
            self.frames.append(extract_features(self.ram, x, action))
            obs = np.concatenate(self.frames)
        return obs, float(reward), done, episode

    def close(self) -> None:
        self.backend.close()


def _worker(
    remote: Any, env_id: str, reward: RewardConfig, backend: str, levels: tuple[str, ...]
) -> None:
    env = MarioEnv(env_id, reward, backend=backend, levels=levels)
    while True:
        cmd, data = remote.recv()
        if cmd == "reset":
            remote.send(env.reset())
        elif cmd == "step":
            remote.send(env.step(data))
        elif cmd == "close":
            env.close()
            remote.close()
            return


class VecMarioEnv:
    """N MarioEnv workers in separate processes, stepped in lockstep."""

    def __init__(
        self,
        n: int,
        env_id: str = "SuperMarioBros-1-1-v0",
        reward: RewardConfig | None = None,
        backend: str = "nes-py",
        levels: tuple[str, ...] = (),
    ) -> None:
        reward = reward or RewardConfig()
        ctx = mp.get_context("spawn")
        self.n = n
        self.remotes, self.procs = [], []
        for _ in range(n):
            parent, child = ctx.Pipe()
            proc = ctx.Process(
                target=_worker, args=(child, env_id, reward, backend, levels), daemon=True
            )
            proc.start()
            child.close()
            self.remotes.append(parent)
            self.procs.append(proc)

    def reset(self) -> np.ndarray:
        for remote in self.remotes:
            remote.send(("reset", None))
        return np.stack([remote.recv() for remote in self.remotes])

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", int(action)))
        results = [remote.recv() for remote in self.remotes]
        obs, rewards, dones, infos = zip(*results)
        return (
            np.stack(obs),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
            list(infos),
        )

    def close(self) -> None:
        for remote in self.remotes:
            remote.send(("close", None))
        for proc in self.procs:
            proc.join(timeout=5)


__all__ = ["NUM_ACTIONS", "OBS_DIM", "MarioEnv", "RewardConfig", "VecMarioEnv"]
