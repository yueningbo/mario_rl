"""Emulator backends: nes-py (default) or stable-retro (about 3x faster).

Both hand back the 2 KiB NES RAM after every frame, and ``read_status`` derives position,
death and level-clear from that RAM with the same formulas gym-super-mario-bros uses. So the
observation, reward and termination logic in ``env.py`` is identical for either emulator.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from ..actions import ACTION_TO_INDEX, Action
from .config import BACKENDS

RETRO_GAME = "SuperMarioBros-Nes-v0"
NES_RAM_SIZE = 2048
_STAGE_OVER_ENEMIES = (0x2D, 0x31)  # Bowser or flag pole "enemy" types
_ENEMY_TYPE_ADDRESSES = range(0x16, 0x1B)

# Retro button names held for each Action (same mapping as gym's SIMPLE_MOVEMENT).
_RETRO_BUTTONS: dict[Action, tuple[str, ...]] = {
    Action.NOOP: (),
    Action.RIGHT: ("RIGHT",),
    Action.RIGHT_JUMP: ("RIGHT", "A"),
    Action.RIGHT_RUN: ("RIGHT", "B"),
    Action.RIGHT_RUN_JUMP: ("RIGHT", "A", "B"),
    Action.JUMP: ("A",),
    Action.LEFT: ("LEFT",),
}


def read_status(ram: np.ndarray) -> tuple[int, bool, bool]:
    """Return (x position, mario died or is dying, level cleared) from NES RAM."""
    x = int(ram[0x6D]) * 256 + int(ram[0x86])
    player_state = int(ram[0x0E])
    died = player_state in (0x06, 0x0B) or int(ram[0xB5]) > 1
    world_over = int(ram[0x0770]) == 2
    stage_over = int(ram[0x1D]) == 3 and any(
        int(ram[address]) in _STAGE_OVER_ENEMIES for address in _ENEMY_TYPE_ADDRESSES
    )
    return x, died, world_over or stage_over


class NesPyBackend:
    name = "nes-py"

    def __init__(self, env_id: str = "SuperMarioBros-1-1-v0", render: bool = False) -> None:
        import gym_super_mario_bros  # noqa: F401
        import gymnasium as gym
        from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
        from nes_py.wrappers import JoypadSpace

        render_mode = "human" if render else "rgb_array"
        self.env = JoypadSpace(gym.make(env_id, render_mode=render_mode), SIMPLE_MOVEMENT)
        self._ram = self.env.unwrapped.ram  # live view, updated by every step

    def reset(self) -> np.ndarray:
        self.env.reset()
        return self._ram

    def step(self, button: Action) -> np.ndarray:
        self.env.step(ACTION_TO_INDEX[button])
        return self._ram

    def screen(self) -> np.ndarray:
        return np.array(self.env.unwrapped.screen)

    def render(self) -> None:
        self.env.render()

    def close(self) -> None:
        self.env.close()


def ensure_retro_rom() -> None:
    """Make the Super Mario ROM that gym-super-mario-bros ships visible to stable-retro.

    stable-retro's own import tool rejects this dump (its SHA-1 differs from the expected
    one), but RAM trajectories match nes-py frame for frame, so we copy it in directly.
    The copy is atomic because several worker processes may call this at once.
    """
    import gym_super_mario_bros
    import stable_retro

    target = Path(stable_retro.__file__).parent / "data" / "stable" / RETRO_GAME / "rom.nes"
    if target.exists():
        return
    source = Path(gym_super_mario_bros.__file__).parent / "_roms" / "super-mario-bros.nes"
    temp = target.with_suffix(f".{os.getpid()}.tmp")
    shutil.copyfile(source, temp)
    os.replace(temp, target)


class RetroBackend:
    name = "retro"

    def __init__(
        self, env_id: str = "", render: bool = False, levels: tuple[str, ...] = ()
    ) -> None:
        if render:
            raise ValueError("The retro backend has no live window; use --backend nes-py.")
        try:
            import stable_retro as retro
        except ImportError as exc:
            raise RuntimeError(
                "stable-retro is not installed. Its pyglet pin conflicts with nes-py, so use: "
                "uv pip install --no-deps stable-retro farama-notifications"
            ) from exc
        ensure_retro_rom()
        self.env = retro.make(RETRO_GAME, render_mode=None)  # starts at World 1-1
        # Each level is a save-state shipped with stable-retro; one is drawn per reset.
        self._states: dict[str, bytes] = {}
        for level in levels:
            self.env.unwrapped.load_state(f"Level{level}")  # raises if no such save-state
            self._states[level] = self.env.unwrapped.initial_state
        self._rng = np.random.default_rng()
        self.level = levels[0] if levels else "1-1"
        names = self.env.unwrapped.buttons
        self._pads: dict[Action, np.ndarray] = {}
        for action, held in _RETRO_BUTTONS.items():
            pad = np.zeros(len(names), dtype=np.int8)
            for button in held:
                pad[names.index(button)] = 1
            self._pads[action] = pad

    def _ram(self) -> np.ndarray:
        return self.env.get_ram()[:NES_RAM_SIZE]  # get_ram returns a fresh copy each call

    def reset(self) -> np.ndarray:
        if self._states:
            self.level = str(self._rng.choice(list(self._states)))
            self.env.unwrapped.initial_state = self._states[self.level]
        self.env.reset()
        return self._ram()

    def step(self, button: Action) -> np.ndarray:
        self.env.step(self._pads[button])
        return self._ram()

    def screen(self) -> np.ndarray:
        return np.array(self.env.get_screen())

    def render(self) -> None:
        pass

    def close(self) -> None:
        self.env.close()


def make_backend(
    name: str,
    env_id: str = "SuperMarioBros-1-1-v0",
    render: bool = False,
    levels: tuple[str, ...] = (),
) -> Any:
    if name == "nes-py":
        if levels:
            raise ValueError("--levels needs --backend retro (nes-py trains one level per env).")
        return NesPyBackend(env_id, render)
    if name == "retro":
        return RetroBackend(env_id, render, levels)
    raise ValueError(f"Unknown backend {name!r}; choose from {BACKENDS}")
