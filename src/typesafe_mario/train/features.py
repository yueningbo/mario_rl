"""Vectorised RAM -> feature extraction for the trained policy.

Uses the same SMB RAM addresses and grid geometry as ``state.MarioStateParser`` but
returns a flat float32 vector directly, which is much cheaper than building strings.
"""

from __future__ import annotations

import numpy as np

from ..actions import Action

GRID_W, GRID_H = 11, 9
MAX_ENEMIES = 5
# Each policy action = a button combo (Action) held for one of HOLD_FRAMES emulator frames.
# Measured apex height for a jump held 2/4/8/16 frames: 30/36/46/61 pixels.
HOLD_FRAMES = (2, 4, 8, 16)
NUM_BUTTONS = len(Action)
NUM_ACTIONS = NUM_BUTTONS * len(HOLD_FRAMES)
# grid + enemies(active, dx, dy) + mario(5) + previous (button, duration) one-hot
OBS_DIM = GRID_W * GRID_H + MAX_ENEMIES * 3 + 5 + NUM_ACTIONS
# The policy sees the last STACK_FRAMES decisions so it can infer motion trends.
STACK_FRAMES = 4
STACKED_DIM = OBS_DIM * STACK_FRAMES

_COLS = np.arange(-2, 9) * 16
_ROWS = np.arange(-4, 5) * 16


def decode_action(index: int) -> tuple[Action, int]:
    """Policy action index -> (button combo, frames to hold it)."""
    button, hold = divmod(index, len(HOLD_FRAMES))
    return list(Action)[button], HOLD_FRAMES[hold]


def _signed(value: int) -> int:
    return value - 256 if value > 127 else value


def extract_features(ram: np.ndarray, x_pos: int, prev_action: int) -> np.ndarray:
    """Build the observation vector from the NES RAM.

    ``ram`` is the 2 KiB RAM array, ``x_pos`` Mario's world x, ``prev_action`` the
    index of the last executed action.
    """
    screen_y = int(ram[0x03B8])

    sample_x = x_pos + _COLS[None, :]
    sample_y = screen_y + _ROWS[:, None]
    page = (sample_x // 256) % 2
    sub_x = (sample_x % 256) // 16
    sub_y = (sample_y - 32) // 16
    valid = (sub_y >= 0) & (sub_y < 13)
    address = 0x0500 + page * 13 * 16 + np.clip(sub_y, 0, 12) * 16 + sub_x
    grid = ((ram[address] != 0) & valid).astype(np.float32)

    enemies = np.zeros((MAX_ENEMIES, 3), dtype=np.float32)
    for slot in range(MAX_ENEMIES):
        if ram[0x000F + slot] == 0:
            continue
        enemy_x = int(ram[0x006E + slot]) * 256 + int(ram[0x0087 + slot])
        dx = enemy_x - x_pos
        if -192 <= dx <= 320:
            enemies[slot] = (1.0, dx / 128.0, (int(ram[0x00CF + slot]) - screen_y) / 64.0)

    mario = np.array(
        [
            _signed(int(ram[0x0057])) / 40.0,  # horizontal speed
            _signed(int(ram[0x009F])) / 8.0,  # vertical velocity
            screen_y / 240.0,
            1.0 if ram[0x001D] == 0 else 0.0,  # on ground
            1.0 if ram[0x0756] > 0 else 0.0,  # powered up
        ],
        dtype=np.float32,
    )
    previous = np.zeros(NUM_ACTIONS, dtype=np.float32)
    previous[prev_action] = 1.0
    return np.concatenate([grid.ravel(), enemies.ravel(), mario, previous])
