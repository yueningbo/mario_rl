"""Training configuration that must stay import-light (no numpy/torch)."""

from __future__ import annotations

from dataclasses import dataclass

BACKENDS = ("nes-py", "retro")
BACKEND_HELP = "Emulator: nes-py (default, has --render window) or retro (about 3x faster)"


@dataclass(frozen=True)
class RewardConfig:
    """Every number that shapes what the agent is trying to achieve."""

    progress_scale: float = 8.0  # reward = pixels moved right / progress_scale
    time_penalty: float = 0.02  # charged every decision, so faster clears score higher
    stall_penalty: float = 0.05  # additionally charged when there is no new progress
    death_penalty: float = -15.0
    flag_bonus: float = 50.0
    stall_limit: int = 120  # decisions without new best x before the episode is cut
