"""Live viewer: replays the latest checkpoint in a pygame window, reloading between episodes."""

from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.distributions import Categorical

from .env import MarioEnv
from .features import decode_action
from .model import ActorCritic

SCALE = 3
FONT_SIZE = 18


def _load(model: ActorCritic, checkpoint: Path, loaded_mtime: float) -> float:
    """Reload weights if the file changed. Tolerates a half-written checkpoint."""
    try:
        mtime = checkpoint.stat().st_mtime
        if mtime != loaded_mtime:
            model.load_state_dict(torch.load(checkpoint))
            return mtime
    except (FileNotFoundError, RuntimeError, EOFError):
        pass
    return loaded_mtime


def watch(checkpoint: Path, fps: int = 60, greedy: bool = False, backend: str = "nes-py") -> None:
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((256 * SCALE, 240 * SCALE))
    pygame.display.set_caption("Mario policy (Q/Esc to quit)")
    font = pygame.font.SysFont("menlo", FONT_SIZE)
    clock = pygame.time.Clock()

    model = ActorCritic().eval()
    env = MarioEnv(backend=backend)
    mtime = 0.0
    while mtime == 0.0:  # wait until training has written its first checkpoint
        mtime = _load(model, checkpoint, mtime)
        if mtime == 0.0:
            print(f"Waiting for {checkpoint} ...", flush=True)
            time.sleep(2)

    episode, wins, running = 0, 0, True
    while running:
        mtime = _load(model, checkpoint, mtime)
        obs, done, info = env.reset(), False, {}
        episode += 1
        while not done and running:
            with torch.no_grad():
                logits = model(torch.as_tensor(obs)[None])[0][0]
            probs = torch.softmax(logits, -1)
            index = int(logits.argmax() if greedy else Categorical(probs=probs).sample())
            button, hold = decode_action(index)
            # Step one decision but present every emulator frame for smooth playback.
            frames = []
            original = env.backend.step

            def capture(b: object, _orig: object = original, _frames: list = frames) -> object:
                result = _orig(b)
                _frames.append(env.backend.screen())
                return result

            env.backend.step = capture
            try:
                obs, _, done, info = env.step(index)
            finally:
                env.backend.step = original

            for frame in frames:
                surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                screen.blit(pygame.transform.scale(surface, screen.get_size()), (0, 0))
                lines = [
                    f"episode {episode}  clears {wins}  x {env.x}",
                    f"action {button.value} x{hold}f  p={float(probs[index]):.2f}",
                    f"ckpt {time.strftime('%H:%M:%S', time.localtime(mtime))}",
                ]
                for row, text in enumerate(lines):
                    screen.blit(
                        font.render(text, True, (255, 255, 255), (0, 0, 0)), (6, 6 + row * 22)
                    )
                pygame.display.flip()
                clock.tick(fps)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE)
                    ):
                        running = False
        if not done:  # window closed mid-episode
            break
        if info["won"]:
            wins += 1
        print(f"episode {episode}: x={info['x']} won={info['won']}", flush=True)
    env.close()
    pygame.quit()
