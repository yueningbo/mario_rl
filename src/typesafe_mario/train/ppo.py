"""PPO training loop and evaluation for the Mario MLP policy."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from .env import MarioEnv, RewardConfig, VecMarioEnv
from .model import ActorCritic


def train(
    *,
    total_steps: int,
    out_dir: Path,
    reward: RewardConfig | None = None,
    n_envs: int = 8,
    rollout_len: int = 128,
    epochs: int = 4,
    minibatches: int = 4,
    lr: float = 2.5e-4,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    seed: int = 0,
    backend: str = "nes-py",
    levels: tuple[str, ...] = (),
) -> Path:
    torch.manual_seed(seed)
    np.random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = ActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    envs = VecMarioEnv(n_envs, reward=reward, backend=backend, levels=levels)

    obs = torch.as_tensor(envs.reset())
    obs_dim = obs.shape[1]
    buf_obs = torch.zeros(rollout_len, n_envs, obs_dim)
    buf_act = torch.zeros(rollout_len, n_envs, dtype=torch.long)
    buf_logp = torch.zeros(rollout_len, n_envs)
    buf_rew = torch.zeros(rollout_len, n_envs)
    buf_done = torch.zeros(rollout_len, n_envs)
    buf_val = torch.zeros(rollout_len, n_envs)

    n_updates = total_steps // (rollout_len * n_envs)
    recent_x: list[int] = []
    recent_won: list[bool] = []
    recent_level: list[str] = []
    best_x = 0
    steps = 0
    started = time.time()
    checkpoint = out_dir / "policy.pt"

    try:
        for update in range(1, n_updates + 1):
            for group in optimizer.param_groups:  # linear learning-rate decay
                group["lr"] = lr * (1.0 - (update - 1) / n_updates)

            for t in range(rollout_len):
                with torch.no_grad():
                    logits, value = model(obs)
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                buf_obs[t], buf_act[t] = obs, action
                buf_logp[t], buf_val[t] = dist.log_prob(action), value
                next_obs, reward, done, infos = envs.step(action.numpy())
                buf_rew[t], buf_done[t] = torch.as_tensor(reward), torch.as_tensor(done)
                obs = torch.as_tensor(next_obs)
                for info in infos:
                    if info:
                        recent_x.append(info["x"])
                        recent_won.append(info["won"])
                        recent_level.append(info["level"])
                        best_x = max(best_x, info["x"])
                steps += n_envs

            with torch.no_grad():
                next_value = model(obs)[1]
            advantages = torch.zeros_like(buf_rew)
            last = torch.zeros(n_envs)
            for t in reversed(range(rollout_len)):
                following = next_value if t == rollout_len - 1 else buf_val[t + 1]
                mask = 1.0 - buf_done[t]
                delta = buf_rew[t] + gamma * following * mask - buf_val[t]
                last = delta + gamma * lam * mask * last
                advantages[t] = last
            returns = advantages + buf_val

            flat = rollout_len * n_envs
            b_obs, b_act = buf_obs.reshape(flat, -1), buf_act.reshape(flat)
            b_logp, b_adv, b_ret = (
                buf_logp.reshape(flat),
                advantages.reshape(flat),
                returns.reshape(flat),
            )
            for _ in range(epochs):
                order = torch.randperm(flat)
                for chunk in order.chunk(minibatches):
                    logits, value = model(b_obs[chunk])
                    dist = Categorical(logits=logits)
                    ratio = (dist.log_prob(b_act[chunk]) - b_logp[chunk]).exp()
                    adv = b_adv[chunk]
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                    policy_loss = -torch.min(
                        ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv
                    ).mean()
                    value_loss = 0.5 * (value - b_ret[chunk]).pow(2).mean()
                    loss = policy_loss + vf_coef * value_loss - ent_coef * dist.entropy().mean()
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()

            if update % 10 == 0 or update == n_updates:
                torch.save(model.state_dict(), checkpoint)
                window = recent_x[-40:]
                wins = sum(recent_won[-40:])
                per_level = ""
                if len(levels) > 1:  # mean x / clears per level over the last 200 episodes
                    recent = list(zip(recent_level[-200:], recent_x[-200:], recent_won[-200:]))
                    parts = []
                    for name in levels:
                        rows = [(x, won) for lvl, x, won in recent if lvl == name]
                        if rows:
                            parts.append(
                                f"{name}:{np.mean([x for x, _ in rows]):.0f}"
                                f"/{sum(won for _, won in rows)}of{len(rows)}"
                            )
                    per_level = " | " + " ".join(parts)
                print(
                    f"update {update}/{n_updates} steps {steps} "
                    f"{steps / (time.time() - started):.0f} steps/s | "
                    f"mean_x {np.mean(window) if window else 0:.0f} best_x {best_x} "
                    f"wins(last40) {wins}{per_level}",
                    flush=True,
                )
    finally:
        envs.close()
    torch.save(model.state_dict(), checkpoint)
    return checkpoint


def evaluate(
    checkpoint: Path,
    episodes: int = 5,
    render: bool = False,
    backend: str = "nes-py",
    levels: tuple[str, ...] = (),
) -> None:
    """Play with the greedy policy and report progress for each episode."""
    model = ActorCritic()
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    env = MarioEnv(render=render, backend=backend, levels=levels)
    for episode in range(episodes):
        obs, done = env.reset(), False
        while not done:
            with torch.no_grad():
                action = int(model(torch.as_tensor(obs)[None])[0].argmax())
            obs, _, done, info = env.step(action)
            if render:
                env.backend.render()
        print(f"episode {episode}: level={info['level']} x={info['x']} won={info['won']}")
    env.close()
