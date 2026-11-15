"""DDoS-Flow-Klassifikation mit Double DQN (PyTorch, CUDA) und AWGN-Kanal.

Umgebung (``DDoSFlowEnv``, Gymnasium)
-------------------------------------
* Jeder Netzwerkfluss ist ein Feature-Vektor (8 Merkmale, z. B. Paketrate, SYN-Anteil,
  Quell-IP-Entropie ...), klassenbedingt gaußverteilt (Normal vs. DDoS, überlappend).
* Der Agent beobachtet den Vektor über einen AWGN-Kanal:
  Rauschvarianz je Merkmal = Signalleistung / 10^(SNR/10), SNR in {30, 20, 10, 5} dB.
* Aktionen (2): 0 = Normal, 1 = DDoS.
* Reward: TP +1.0 | TN +0.5 | FP -1.0 | FN -2.0 (False Negatives werden am stärksten bestraft).
* Jeder Fluss ist eine eigene Entscheidung, die den nächsten Fluss nicht beeinflusst ->
  ein Fluss = ein Ein-Schritt-Episode (terminated=True, kein Bootstrapping nötig).
  Das DDQN (Online-/Target-Netz, Double-Q-Target, Replay Buffer) bleibt dennoch voll
  implementiert; das Ziel ist r + gamma * (1 - done) * Q_target(s', argmax Q_online(s', .)).

Ausgabe: evaluation_results.csv (Precision/Recall/F1 je Seed und SNR + Mittelwert/Std).
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import torch
from gymnasium import spaces
from torch import nn

FloatArr = npt.NDArray[np.float32]

SNR_LEVELS_DB: tuple[int, ...] = (30, 20, 10, 5)
SEEDS: tuple[int, ...] = (0, 1, 2)
N_FEATURES = 8
N_ACTIONS = 2
ATTACK_PROB = 0.3

R_TP, R_TN, R_FP, R_FN = 1.0, 0.5, -1.0, -2.0

# Klassenbedingte Mittelwerte/Std der Merkmale (bewusst überlappend)
MU_NORMAL = np.array([0.30, 0.35, 0.25, 0.55, 0.50, 0.50, 0.50, 0.45], dtype=np.float32)
MU_ATTACK = np.array([0.55, 0.60, 0.50, 0.35, 0.30, 0.35, 0.35, 0.65], dtype=np.float32)
SIGMA = np.float32(0.20)
# Signalleistung je Merkmal E[x^2] (Mischung beider Klassen) als Referenz für den SNR
_MU_MIX = (1 - ATTACK_PROB) * MU_NORMAL + ATTACK_PROB * MU_ATTACK
SIGNAL_POWER = ((1 - ATTACK_PROB) * MU_NORMAL**2 + ATTACK_PROB * MU_ATTACK**2 + SIGMA**2).astype(np.float32)


def sample_flows(
    rng: np.random.Generator, snr_db: npt.NDArray[np.float64] | float, n: int
) -> tuple[FloatArr, npt.NDArray[np.int64]]:
    """Erzeugt n Flüsse: (verrauschte Features, Labels). snr_db: Skalar oder Array (n,)."""
    labels = (rng.random(n) < ATTACK_PROB).astype(np.int64)
    mu = np.where(labels[:, None] == 1, MU_ATTACK, MU_NORMAL)
    clean = mu + SIGMA * rng.standard_normal((n, N_FEATURES), dtype=np.float32)
    snr_lin = 10.0 ** (np.asarray(snr_db, dtype=np.float64) / 10.0)
    noise_std = np.sqrt(SIGNAL_POWER[None, :] / np.reshape(snr_lin, (-1, 1))).astype(np.float32)
    noisy = clean + noise_std * rng.standard_normal((n, N_FEATURES), dtype=np.float32)
    return noisy.astype(np.float32), labels


class DDoSFlowEnv(gym.Env[FloatArr, int]):
    """Gymnasium-Umgebung: binäre Klassifikation eines Flusses bei gegebenem SNR."""

    metadata = {"render_modes": []}

    def __init__(self, snr_db: float = 30.0) -> None:
        super().__init__()
        self.snr_db = float(snr_db)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(-np.inf, np.inf, (N_FEATURES,), np.float32)
        self._label = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[FloatArr, dict[str, Any]]:
        super().reset(seed=seed)
        obs, labels = sample_flows(self.np_random, self.snr_db, 1)
        self._label = int(labels[0])
        return obs[0], {}

    def step(self, action: int) -> tuple[FloatArr, float, bool, bool, dict[str, Any]]:
        reward = reward_fn(int(action), self._label)
        obs = np.zeros(N_FEATURES, dtype=np.float32)  # Ein-Schritt-Episode, Terminal-Obs unbenutzt
        return obs, reward, True, False, {"label": self._label}


def reward_fn(action: int, label: int) -> float:
    if label == 1:
        return R_TP if action == 1 else R_FN
    return R_TN if action == 0 else R_FP


REWARD_TABLE = np.array([[R_TN, R_FP], [R_FN, R_TP]], dtype=np.float32)  # [label, action]


# ---------------------------------------------------------------------------
# DDQN
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    hidden: int = 128
    lr: float = 1e-4
    buffer_size: int = 50_000
    batch_size: int = 64
    gamma: float = 0.99
    eps_start: float = 1.0
    eps_end: float = 0.02
    eps_decay_frac: float = 0.4
    learn_start: int = 1_000
    target_update: int = 250
    num_envs: int = 32  # parallel erzeugte Flüsse pro Iteration
    iterations: int = 20_000
    eval_flows: int = 20_000


class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Ring-Buffer, vollständig auf dem Zielgerät (kein Host<->GPU-Transfer beim Sampling)."""

    def __init__(self, capacity: int, obs_dim: int, device: torch.device) -> None:
        self.cap, self.size, self.ptr = capacity, 0, 0
        self.obs = torch.zeros((capacity, obs_dim), device=device)
        self.next_obs = torch.zeros((capacity, obs_dim), device=device)
        self.act = torch.zeros(capacity, dtype=torch.int64, device=device)
        self.rew = torch.zeros(capacity, device=device)
        self.done = torch.zeros(capacity, device=device)

    def add_batch(self, o: torch.Tensor, a: torch.Tensor, r: torch.Tensor,
                  no: torch.Tensor, d: torch.Tensor) -> None:
        n = o.shape[0]
        idx = (self.ptr + torch.arange(n, device=o.device)) % self.cap
        self.obs[idx], self.act[idx], self.rew[idx] = o, a, r
        self.next_obs[idx], self.done[idx] = no, d
        self.ptr = (self.ptr + n) % self.cap
        self.size = min(self.size + n, self.cap)

    def sample(self, n: int) -> tuple[torch.Tensor, ...]:
        i = torch.randint(0, self.size, (n,), device=self.obs.device)
        return self.obs[i], self.act[i], self.rew[i], self.next_obs[i], self.done[i]


class DDQNAgent:
    def __init__(self, cfg: Config, device: torch.device) -> None:
        self.cfg, self.device = cfg, device
        self.online = QNet(N_FEATURES, N_ACTIONS, cfg.hidden).to(device)
        self.target = QNet(N_FEATURES, N_ACTIONS, cfg.hidden).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.opt = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.buffer_size, N_FEATURES, device)
        self.learn_steps = 0

    @torch.no_grad()
    def act(self, obs: torch.Tensor, eps: float) -> torch.Tensor:
        greedy = self.online(obs).argmax(dim=1)
        if eps <= 0.0:
            return greedy
        rand = torch.randint(0, N_ACTIONS, greedy.shape, device=self.device)
        return torch.where(torch.rand(greedy.shape, device=self.device) < eps, rand, greedy)

    def learn(self) -> None:
        if self.buffer.size < max(self.cfg.batch_size, self.cfg.learn_start):
            return
        o, a, r, no, d = self.buffer.sample(self.cfg.batch_size)
        with torch.no_grad():
            best = self.online(no).argmax(dim=1, keepdim=True)         # Aktionsauswahl: online
            q_next = self.target(no).gather(1, best).squeeze(1)        # Bewertung: target
            y = r + self.cfg.gamma * (1.0 - d) * q_next
        q = self.online(o).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = nn.functional.smooth_l1_loss(q, y)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()
        self.learn_steps += 1
        if self.learn_steps % self.cfg.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())


def select_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        print("WARNUNG: CUDA nicht verfügbar, falle auf CPU zurück.", flush=True)
        return torch.device("cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(agent: DDQNAgent, cfg: Config, seed: int, log_every: int = 5_000) -> None:
    """Training mit pro Fluss zufällig gemischtem SNR (30/20/10/5 dB)."""
    dev = agent.device
    rng = np.random.default_rng(seed + 12_345)
    reward_t = torch.from_numpy(REWARD_TABLE).to(dev)
    decay = max(1, int(cfg.eps_decay_frac * cfg.iterations))
    snr_choices = np.array(SNR_LEVELS_DB, dtype=np.float64)
    ret_acc = torch.zeros((), device=dev)
    for it in range(cfg.iterations):
        eps = max(cfg.eps_end, cfg.eps_start - (cfg.eps_start - cfg.eps_end) * it / decay)
        feats, labels = sample_flows(rng, rng.choice(snr_choices, cfg.num_envs), cfg.num_envs)
        obs = torch.from_numpy(feats).to(dev)
        lab = torch.from_numpy(labels).to(dev)
        act = agent.act(obs, eps)
        rew = reward_t[lab, act]
        # Terminal-Transition (done=1): Folgezustand wird nicht bootstrapped
        agent.buffer.add_batch(obs, act, rew, obs, torch.ones_like(rew))
        agent.learn()
        ret_acc += rew.mean()
        if (it + 1) % log_every == 0:
            print(f"  [seed {seed}] Iter {it + 1}/{cfg.iterations}  "
                  f"mean reward={ret_acc.item() / log_every:.3f}  eps={eps:.2f}", flush=True)
            ret_acc.zero_()


@torch.no_grad()
def evaluate(agent: DDQNAgent, snr_db: float, n_flows: int, seed: int) -> tuple[float, float, float]:
    """Greedy-Evaluation auf frischen Flüssen; Precision/Recall/F1 für Klasse DDoS."""
    rng = np.random.default_rng(seed * 1_000_003 + int(snr_db))
    feats, labels = sample_flows(rng, snr_db, n_flows)
    pred = agent.act(torch.from_numpy(feats).to(agent.device), 0.0).cpu().numpy()
    tp = int(np.sum((pred == 1) & (labels == 1)))
    fp = int(np.sum((pred == 1) & (labels == 0)))
    fn = int(np.sum((pred == 0) & (labels == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def env_selftest() -> None:
    """Prüft die Gymnasium-Umgebung gegen die API (Checker)."""
    from gymnasium.utils.env_checker import check_env

    check_env(DDoSFlowEnv(snr_db=10.0), skip_render_check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--iterations", type=int, default=Config.iterations)
    parser.add_argument("--eval-flows", type=int, default=Config.eval_flows)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("evaluation_results.csv"))
    args = parser.parse_args()
    cfg = Config(iterations=args.iterations, eval_flows=args.eval_flows)
    device = select_device(args.device)
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    env_selftest()

    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        set_seed(seed)
        t0 = time.time()
        print(f"Training Seed {seed} ...", flush=True)
        agent = DDQNAgent(cfg, device)
        train(agent, cfg, seed)
        print(f"  Trainingszeit: {time.time() - t0:.1f}s", flush=True)
        for snr in SNR_LEVELS_DB:
            p, r, f1 = evaluate(agent, snr, cfg.eval_flows, seed)
            print(f"  eval seed={seed} SNR={snr:>2} dB  P={p:.3f} R={r:.3f} F1={f1:.3f}", flush=True)
            rows.append({"seed": seed, "snr_db": snr, "precision": p, "recall": r, "f1": f1})

    for stat, fn in (("mean", np.mean), ("std", np.std)):
        for snr in SNR_LEVELS_DB:
            sel = [x for x in rows if x["snr_db"] == snr and isinstance(x["seed"], int)]
            rows.append({"seed": stat, "snr_db": snr,
                         **{k: float(fn([x[k] for x in sel])) for k in ("precision", "recall", "f1")}})

    with args.output.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["seed", "snr_db", "precision", "recall", "f1"])
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})
    print(f"Ergebnisse gespeichert: {args.output}")


if __name__ == "__main__":
    main()
