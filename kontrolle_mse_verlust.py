"""Kontrolllauf zur Ursache der impliziten Schwelle: DDQN Seed 0 identisch zu train_eval.py, aber mit
mittlerem quadratischem Fehler (MSE) statt Huber-Verlust.

Hintergrund: Mit Huber-Verlust (beta = 1) liegt die Nullstelle von Q(y,1) - Q(y,0) über der Bayes-Posterior
analytisch bei 3/7 (bayes_referenz.huber_differenz), bei Regression auf den Erwartungswert bei 1/3.
Ausgabe: Konsole (P/R/F1 und Nulldurchgang je SNR-Stufe). Ändert keine Ergebnisdateien.
"""

import numpy as np
import torch
from torch import nn

from bayes_referenz import posterior
from train_eval import SNR_LEVELS_DB, Config, DDQNAgent, evaluate, sample_flows, select_device, set_seed, train


class MSEAgent(DDQNAgent):
    """Wie DDQNAgent, einziger Unterschied: mse_loss statt smooth_l1_loss."""

    def learn(self) -> None:
        if self.buffer.size < max(self.cfg.batch_size, self.cfg.learn_start):
            return
        o, a, r, no, d = self.buffer.sample(self.cfg.batch_size)
        with torch.no_grad():
            best = self.online(no).argmax(dim=1, keepdim=True)
            q_next = self.target(no).gather(1, best).squeeze(1)
            y = r + self.cfg.gamma * (1.0 - d) * q_next
        q = self.online(o).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = nn.functional.mse_loss(q, y)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()
        self.learn_steps += 1
        if self.learn_steps % self.cfg.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())


def nulldurchgang(p: np.ndarray, f: np.ndarray, edges=np.linspace(0, 1, 41)) -> float:
    """Nulldurchgang des Medians von f über p (40 Intervalle, wie entscheidungsfunktion_plot.py)."""
    idx = np.digitize(p, edges) - 1
    xc, md = [], []
    for k in range(len(edges) - 1):
        s = f[idx == k]
        if len(s) >= 40:
            xc.append(0.5 * (edges[k] + edges[k + 1]))
            md.append(np.median(s))
    xc, md = np.array(xc), np.array(md)
    k = np.where((md[:-1] < 0) & (md[1:] >= 0))[0][0]
    return float(xc[k] - md[k] * (xc[k + 1] - xc[k]) / (md[k + 1] - md[k]))


def main() -> None:
    cfg = Config()
    dev = select_device("cuda")
    seed = 0
    set_seed(seed)
    agent = MSEAgent(cfg, dev)
    train(agent, cfg, seed)
    for snr in SNR_LEVELS_DB:
        p_, r_, f_ = evaluate(agent, snr, cfg.eval_flows, seed)
        y, _ = sample_flows(np.random.default_rng(seed * 1_000_003 + snr), snr, cfg.eval_flows)
        with torch.no_grad():
            q = agent.online(torch.from_numpy(y).to(dev)).cpu().numpy().astype(np.float64)
        t = nulldurchgang(posterior(y, snr), q[:, 1] - q[:, 0])
        print(f"MSE-Kontrolle Seed {seed}, SNR {snr:>2} dB: P/R/F1 = {p_:.4f}/{r_:.4f}/{f_:.4f}, "
              f"Nulldurchgang t = {t:.3f}", flush=True)


if __name__ == "__main__":
    main()
