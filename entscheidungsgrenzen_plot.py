"""2D-Schnitt durch den achtdimensionalen Beobachtungsraum: Entscheidungsgrenzen von Bayes-Klassifikator, DDQN, MLP
und Random Forest in der Ebene (y1, y4), übrige sechs Merkmale fest auf der Mitte der Klassenmittel.

Gezeigt je SNR-Stufe (30 und 5 dB): Regionen des Random Forest (achsenparallele Stufen), Bayes-Grenzen bei
p = 1/2, 1/3 und 3/7 (bekanntes SNR, Geraden mit Normalenvektor w), Grenze des DDQN (Q(y,1) = Q(y,0)) und des
MLP (P = 1/2), 1-sigma-Ellipsen der Randverteilungen beider Klassen in (y1, y4).
Modelle: Seed 0 (DDQN aus modelle/, RF und MLP wie eval_baselines.py neu trainiert und gegen die CSV geprüft).
Ausgabe: entscheidungsgrenzen_plot.pdf
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support

import plot_stil
from bayes_referenz import _var, posterior
from eval_baselines import N_TRAIN, make_models
from train_eval import (MU_ATTACK, MU_NORMAL, N_ACTIONS, N_FEATURES, SNR_LEVELS_DB, Config, QNet, sample_flows)

HERE = Path(__file__).parent
SEED = 0
J1, J4 = 0, 3                                   # Merkmale y1 und y4 (nullbasiert), wie Abb. 4.2
C_NORMAL, C_ATTACK = "#3b6ea5", "#c0504d"
C_DDQN, C_MLP = "C0", "C3"
plot_stil.anwenden(**{"axes.labelsize": 10, "legend.fontsize": 7.8})


def gegenpruefen(name: str, modell, cfg: Config) -> None:
    soll = {}
    with open(HERE / "baseline_results.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["seed"] == str(SEED) and r["model"] == name:
                soll[int(r["snr_db"])] = (r["precision"], r["recall"], r["f1"])
    for snr in SNR_LEVELS_DB:
        y, c = sample_flows(np.random.default_rng(SEED * 1_000_003 + snr), snr, cfg.eval_flows)
        p, r, f1, _ = precision_recall_fscore_support(c, modell.predict(y), average="binary", pos_label=1,
                                                      zero_division=0)
        if tuple(f"{v:.4f}" for v in (p, r, f1)) != soll[snr]:
            sys.exit(f"{name} reproduziert baseline_results.csv nicht (SNR {snr}), Abbruch.")


def main() -> None:
    cfg = Config()
    rng = np.random.default_rng(SEED + 12_345)
    snr_tr = rng.choice(np.array(SNR_LEVELS_DB, dtype=np.float64), N_TRAIN)
    x_tr, y_tr = sample_flows(rng, snr_tr, N_TRAIN)
    modelle = make_models(SEED)
    rf = modelle["RandomForest"].fit(x_tr, y_tr)
    mlp = modelle["MLP"].fit(x_tr, y_tr)
    gegenpruefen("RandomForest", rf, cfg)
    gegenpruefen("MLP", mlp, cfg)
    print("RF und MLP (Seed 0): P/R/F1 identisch mit baseline_results.csv")
    net = QNet(N_FEATURES, N_ACTIONS, cfg.hidden)
    net.load_state_dict(torch.load(HERE / "modelle" / f"ddqn_seed{SEED}.pt", map_location="cpu"))
    net.eval()

    mitte = 0.5 * (MU_NORMAL + MU_ATTACK).astype(np.float64)
    g1 = np.linspace(-0.25, 1.05, 400)
    g4 = np.linspace(-0.15, 1.05, 400)
    G1, G4 = np.meshgrid(g1, g4)
    Y = np.tile(mitte, (G1.size, 1))
    Y[:, J1], Y[:, J4] = G1.ravel(), G4.ravel()
    Yf = Y.astype(np.float32)
    rf_reg = rf.predict(Yf).reshape(G1.shape)
    mlp_p = mlp.predict_proba(Yf)[:, 1].reshape(G1.shape)
    with torch.no_grad():
        q = net(torch.from_numpy(Yf)).numpy()
    dq = (q[:, 1] - q[:, 0]).reshape(G1.shape)

    # Bayes-Posterior entlang der DDQN-Grenze im Schnitt (Zeilen mit 0 <= y4 <= 1), für die Einordnung im Text
    for snr in (30, 5):
        post = posterior(Y, snr).reshape(G1.shape)
        werte = []
        for z in range(len(g4)):
            if 0.0 <= g4[z] <= 1.0:
                k = np.where((dq[z, :-1] < 0) & (dq[z, 1:] >= 0))[0]
                if len(k):
                    werte.append(post[z, k[0]])
        print(f"SNR {snr:>2} dB: Bayes-Posterior entlang der DDQN-Grenze im Schnitt {min(werte):.2f} bis {max(werte):.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.55), sharey=True)
    for ax, snr in zip(axes, (30, 5)):
        post = posterior(Y, snr).reshape(G1.shape)
        ax.contourf(G1, G4, rf_reg, levels=[0.5, 1.5], colors=["0.86"], zorder=0)
        ax.contour(G1, G4, rf_reg, levels=[0.5], colors=["0.55"], linewidths=0.6, zorder=1)
        ax.contour(G1, G4, post, levels=[0.5], colors=["0.1"], linewidths=1.1, zorder=3)
        ax.contour(G1, G4, post, levels=[1 / 3], colors=["0.1"], linewidths=1.1, linestyles=[(0, (4, 2))], zorder=3)
        ax.contour(G1, G4, post, levels=[3 / 7], colors=["0.1"], linewidths=1.0, linestyles=[(0, (1, 1.2))],
                   zorder=3)
        ax.contour(G1, G4, dq, levels=[0.0], colors=[C_DDQN], linewidths=1.7, zorder=4)
        ax.contour(G1, G4, mlp_p, levels=[0.5], colors=[C_MLP], linewidths=1.7, zorder=4)
        sd = np.sqrt(_var(snr))
        for mu, col in ((MU_NORMAL, C_NORMAL), (MU_ATTACK, C_ATTACK)):
            ax.add_patch(Ellipse((mu[J1], mu[J4]), 2 * sd[J1], 2 * sd[J4], fill=False, ec=col, lw=1.2, zorder=2))
            ax.plot(mu[J1], mu[J4], marker="x", color=col, ms=6, mew=1.6, zorder=5)
        # Normalenvektor w der Bayes-Grenze (Projektion auf die Ebene), am Grenzpunkt angesetzt
        w = (MU_ATTACK - MU_NORMAL).astype(np.float64) / _var(snr)
        wv = np.array([w[J1], w[J4]]); wv /= np.linalg.norm(wv)
        x0 = np.array([mitte[J1], mitte[J4]])
        ax.annotate("", xy=x0 + 0.24 * wv, xytext=x0, arrowprops=dict(arrowstyle="-|>", color="0.1", lw=1.1),
                    zorder=6)
        ax.text(*(x0 + 0.27 * wv), r"$\mathbf{w}$", fontsize=10, zorder=6)
        ax.set_title(f"SNR = {snr} dB", fontsize=10)
        ax.set_xlim(g1[0], g1[-1]); ax.set_ylim(g4[0], g4[-1])
        ax.set_aspect("equal")
        ax.set_xlabel(r"$y_1$")
        ax.xaxis.set_major_formatter(plot_stil.komma(1)); ax.yaxis.set_major_formatter(plot_stil.komma(1))
    axes[0].set_ylabel(r"$y_4$")
    handles = [
        Patch(facecolor="0.86", edgecolor="0.55", lw=0.6, label="Random Forest: DDoS-Region"),
        Line2D([], [], color=C_DDQN, lw=1.7, label=r"DDQN: $Q(\mathbf{y},1)=Q(\mathbf{y},0)$"),
        Line2D([], [], color=C_MLP, lw=1.7, label=r"MLP: $P=1/2$"),
        Line2D([], [], color="0.1", lw=1.1, label=r"Bayes $p=1/2$"),
        Line2D([], [], color="0.1", lw=1.1, ls=(0, (4, 2)), label=r"Bayes $p=1/3$"),
        Line2D([], [], color="0.1", lw=1.0, ls=(0, (1, 1.2)), label=r"Bayes $p=3/7$"),
        Line2D([], [], color=C_NORMAL, lw=1.2, label=r"Normal: $\boldsymbol{\mu}$, $1\sigma$"),
        Line2D([], [], color=C_ATTACK, lw=1.2, label=r"DDoS: $\boldsymbol{\mu}$, $1\sigma$"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    fig.savefig(HERE / "entscheidungsgrenzen_plot.pdf", format="pdf")
    plt.close(fig)
    print("Gespeichert: entscheidungsgrenzen_plot.pdf")


if __name__ == "__main__":
    main()
