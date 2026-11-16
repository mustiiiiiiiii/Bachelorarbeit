"""Gelernte Entscheidungsfunktion von DDQN und MLP gegenüber der analytischen Bayes-Posterior (30 und 5 dB).

Für jeden Testfluss (dieselben Testgeneratoren wie in train_eval.evaluate und eval_baselines) wird die
Bayes-Posterior p = P(DDoS | y) bei bekanntem SNR in geschlossener Form berechnet (bayes_referenz.posterior).
Aufgetragen sind
  - DDQN: Q(y,1) - Q(y,0) mit der Sollgeraden 4,5 p - 1,5 (Nullstelle p = 1/3) und
  - MLP:  P_MLP(DDoS | y) mit der Solldiagonalen (Entscheidung bei 1/2),
jeweils als Median und 10-90-%-Band je p-Intervall, gepoolt über die Seeds 0, 1, 2. Darunter die Verteilung von p.
Die Schnittpunkte der Mediane mit der Entscheidungsgrenze sind die modellbasierten impliziten Schwellen.

Voraussetzung: modelle/ddqn_seed{0,1,2}.pt (modelle_nachtrainieren.py). Das MLP wird wie in eval_baselines.py neu
trainiert und gegen baseline_results.csv geprüft.
Ausgabe: entscheidungsfunktion_plot.pdf
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support

from bayes_referenz import huber_differenz, posterior, posterior_mischung
import plot_stil
from eval_baselines import N_TRAIN, make_models
from train_eval import N_ACTIONS, N_FEATURES, SEEDS, SNR_LEVELS_DB, Config, QNet, sample_flows

HERE = Path(__file__).parent
SNRS = (30, 5)
C_DDQN, C_MLP = "C0", "C3"          # wie Abb. 7.2/7.3
KOMMA = plot_stil.komma(2, kuerzen=True)

plot_stil.anwenden(**{"axes.labelsize": 9.5, "legend.fontsize": 8})


def testfluesse(seed: int, snr: int, n: int):
    return sample_flows(np.random.default_rng(seed * 1_000_003 + snr), snr, n)


def ddqn_differenz(seed: int, y: np.ndarray, cfg: Config) -> np.ndarray:
    net = QNet(N_FEATURES, N_ACTIONS, cfg.hidden)
    net.load_state_dict(torch.load(HERE / "modelle" / f"ddqn_seed{seed}.pt", map_location="cpu"))
    net.eval()
    with torch.no_grad():
        q = net(torch.from_numpy(y)).numpy().astype(np.float64)
    return q[:, 1] - q[:, 0]


def mlp_soll() -> dict:
    soll = {}
    with open(HERE / "baseline_results.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["seed"].isdigit() and r["model"] == "MLP":
                soll[(int(r["seed"]), int(r["snr_db"]))] = (r["precision"], r["recall"], r["f1"])
    return soll


def band(p: np.ndarray, v: np.ndarray, edges: np.ndarray, nmin: int = 40):
    c, med, lo, hi = [], [], [], []
    idx = np.digitize(p, edges) - 1
    for k in range(len(edges) - 1):
        sel = v[idx == k]
        if len(sel) >= nmin:
            c.append(0.5 * (edges[k] + edges[k + 1]))
            q10, q50, q90 = np.percentile(sel, [10, 50, 90])
            med.append(q50); lo.append(q10); hi.append(q90)
    return map(np.asarray, (c, med, lo, hi))


def nullstelle(x: np.ndarray, f: np.ndarray) -> float:
    """Erste Nullstelle von f(x) per linearer Interpolation (f steigend)."""
    k = np.where((f[:-1] < 0) & (f[1:] >= 0))[0]
    if not len(k):
        return float("nan")
    k = k[0]
    return float(x[k] - f[k] * (x[k + 1] - x[k]) / (f[k + 1] - f[k]))


def main() -> int:
    cfg = Config()
    n = cfg.eval_flows
    soll = mlp_soll()
    data = {snr: {"p": [], "dq": [], "pm": [], "pmix": []} for snr in SNRS}
    abweichung = False
    for seed in SEEDS:
        rng = np.random.default_rng(seed + 12_345)              # wie eval_baselines.py
        snr_tr = rng.choice(np.array(SNR_LEVELS_DB, dtype=np.float64), N_TRAIN)
        x_tr, y_tr = sample_flows(rng, snr_tr, N_TRAIN)
        mlp = make_models(seed)["MLP"].fit(x_tr, y_tr)
        for snr in SNR_LEVELS_DB:
            y, c = testfluesse(seed, snr, n)
            pr, rc, f1, _ = precision_recall_fscore_support(c, mlp.predict(y), average="binary", pos_label=1,
                                                            zero_division=0)
            ist = tuple(f"{v:.4f}" for v in (pr, rc, f1))
            if ist != soll[(seed, snr)]:
                abweichung = True
                print(f"MLP seed={seed} SNR={snr}: ist {ist} soll {soll[(seed, snr)]}  ABWEICHUNG")
            if snr in SNRS:
                data[snr]["p"].append(posterior(y, snr))
                data[snr]["pmix"].append(posterior_mischung(y))
                data[snr]["dq"].append(ddqn_differenz(seed, y, cfg))
                data[snr]["pm"].append(mlp.predict_proba(y)[:, 1])
    if abweichung:
        print("MLP reproduziert die CSV nicht exakt, Abbruch.")
        return 1
    print("MLP: P/R/F1 aller Seeds und Stufen identisch mit baseline_results.csv")

    edges = np.linspace(0, 1, 41)
    fig, axes = plt.subplots(3, 2, figsize=(6.0, 6.4), sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 1.0, 0.38]})
    for col, snr in enumerate(SNRS):
        p = np.concatenate(data[snr]["p"]); dq = np.concatenate(data[snr]["dq"])
        pm = np.concatenate(data[snr]["pm"])
        # DDQN
        ax = axes[0, col]
        xc, med, lo, hi = band(p, dq, edges)
        ax.fill_between(xc, lo, hi, color=C_DDQN, alpha=0.22, lw=0, label=r"DDQN, 10--90\,\%")
        ax.plot(xc, med, color=C_DDQN, lw=1.6, label="DDQN, Median")
        pp = np.linspace(0, 1, 2)
        ax.plot(pp, 4.5 * pp - 1.5, color="0.1", lw=1.0, ls=(0, (5, 2.5)), label=r"Soll MSE: $4{,}5\,p-1{,}5$")
        # Huber-Minimierer (beta = 1) der Zweipunkt-Rewardverteilungen, geschlossene Form, Sprung bei p = 1/2
        pl, pr_ = np.linspace(0, 0.4999, 200), np.linspace(0.5001, 1, 200)
        ax.plot(pl, huber_differenz(pl), color="0.1", lw=1.1, ls=(0, (1, 1.2)),
                label=r"Soll Huber: Nullstelle $3/7$")
        ax.plot(pr_, huber_differenz(pr_), color="0.1", lw=1.1, ls=(0, (1, 1.2)))
        ax.axhline(0, color="0.45", lw=0.6)
        t_dq = nullstelle(xc, med)
        ax.axvline(1 / 3, color="0.55", lw=0.7)
        ax.plot([t_dq], [0], marker="o", ms=5, mfc="white", mec=C_DDQN, mew=1.3, zorder=5)
        ax.annotate(f"$t={t_dq:.2f}$".replace(".", "{,}"), (t_dq, 0), xytext=(8, -14), textcoords="offset points",
                    fontsize=8, color=C_DDQN)
        ax.set_ylim(-1.75, 3.25)
        ax.set_title(f"SNR = {snr} dB", fontsize=10)
        # MLP
        ax = axes[1, col]
        xc, med, lo, hi = band(p, pm, edges)
        ax.fill_between(xc, lo, hi, color=C_MLP, alpha=0.20, lw=0, label=r"MLP, 10--90\,\%")
        ax.plot(xc, med, color=C_MLP, lw=1.6, label="MLP, Median")
        ax.plot(pp, pp, color="0.1", lw=1.0, ls=(0, (5, 2.5)), label="Soll (kalibriert)")
        ax.axhline(0.5, color="0.45", lw=0.6)
        t_m = nullstelle(xc, med - 0.5)
        ax.axvline(0.5, color="0.55", lw=0.7)
        ax.plot([t_m], [0.5], marker="o", ms=5, mfc="white", mec=C_MLP, mew=1.3, zorder=5)
        ax.annotate(f"$t={t_m:.2f}$".replace(".", "{,}"), (t_m, 0.5), xytext=(8, -14), textcoords="offset points",
                    fontsize=8, color=C_MLP)
        ax.set_ylim(-0.03, 1.03)
        # Verteilung von p
        ax = axes[2, col]
        ax.hist(p, bins=edges, color="0.55", weights=np.full(len(p), 100.0 / len(p)))
        ax.set_ylim(0, None)
        ax.set_xlabel(r"Bayes-Posterior $p=P(\mathrm{DDoS}\mid\mathbf{y})$")
        pmix = np.concatenate(data[snr]["pmix"])
        ref = {"Mischung, Schwelle 1/2": pmix - 0.5, "Mischung, MSE (1/3)": 4.5 * pmix - 1.5,
               "Mischung, Huber (3/7)": huber_differenz(pmix)}
        ref_t = {k: nullstelle(*list(band(p, v, edges))[:2]) for k, v in ref.items()}
        print(f"SNR {snr:>2} dB: Nullstelle DDQN t={t_dq:.3f}, MLP t={t_m:.3f}; "
              f"Anteil Flüsse mit 0,2<p<0,6: {100 * np.mean((p > 0.2) & (p < 0.6)):.1f} %")
        print("           Referenzen ohne SNR-Kenntnis: " + ", ".join(f"{k}: t={v:.3f}" for k, v in ref_t.items()))
    axes[0, 0].set_ylabel(r"$Q(\mathbf{y},1)-Q(\mathbf{y},0)$")
    axes[1, 0].set_ylabel(r"$P_{\mathrm{MLP}}(\mathrm{DDoS}\mid\mathbf{y})$")
    axes[2, 0].set_ylabel(r"Anteil in \%")
    for ax in axes.flat:
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(KOMMA)
        ax.yaxis.set_major_formatter(KOMMA)
    h1, l1 = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[1, 0].get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=4, frameon=True, bbox_to_anchor=(0.5, 0.0), fontsize=7.4)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig(HERE / "entscheidungsfunktion_plot.pdf", format="pdf")
    plt.close(fig)
    print("Gespeichert: entscheidungsfunktion_plot.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
