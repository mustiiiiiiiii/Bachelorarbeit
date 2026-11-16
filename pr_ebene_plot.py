"""Precision-Recall-Ebene je SNR-Stufe: Bayes-Grenze, Bayes-Arbeitspunkte (p>1/2, p>1/3) und gemessene Modelle.

Bayes-Referenz in geschlossener Form: Beide Klassen haben die Kovarianz diag(sigma^2 + sigma_n,j^2), das
Log-Likelihood-Verhältnis ist daher linear in y und klassenbedingt normalverteilt,
    LLR | DDoS ~ N(+D^2/2, D^2),   LLR | Normal ~ N(-D^2/2, D^2),
mit D^2 = sum_j dmu_j^2 / (sigma^2 + sigma_n,j^2). Die Posterior-Schwelle p > t entspricht
    LLR > tau(t) = ln(t/(1-t)) - ln(pi/(1-pi)).
Daraus folgen Recall = Q((tau - D^2/2)/D), FPR = Q((tau + D^2/2)/D) und
Precision = pi*R / (pi*R + (1-pi)*FPR). Variiert man t, entsteht die Bayes-PR-Kurve.

Eingaben: evaluation_results.csv (DDQN), baseline_results.csv (RF/SVM/MLP), Generatorparameter aus train_eval.py.
Ausgabe: pr_ebene_plot.pdf
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np

from bayes_referenz import PI, bayes_point, mahalanobis_sq
from train_eval import SIGMA, SNR_LEVELS_DB

HERE = Path(__file__).parent

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.labelsize": 11, "legend.fontsize": 9,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


# ---------------------------------------------------------------- Messwerte
def load(path, model=None):
    """-> {stat: {snr: {metric: wert}}}; stat in {'mean','std'} (wie generate_plot.py)."""
    out = {"mean": {}, "std": {}}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["seed"] in out and (model is None or r.get("model") == model):
                out[r["seed"]][int(r["snr_db"])] = {k: float(r[k]) for k in ("precision", "recall", "f1")}
    return out


MODELS = [("DDQN", load(HERE / "evaluation_results.csv"))] + [
    (lbl, load(HERE / "baseline_results.csv", key))
    for lbl, key in (("Random Forest", "RandomForest"), ("SVM (RBF)", "SVM_RBF"), ("MLP", "MLP"))]
STYLE = {"DDQN": ("o", "C0"), "Random Forest": ("s", "C1"), "SVM (RBF)": ("D", "C2"), "MLP": ("^", "C3")}

# ---------------------------------------------------------------- Konsolenprotokoll (Datenmapping)
print(f"pi = {PI}, sigma = {float(SIGMA)}")
print(f"{'SNR':>4} {'D^2':>7} | {'Bayes p>1/2 (P/R/F1)':>24} | {'Bayes p>1/3 (P/R/F1)':>24}")
for snr in SNR_LEVELS_DB:
    p2, r2, f2 = bayes_point(snr, 0.5)
    p3, r3, f3 = bayes_point(snr, 1 / 3)
    print(f"{snr:>4} {mahalanobis_sq(snr):7.3f} | {p2:.4f} {r2:.4f} {f2:.4f} | {p3:.4f} {r3:.4f} {f3:.4f}")
print(f"{'SNR':>4} {'Modell':<14} {'P (mean±std)':>16} {'R (mean±std)':>16}")
for snr in SNR_LEVELS_DB:
    for lbl, d in MODELS:
        m, s = d["mean"][snr], d["std"][snr]
        print(f"{snr:>4} {lbl:<14} {m['precision']:.4f}±{s['precision']:.4f} {m['recall']:.4f}±{s['recall']:.4f}")

# ---------------------------------------------------------------- Abbildung
t_grid = np.linspace(0.01, 0.99, 981)
fig, axes = plt.subplots(2, 2, figsize=(6.0, 6.3))
for ax, snr in zip(axes.flat, SNR_LEVELS_DB):
    pts = [(d["mean"][snr]["recall"], d["mean"][snr]["precision"]) for _, d in MODELS]
    pb2, rb2, _ = bayes_point(snr, 0.5)
    pb3, rb3, _ = bayes_point(snr, 1 / 3)
    xs = [p[0] for p in pts] + [rb2, rb3]
    ys = [p[1] for p in pts] + [pb2, pb3]
    pad = 0.035
    xlim = (min(xs) - pad, max(xs) + pad)
    ylim = (min(ys) - pad, max(ys) + pad)

    # Iso-F1-Linien als Orientierung
    rr = np.linspace(*xlim, 200)
    for f in np.arange(0.60, 0.96, 0.02):
        with np.errstate(divide="ignore", invalid="ignore"):
            pp = f * rr / (2 * rr - f)
        ok = (2 * rr > f) & (pp > 0)
        ax.plot(rr[ok], pp[ok], color="0.72", lw=0.6, ls=(0, (4, 3)), zorder=0,
                label="konstanter F1 (Abstand 0,02)" if abs(f - 0.60) < 1e-9 else None)

    pc, rc, _ = bayes_point(snr, t_grid)
    ax.plot(rc, pc, color="0.25", lw=1.3, zorder=1, label="Bayes-Grenze (Schwelle variiert)")
    ax.plot(rb2, pb2, ls="none", marker="o", ms=8, mfc="white", mec="0.15", mew=1.3, zorder=2,
            label="Bayes, $p>1/2$")
    ax.plot(rb3, pb3, ls="none", marker="o", ms=8, mfc="0.15", mec="0.15", zorder=2, label="Bayes, $p>1/3$")
    ax.annotate("", xy=(rb3, pb3), xytext=(rb2, pb2), zorder=2,
                arrowprops=dict(arrowstyle="-|>", color="0.15", lw=1.0, shrinkA=6, shrinkB=6,
                                connectionstyle="arc3,rad=0.25"))

    for lbl, d in MODELS:
        m, s = d["mean"][snr], d["std"][snr]
        mk, col = STYLE[lbl]
        ax.errorbar(m["recall"], m["precision"], xerr=s["recall"], yerr=s["precision"], ls="none", marker=mk,
                    ms=5.5, color=col, mec="black", mew=0.5, capsize=2, lw=0.9, zorder=3, label=lbl)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(f"SNR = {snr} dB", fontsize=10)
    ax.tick_params(labelsize=8.5)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MaxNLocator(5))
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ",")))

for ax in axes[1]:
    ax.set_xlabel("Recall")
for ax in axes[:, 0]:
    ax.set_ylabel("Precision")

handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.14, 1, 1))
fig.savefig(HERE / "pr_ebene_plot.pdf", format="pdf")
plt.close(fig)
print("Gespeichert: pr_ebene_plot.pdf")
