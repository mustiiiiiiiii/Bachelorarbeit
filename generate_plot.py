"""Erzeugt die Abbildungen aus evaluation_results.csv (DDQN) und baseline_results.csv (RF/SVM/MLP)."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Deutsches Dezimalkomma auf den Achsen
KOMMA = FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ","))

HERE = Path(__file__).parent
SNRS = [30, 20, 10, 5]

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.labelsize": 11, "legend.fontsize": 9,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def load(path, model=None):
    """-> {stat: {snr: {metric: wert}}}; stat in {'mean','std'}."""
    out = {"mean": {}, "std": {}}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["seed"] in out and (model is None or r.get("model") == model):
                out[r["seed"]][int(r["snr_db"])] = {k: float(r[k]) for k in ("precision", "recall", "f1")}
    return out


def series(d, stat, metric):
    return [d[stat][s][metric] for s in SNRS]


ddqn = load(HERE / "evaluation_results.csv")

# --- Abb. 1: DDQN über SNR ---
fig, ax = plt.subplots(figsize=(6.0, 3.8))
for metric, label, mk, ls in (("f1", "F1-Score", "o", "-"), ("precision", "Precision", "s", "--"),
                              ("recall", "Recall", "^", "-.")):
    ax.errorbar(SNRS, series(ddqn, "mean", metric), yerr=series(ddqn, "std", metric), marker=mk, ls=ls,
                lw=1.6, ms=5, capsize=3, label=label)
ax.set_xlabel("SNR in dB")
ax.set_ylabel("Metrikwert")
ax.yaxis.set_major_formatter(KOMMA)
ax.set_xticks(SNRS)
ax.invert_xaxis()
ax.grid(True, ls=":", alpha=0.6)
ax.legend(loc="lower left", frameon=True)
fig.tight_layout()
fig.savefig(HERE / "snr_robustheit_plot.pdf", format="pdf")
plt.close(fig)

# --- Abb. 2: F1-Vergleich ---
models = [("DDQN", ddqn)] + [(lbl, load(HERE / "baseline_results.csv", key)) for lbl, key in
                             (("Random Forest", "RandomForest"), ("SVM (RBF)", "SVM_RBF"), ("MLP", "MLP"))]
fig, ax = plt.subplots(figsize=(6.0, 3.8))
for (lbl, d), mk, ls in zip(models, "osD^", ("-", "--", "-.", ":")):
    ax.errorbar(SNRS, series(d, "mean", "f1"), yerr=series(d, "std", "f1"), marker=mk, ls=ls, lw=1.6, ms=5,
                capsize=3, label=lbl)
ax.set_xlabel("SNR in dB")
ax.set_ylabel("F1-Score")
ax.yaxis.set_major_formatter(KOMMA)
ax.set_xticks(SNRS)
ax.invert_xaxis()
ax.grid(True, ls=":", alpha=0.6)
ax.legend(loc="lower left", frameon=True)
fig.tight_layout()
fig.savefig(HERE / "baseline_vergleich_plot.pdf", format="pdf")
plt.close(fig)
print("Gespeichert: snr_robustheit_plot.pdf, baseline_vergleich_plot.pdf")
