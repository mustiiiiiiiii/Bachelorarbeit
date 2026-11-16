"""Signalraumdarstellung der binären Entscheidung auf der Prüfgröße LLR (Log-Likelihood-Verhältnis) bei 30 und 5 dB.

Gezeigt sind die mit den Priors gewichteten Klassendichten
    (1-pi) * N(-D^2/2, D^2)   (Normal)      und      pi * N(+D^2/2, D^2)   (DDoS)
sowie die Schwellen tau(1/2) = ln((1-pi)/pi) (minimaler Fehler, symmetrische Kosten) und tau(1/3) (Soll-Schwelle der
Reward-Matrix). Die schraffierten Flächen sind die Fehleranteile bei p > 1/3: FN = pi*Phi((tau-D^2/2)/D),
FP = (1-pi)*Q((tau+D^2/2)/D). Ihre Summe bei tau(1/2) ist der Bayes-Fehler P_e^pi aus Tab. 5.1.

Ausgabe: llr_signalraum_plot.pdf
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
from scipy.stats import norm

from bayes_referenz import PI, mahalanobis_sq, tau
import plot_stil

HERE = Path(__file__).parent
C_NORMAL, C_ATTACK = "#3b6ea5", "#c0504d"   # wie Abb. 4.1/4.2
KOMMA = FuncFormatter(lambda v, _: plot_stil.zahl(v, 2, kuerzen=True) if abs(v) < 1 else plot_stil.zahl(v, 0))

plot_stil.anwenden(**{"hatch.linewidth": 0.6})


def fehler(snr: float, t: float):
    d2 = mahalanobis_sq(snr)
    d = np.sqrt(d2)
    th = float(tau(t))
    fn = PI * norm.cdf((th - d2 / 2) / d)
    fp = (1 - PI) * norm.sf((th + d2 / 2) / d)
    return fn, fp


t_half, t_third = 0.5, 1 / 3
tau_h, tau_t = float(tau(t_half)), float(tau(t_third))
print(f"tau(1/2) = {tau_h:.3f} (= ln((1-pi)/pi)),  tau(1/3) = {tau_t:.3f}")
for snr in (30, 5):
    for t in (t_half, t_third):
        fn, fp = fehler(snr, t)
        print(f"SNR {snr:>2} dB  t={t:.3f}  FN={100*fn:.2f} %  FP={100*fp:.2f} %  Summe={100*(fn+fp):.2f} %")

fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.7), sharey=False)
for ax, snr in zip(axes, (30, 5)):
    d2 = mahalanobis_sq(snr)
    d = np.sqrt(d2)
    x = np.linspace(-d2 / 2 - 3.6 * d, d2 / 2 + 3.6 * d, 1200)
    f_n = (1 - PI) * norm.pdf(x, -d2 / 2, d)
    f_a = PI * norm.pdf(x, d2 / 2, d)
    ax.plot(x, f_n, color=C_NORMAL, lw=1.4, label=r"$(1-\pi)\,p(\Lambda\mid\mathrm{Normal})$")
    ax.plot(x, f_a, color=C_ATTACK, lw=1.4, label=r"$\pi\,p(\Lambda\mid\mathrm{DDoS})$")
    # Fehlerflächen bei der Soll-Schwelle p > 1/3
    m_fn, m_fp = x < tau_t, x > tau_t
    ax.fill_between(x[m_fn], 0, f_a[m_fn], facecolor="none", edgecolor=C_ATTACK, hatch="////", lw=0,
                    label=r"FN bei $p>1/3$")
    ax.fill_between(x[m_fp], 0, f_n[m_fp], facecolor="none", edgecolor=C_NORMAL, hatch="\\\\\\\\", lw=0,
                    label=r"FP bei $p>1/3$")
    ymax = 1.12 * max(f_n.max(), f_a.max())
    ax.axvline(tau_h, color="0.15", lw=1.0, ls="-", label=r"$\tau(1/2)=\ln\frac{1-\pi}{\pi}$")
    ax.axvline(tau_t, color="0.15", lw=1.0, ls=(0, (4, 2)), label=r"$\tau(1/3)$")
    ax.annotate("", xy=(tau_t, 0.93 * ymax), xytext=(tau_h, 0.93 * ymax),
                arrowprops=dict(arrowstyle="-|>", color="0.15", lw=0.9, shrinkA=0, shrinkB=0))
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, ymax)
    ax.set_title(f"SNR = {snr} dB" + "  ($D^2=" + f"{d2:.2f}".replace(".", "{,}") + "$)", fontsize=9.5)
    ax.set_xlabel(r"Log-Likelihood-Verhältnis $\Lambda(\mathbf{y})$")
    ax.xaxis.set_major_locator(MaxNLocator(6, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_formatter(KOMMA)
    ax.yaxis.set_major_formatter(KOMMA)
axes[0].set_ylabel("gewichtete Dichte")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 0.0), fontsize=7.8)
fig.tight_layout(rect=(0, 0.17, 1, 1))
fig.savefig(HERE / "llr_signalraum_plot.pdf", format="pdf")
plt.close(fig)
print("Gespeichert: llr_signalraum_plot.pdf")
