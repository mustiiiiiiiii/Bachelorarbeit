"""3D-Streudiagramm der Flussmerkmale (Feature 1, 2, 4) bei 30 dB und 5 dB SNR.

Nutzt Generator und AWGN-Kanal unverändert aus train_eval.py (sample_flows).
Ausgabe: three_d_feature_space.pdf
"""

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train_eval import ATTACK_PROB, MU_ATTACK, MU_NORMAL, sample_flows

HERE = Path(__file__).parent
FEATURES = (0, 1, 3)  # Feature 1, 2, 4 (nullbasiert)
SNRS = (30, 5)
N_FLOWS = 1500
SEED = 7
C_NORMAL, C_ATTACK = "#3b6ea5", "#c0504d"

USE_TEX = shutil.which("latex") is not None and shutil.which("dvipng") is not None
plt.rcParams.update({
    "text.usetex": USE_TEX, "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
    "legend.fontsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "pdf.fonttype": 42,
    "mathtext.fontset": "cm" if not USE_TEX else "dejavusans",
})


def main() -> None:
    fig = plt.figure(figsize=(6.3, 3.3))  # Breite = 	extwidth, Schrift damit in Originalgröße
    lims = None
    for k, snr in enumerate(SNRS):
        rng = np.random.default_rng(SEED)  # identische Flüsse, nur der Kanal unterscheidet sich
        x, c = sample_flows(rng, snr, N_FLOWS)
        p = x[:, FEATURES]
        if lims is None:
            lims = [(-0.3, 1.2)] * 3
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        for lab, col, name in ((0, C_NORMAL, "Normal"), (1, C_ATTACK, "DDoS")):
            m = c == lab
            ax.scatter(p[m, 0], p[m, 1], p[m, 2], s=3, c=col, alpha=0.7, linewidths=0, label=name)
        for mu, col in ((MU_NORMAL, C_NORMAL), (MU_ATTACK, C_ATTACK)):
            ax.scatter(*mu[list(FEATURES)], s=45, c=col, marker="X", edgecolors="black", linewidths=0.7)
        ax.set_xlabel(r"$y_1$", labelpad=-4); ax.set_ylabel(r"$y_2$", labelpad=-4); ax.set_zlabel(r"$y_4$", labelpad=-4)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_major_locator(plt.MaxNLocator(4))
        ax.tick_params(pad=-2)
        ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_zlim(*lims[2])
        ax.view_init(elev=20, azim=-55)
        ax.set_box_aspect(None, zoom=0.86)  # Platz für die z-Beschriftung
        ax.set_title(f"({'ab'[k]}) SNR $= {snr}$\\,dB" if USE_TEX else f"({'ab'[k]}) SNR = {snr} dB", pad=2)
        ax.xaxis.pane.set_alpha(0.0); ax.yaxis.pane.set_alpha(0.0); ax.zaxis.pane.set_alpha(0.0)
        ax.grid(True, alpha=0.3)
        if k == 0:
            ax.legend(loc="upper left", frameon=False, markerscale=3)
    fig.subplots_adjust(left=0.0, right=0.97, wspace=0.05, bottom=0.04, top=0.93)
    out = HERE / "three_d_feature_space.pdf"
    fig.savefig(out, dpi=300)
    print(f"gespeichert: {out} (usetex={USE_TEX}, pi={ATTACK_PROB})")


if __name__ == "__main__":
    main()
