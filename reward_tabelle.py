"""Mittlerer Reward je Fluss unter der Reward-Matrix des Agenten für DDQN, Random Forest, SVM und MLP je SNR-Stufe.

Alle Modelle werden auf genau den Testflüssen ausgewertet, die auch train_eval.evaluate / eval_baselines nutzen.
DDQN: Gewichte aus modelle/ (modelle_nachtrainieren.py). Baselines: wie eval_baselines.py neu trainiert.
Vorher wird geprüft, dass Precision/Recall/F1 aller Modelle exakt mit den CSV-Dateien übereinstimmen.
Ausgabe: Konsole und reward_tabelle.tex (vollständige tabular-Umgebung zum Einbinden mit \\input).
"""

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support

from eval_baselines import N_TRAIN, make_models
from train_eval import (N_ACTIONS, N_FEATURES, REWARD_TABLE, SEEDS, SNR_LEVELS_DB, Config, QNet, sample_flows,
                        select_device)

HERE = Path(__file__).parent
NAMEN = {"DDQN": "DDQN", "RandomForest": "Random Forest", "SVM_RBF": "SVM (RBF)", "MLP": "MLP"}


def sollwerte() -> dict:
    soll = {}
    for datei, modell in (("evaluation_results.csv", None), ("baseline_results.csv", "model")):
        with open(HERE / datei, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["seed"].isdigit():
                    m = r[modell] if modell else "DDQN"
                    soll[(m, int(r["seed"]), int(r["snr_db"]))] = (r["precision"], r["recall"], r["f1"])
    return soll


def main() -> int:
    cfg = Config()
    dev = select_device("cuda")
    soll = sollwerte()
    reward = {}          # (modell, snr) -> Liste je Seed
    fehler = False
    for seed in SEEDS:
        net = QNet(N_FEATURES, N_ACTIONS, cfg.hidden).to(dev)
        net.load_state_dict(torch.load(HERE / "modelle" / f"ddqn_seed{seed}.pt", map_location=dev))
        net.eval()
        rng = np.random.default_rng(seed + 12_345)              # wie eval_baselines.py
        snr_tr = rng.choice(np.array(SNR_LEVELS_DB, dtype=np.float64), N_TRAIN)
        x_tr, y_tr = sample_flows(rng, snr_tr, N_TRAIN)
        modelle = {name: m.fit(x_tr, y_tr) for name, m in make_models(seed).items()}
        for snr in SNR_LEVELS_DB:
            y, c = sample_flows(np.random.default_rng(seed * 1_000_003 + snr), snr, cfg.eval_flows)
            with torch.no_grad():
                pred = {"DDQN": net(torch.from_numpy(y).to(dev)).argmax(dim=1).cpu().numpy()}
            pred.update({name: m.predict(y) for name, m in modelle.items()})
            for name, a in pred.items():
                p, r, f1, _ = precision_recall_fscore_support(c, a, average="binary", pos_label=1, zero_division=0)
                ist = tuple(f"{v:.4f}" for v in (p, r, f1))
                if ist != soll[(name, seed, snr)]:
                    fehler = True
                    print(f"ABWEICHUNG {name} seed={seed} SNR={snr}: ist {ist}, soll {soll[(name, seed, snr)]}")
                reward.setdefault((name, snr), []).append(float(REWARD_TABLE[c, a].mean()))
    if fehler:
        print("Modelle reproduzieren die CSV-Dateien nicht exakt, Abbruch.")
        return 1
    print("Alle Modelle: P/R/F1 identisch mit evaluation_results.csv und baseline_results.csv")

    print(f"{'SNR':>4} " + " ".join(f"{NAMEN[m]:>16}" for m in NAMEN))
    zeilen = []
    for snr in SNR_LEVELS_DB:
        mittel = {m: np.mean(reward[(m, snr)]) for m in NAMEN}
        std = {m: np.std(reward[(m, snr)]) for m in NAMEN}
        print(f"{snr:>4} " + " ".join(f"{mittel[m]:>9.4f}±{std[m]:.4f}" for m in NAMEN))
        best = max(mittel, key=mittel.get)
        zellen = []
        for m in NAMEN:
            wert = f"{mittel[m]:.3f}".replace(".", "{,}")
            wert = rf"\mathbf{{{wert}}}" if m == best else wert
            zellen.append(f"${wert} \\pm {std[m]:.3f}$".replace("0.", "0{,}"))
        zeilen.append(f"\t\t{snr} & " + " & ".join(zellen) + r" \\")
    kopf = ("% erzeugt von reward_tabelle.py, nicht von Hand ändern\n"
            r"\begin{tabular}{ccccc}" + "\n\t" + r"\toprule" + "\n\t"
            r"\textbf{SNR in dB} & \textbf{DDQN} & \textbf{Random Forest} & \textbf{SVM (RBF)} & \textbf{MLP} \\"
            + "\n\t" + r"\midrule" + "\n")
    fuss = "\t" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n"
    (HERE / "reward_tabelle.tex").write_text(kopf + "\n".join(zeilen) + "\n" + fuss, encoding="utf-8")
    print("Gespeichert: reward_tabelle.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
