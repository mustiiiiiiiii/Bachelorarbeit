"""Klassische Baselines (RF, SVM-RBF, MLP) auf der Datengenerierung/AWGN-Kanalmodellierung aus train_eval.py.

Training wie beim DDQN mit pro Fluss zufällig gemischtem SNR (30/20/10/5 dB); Evaluation auf
denselben frischen Flüssen (gleicher RNG-Seed je Seed/SNR) wie ``train_eval.evaluate``.
Ausgabe: baseline_results.csv (seed, model, snr_db, precision, recall, f1 + mean/std).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from train_eval import SEEDS, SNR_LEVELS_DB, Config, sample_flows

N_TRAIN = 20_000
OUT = Path(__file__).with_name("baseline_results.csv")


def make_models(seed: int) -> dict:
    return {
        "RandomForest": RandomForestClassifier(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=seed),
        "SVM_RBF": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale", random_state=seed)),
        "MLP": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 128), max_iter=200,
                                                            early_stopping=True, random_state=seed)),
    }


def main() -> None:
    n_eval = Config.eval_flows
    rows: list[dict] = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed + 12_345)  # wie Trainings-RNG des DDQN
        snr = rng.choice(np.array(SNR_LEVELS_DB, dtype=np.float64), N_TRAIN)
        x_tr, y_tr = sample_flows(rng, snr, N_TRAIN)
        for name, model in make_models(seed).items():
            model.fit(x_tr, y_tr)
            for s in SNR_LEVELS_DB:
                x_te, y_te = sample_flows(np.random.default_rng(seed * 1_000_003 + s), s, n_eval)
                p, r, f1, _ = precision_recall_fscore_support(y_te, model.predict(x_te), average="binary",
                                                              pos_label=1, zero_division=0)
                print(f"seed={seed} {name:<12} SNR={s:>2} dB  P={p:.3f} R={r:.3f} F1={f1:.3f}", flush=True)
                rows.append({"seed": seed, "model": name, "snr_db": s, "precision": p, "recall": r, "f1": f1})

    for stat, fn in (("mean", np.mean), ("std", np.std)):
        for name in make_models(0):
            for s in SNR_LEVELS_DB:
                sel = [x for x in rows if x["model"] == name and x["snr_db"] == s and isinstance(x["seed"], int)]
                rows.append({"seed": stat, "model": name, "snr_db": s,
                             **{k: float(fn([x[k] for x in sel])) for k in ("precision", "recall", "f1")}})

    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["seed", "model", "snr_db", "precision", "recall", "f1"])
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})
    print(f"Ergebnisse gespeichert: {OUT}")


if __name__ == "__main__":
    main()
