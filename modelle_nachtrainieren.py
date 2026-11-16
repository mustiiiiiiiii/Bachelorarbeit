"""Trainiert die DDQN-Agenten der Seeds 0, 1, 2 exakt wie train_eval.main() erneut und speichert die Gewichte.

Die ursprünglichen Läufe haben keine Netzgewichte gespeichert. Für die Analyse der gelernten Entscheidungsfunktion
(entscheidungsfunktion_plot.py) werden sie hier mit identischer Konfiguration, identischen Seeds und identischer
Reihenfolge der Zufallszahlen neu erzeugt. Gespeichert wird nur, wenn Precision, Recall und F1 auf allen vier
SNR-Stufen auf 4 Nachkommastellen mit evaluation_results.csv übereinstimmen. Die CSV wird nicht verändert.

Ausgabe: modelle/ddqn_seed{0,1,2}.pt
"""

import csv
import sys
import time
from pathlib import Path

import torch

from train_eval import SEEDS, SNR_LEVELS_DB, Config, DDQNAgent, evaluate, select_device, set_seed, train

HERE = Path(__file__).parent
OUT = HERE / "modelle"


def sollwerte() -> dict:
    soll = {}
    with open(HERE / "evaluation_results.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["seed"].isdigit():
                soll[(int(r["seed"]), int(r["snr_db"]))] = (r["precision"], r["recall"], r["f1"])
    return soll


def main() -> int:
    soll = sollwerte()
    cfg = Config()
    device = select_device("cuda")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    OUT.mkdir(exist_ok=True)
    alles_ok = True
    for seed in SEEDS:
        set_seed(seed)
        t0 = time.time()
        agent = DDQNAgent(cfg, device)
        train(agent, cfg, seed)
        ok = True
        for snr in SNR_LEVELS_DB:
            ist = tuple(f"{v:.4f}" for v in evaluate(agent, snr, cfg.eval_flows, seed))
            gleich = ist == soll[(seed, snr)]
            ok &= gleich
            print(f"  seed={seed} SNR={snr:>2} dB  ist P/R/F1={'/'.join(ist)}  "
                  f"soll={'/'.join(soll[(seed, snr)])}  {'OK' if gleich else 'ABWEICHUNG'}", flush=True)
        print(f"  Seed {seed}: {time.time() - t0:.0f} s", flush=True)
        if ok:
            torch.save(agent.online.state_dict(), OUT / f"ddqn_seed{seed}.pt")
        alles_ok &= ok
    print("ERGEBNIS:", "reproduziert, Gewichte gespeichert" if alles_ok else "NICHT reproduziert")
    return 0 if alles_ok else 1


if __name__ == "__main__":
    sys.exit(main())
