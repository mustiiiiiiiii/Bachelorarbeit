"""Nachweis der abgeleiteten Zahlen in Text und Tabellen der Arbeit.

Jede Zahl, die nicht direkt aus einer CSV-Datei abgeschrieben ist, wird hier aus den Primärdaten nachgerechnet:
evaluation_results.csv (DDQN), baseline_results.csv (Random Forest, SVM, MLP), Generator-, Kanal- und
Trainingskonfiguration (train_eval.py, eval_baselines.py), analytische Bayes-Referenz (bayes_referenz.py) und die
gespeicherten DDQN-Gewichte in modelle/. Danach wird geprüft, ob der Wert mit seinem Kontext im LaTeX-Quelltext
steht, z. B. "schrumpft aber von $0{,}11$ auf $0{,}04$".

Rechenregel
  * Basis sind die abgedruckten Mittelwerte und Standardabweichungen (mean/std-Zeilen der CSV-Dateien, also
    Tabelle 7.1, 7.2 und A.1) bzw. Werte aus Code und Modellen. Gerundet wird genau einmal, kaufmännisch.
  * Steht eine Differenz im Text, deren Eingangswerte gerundet abgedruckt sind, muss sie auch aus den gedruckten
    Eingangswerten denselben Wert ergeben (Leserprobe).
  * "rund"-Angaben haben eine ausgewiesene Rundungseinheit (Faktoren ganzzahlig, Prozent auf Zehner).

Nicht nachgerechnet, nur verwiesen: Tabelle 7.3 (reward_tabelle.py prüft die Modelle selbst gegen beide CSV-Dateien),
der MSE-Kontrolllauf (kontrolle_mse_verlust.py) und Literaturwerte (gegen die Quell-PDFs geprüft).

Aufruf:  python nachweis_zahlen.py [--ohne-modelle]
Exit-Code 1, sobald eine Prüfung abweicht.
"""

import argparse
import csv
import math
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from bayes_referenz import PI, _var, bayes_point, huber_differenz, mahalanobis_sq, tau
from eval_baselines import N_TRAIN, make_models
from train_eval import (MU_ATTACK, MU_NORMAL, N_ACTIONS, N_FEATURES, R_FN, R_FP, R_TN, R_TP, SEEDS, SIGMA,
                        SIGNAL_POWER, SNR_LEVELS_DB, Config, QNet, sample_flows)

HERE = Path(__file__).parent
MODELLE = {"DDQN": None, "RF": "RandomForest", "SVM": "SVM_RBF", "MLP": "MLP"}
NAMEN = {"DDQN": "DDQN", "RF": "Random Forest", "SVM": "SVM (RBF)", "MLP": "MLP"}
METRIKEN = ("precision", "recall", "f1")
K2, K4, K5, K6, K7, K8, KA = ("02_grundlagen", "04_dataset", "05_implementierung", "06_evaluation",
                              "07_evaluation_ergebnisse", "08_fazit", "anhang_a")

# Generatormodell (wie bayes_referenz.py: Varianz je Merkmal sigma^2 + P_j / 10^(SNR/10))
DMU = (MU_ATTACK - MU_NORMAL).astype(np.float64)
MU_ALLE = np.concatenate([MU_NORMAL, MU_ATTACK]).astype(np.float64)
P_SIG = SIGNAL_POWER.astype(np.float64)
SIG = float(SIGMA)
IOT_DH_TEST = (748, 2000)        # Angriffsflüsse / Flüsse im Testsplit von IoT-DH (Saif et al., Literaturwert)


# ------------------------------------------------------------------ Rundung und LaTeX-Format
def rund(x, stellen: int = 2, einheit: int = 1) -> Decimal:
    """Einmal kaufmännisch gerundet. einheit=10 rundet auf Zehner ("rund 30 bis 40 %")."""
    d = Decimal(repr(round(float(x) / einheit, 10)))
    return d.quantize(Decimal(1).scaleb(-stellen), rounding=ROUND_HALF_UP) * einheit


def tex(d: Decimal, tausender: bool = False) -> str:
    """Decimal -> LaTeX-Zahl: Dezimalkomma {,}, schmales Leerzeichen als Tausendertrenner ab fünf Stellen."""
    s = format(d, "f")
    vorz, s = ("-", s[1:]) if s.startswith("-") else ("", s)
    ganz, _, rest = s.partition(".")
    if len(ganz) >= 5 or (tausender and len(ganz) >= 4):
        gruppen = []
        while len(ganz) > 3:
            gruppen.insert(0, ganz[-3:])
            ganz = ganz[:-3]
        ganz = "\\,".join([ganz] + gruppen)
    return vorz + ganz + ("{,}" + rest if rest else "")


def vz(x: float) -> str:
    """Reward mit Vorzeichen, kürzeste Form: +1, +0{,}5, -1, -2."""
    s = tex(Decimal(repr(float(x))).normalize())
    return s if s.startswith("-") else "+" + s


class Z:
    """Zahl im Text: Rechenwert, Nachkommastellen, optional Leserprobe aus den gedruckten Eingangswerten."""

    def __init__(self, wert, stellen: int = 2, leser=None, einheit: int = 1, tausender: bool = False):
        self.wert, self.stellen, self.leser = float(wert), stellen, leser
        self.einheit, self.tausender = einheit, tausender

    def soll(self) -> Decimal:
        return rund(self.wert, self.stellen, self.einheit)

    def text(self) -> str:
        return tex(self.soll(), self.tausender)


# ------------------------------------------------------------------ Daten und abgeleitete Größen
def lade_csv() -> dict:
    """-> {modell: {(seed|'mean'|'std', snr): {metrik: wert}}}"""
    daten = {m: {} for m in MODELLE}
    for datei in ("evaluation_results.csv", "baseline_results.csv"):
        with open(HERE / datei, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                m = "DDQN" if "model" not in r else next(k for k, v in MODELLE.items() if v == r["model"])
                daten[m][(r["seed"], int(r["snr_db"]))] = {k: float(r[k]) for k in METRIKEN}
    return daten


def implizite_schwelle(snr: float, recall: float) -> float:
    """Posterior-Schwelle, bei der der Bayes-Klassifikator (SNR bekannt) den gegebenen Recall erreicht."""
    return brentq(lambda t: float(bayes_point(snr, t)[1]) - recall, 0.01, 0.99, xtol=1e-12)


def sn(snr: float) -> np.ndarray:
    """Rauschstandardabweichung je Merkmal."""
    return np.sqrt(P_SIG / 10.0 ** (snr / 10.0))


def seff(snr: float) -> np.ndarray:
    """Effektive Streuung je Merkmal sqrt(sigma^2 + sigma_n,j^2)."""
    return np.sqrt(SIG**2 + sn(snr) ** 2)


def fehlerflaechen(d2: float, t: float):
    """FN- und FP-Anteil (bezogen auf alle Flüsse) des Bayes-Klassifikators mit Posterior-Schwelle t."""
    d, th = math.sqrt(d2), float(tau(t))
    return PI * norm.cdf((th - d2 / 2) / d), (1 - PI) * norm.sf((th + d2 / 2) / d)


def erwarteter_reward(d2: float, t: float) -> float:
    fn, fp = fehlerflaechen(d2, t)
    return (PI - fn) * R_TP + (1 - PI - fp) * R_TN + fp * R_FP + fn * R_FN


def winkel_grad(snr: float) -> float:
    """Drehwinkel der Bayes-Normalen w = Sigma^-1 dmu gegenüber 30 dB."""
    a, b = DMU / _var(30), DMU / _var(snr)
    return math.degrees(math.acos(min(1.0, float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))))


def trainingszaehler(cfg: Config):
    """Zählt wie train()/learn() in train_eval.py: Puffer füllt sich mit num_envs je Iteration, gelernt wird ab
    max(batch_size, learn_start) Einträgen einmal je Iteration, Zielnetz-Synchronisation alle target_update Schritte."""
    groesse = schritte = syncs = 0
    erste = voll = None
    for it in range(cfg.iterations):
        groesse = min(groesse + cfg.num_envs, cfg.buffer_size)
        if voll is None and groesse == cfg.buffer_size:
            voll = it + 1
        if groesse >= max(cfg.batch_size, cfg.learn_start):
            erste = erste or it + 1
            schritte += 1
            if schritte % cfg.target_update == 0:
                syncs += 1
    return erste, schritte, syncs, voll


def lade_reward_tabelle() -> dict:
    """reward_tabelle.tex (von reward_tabelle.py erzeugt) -> {snr: {modell: (mittel, std)}} als gedruckte Werte."""
    txt = (HERE / "reward_tabelle.tex").read_text(encoding="utf-8")
    out = {}
    for snr, zeile in re.findall(r"\n\s*(\d+)\s*&([^\n]*)", txt):
        z = [Decimal(x.replace("{,}", ".")) for x in re.findall(r"-?\d+(?:\{,\}\d+)?", zeile)]
        out[int(snr)] = {m: (z[2 * i], z[2 * i + 1]) for i, m in enumerate(("DDQN", "RF", "SVM", "MLP"))}
    return out


def modellwerte(cfg: Config, daten: dict):
    """Nulldurchgänge der Entscheidungsfunktionen (Abb. 7.2) und Posterior entlang der DDQN-Grenze (Abb. 7.5).
    Rechnet exakt wie entscheidungsfunktion_plot.py und entscheidungsgrenzen_plot.py."""
    import torch
    from sklearn.metrics import precision_recall_fscore_support

    from bayes_referenz import posterior, posterior_mischung
    from entscheidungsfunktion_plot import band, ddqn_differenz, nullstelle, testfluesse

    sammel = {s: {"p": [], "dq": [], "pm": [], "pmix": []} for s in (30, 5)}
    for seed in SEEDS:
        rng = np.random.default_rng(seed + 12_345)              # wie eval_baselines.py
        snr_tr = rng.choice(np.array(SNR_LEVELS_DB, dtype=np.float64), N_TRAIN)
        x_tr, y_tr = sample_flows(rng, snr_tr, N_TRAIN)
        mlp = make_models(seed)["MLP"].fit(x_tr, y_tr)
        for snr in SNR_LEVELS_DB:
            y, c = testfluesse(seed, snr, cfg.eval_flows)
            ist = precision_recall_fscore_support(c, mlp.predict(y), average="binary", pos_label=1,
                                                  zero_division=0)[:3]
            soll = daten["MLP"][(str(seed), snr)]
            if any(f"{a:.4f}" != f"{soll[k]:.4f}" for a, k in zip(ist, METRIKEN)):
                print(f"MLP Seed {seed}, {snr} dB reproduziert baseline_results.csv nicht.")
                return None
            if snr in sammel:
                sammel[snr]["p"].append(posterior(y, snr))
                sammel[snr]["pmix"].append(posterior_mischung(y))
                sammel[snr]["dq"].append(ddqn_differenz(seed, y, cfg))
                sammel[snr]["pm"].append(mlp.predict_proba(y)[:, 1])
    erg = {}
    edges = np.linspace(0, 1, 41)
    for snr in (30, 5):
        p, dq, pm, pmix = (np.concatenate(sammel[snr][k]) for k in ("p", "dq", "pm", "pmix"))
        xc, med = list(band(p, dq, edges))[:2]
        erg[("ddqn", snr)] = nullstelle(xc, med)
        xc, med = list(band(p, pm, edges))[:2]
        erg[("mlp", snr)] = nullstelle(xc, med - 0.5)
        for name, v in (("mischung", pmix - 0.5), ("mischung_huber", huber_differenz(pmix))):
            xc, med = list(band(p, v, edges))[:2]
            erg[(name, snr)] = nullstelle(xc, med)

    net = QNet(N_FEATURES, N_ACTIONS, cfg.hidden)
    net.load_state_dict(torch.load(HERE / "modelle" / "ddqn_seed0.pt", map_location="cpu"))
    net.eval()
    mitte = 0.5 * (MU_NORMAL + MU_ATTACK).astype(np.float64)
    g1, g4 = np.linspace(-0.25, 1.05, 400), np.linspace(-0.15, 1.05, 400)
    G1, G4 = np.meshgrid(g1, g4)
    Y = np.tile(mitte, (G1.size, 1))
    Y[:, 0], Y[:, 3] = G1.ravel(), G4.ravel()
    with torch.no_grad():
        q = net(torch.from_numpy(Y.astype(np.float32))).numpy()
    dqg = (q[:, 1] - q[:, 0]).reshape(G1.shape)
    post = posterior(Y, 30).reshape(G1.shape)
    werte = []
    for z in range(len(g4)):
        if 0.0 <= g4[z] <= 1.0:
            k = np.where((dqg[z, :-1] < 0) & (dqg[z, 1:] >= 0))[0]
            if len(k):
                werte.append(post[z, k[0]])
    erg["post_min"], erg["post_max"] = float(min(werte)), float(max(werte))
    return erg


# ------------------------------------------------------------------ Protokoll
def kurz(s: str, n: int = 96) -> str:
    return s if len(s) <= n else s[: n - 3] + "..."


class Protokoll:
    def __init__(self, texte: dict):
        self.texte, self.n_ok, self.fehler = texte, 0, []

    def abschnitt(self, titel: str) -> None:
        print(f"\n== {titel}")

    def _melde(self, ok: bool, was: str, detail: str = "") -> None:
        if ok:
            self.n_ok += 1
        else:
            self.fehler.append(was)
        print(f"  {'OK' if ok else '!!'}  {was}" + (f"   [{detail}]" if detail else ""))

    def text(self, datei: str, *teile) -> None:
        """Verkettet Literale und Zahlen zum Suchstring und prüft, ob er im Quelltext steht."""
        soll = "".join(t.text() if isinstance(t, Z) else t for t in teile)
        zahlen = [t for t in teile if isinstance(t, Z)]
        probleme = []
        if soll not in self.texte[datei]:
            anker = next((t for t in teile if isinstance(t, str) and len(t) >= 6), "")
            i = self.texte[datei].find(anker) if anker else -1
            probleme.append("im Quelltext steht: " + (repr(self.texte[datei][i:i + len(soll) + 20]) if i >= 0
                                                     else "Kontext nicht gefunden"))
        for z in zahlen:
            if z.leser is not None and rund(z.leser, z.stellen, z.einheit) != z.soll():
                probleme.append(f"Leserprobe ergibt {rund(z.leser, z.stellen, z.einheit)} statt {z.soll()}")
        detail = ", ".join(f"{z.wert:.6g}" for z in zahlen)
        self._melde(not probleme, f"{datei[:2]}: {kurz(soll)}", "; ".join([detail] + probleme) if probleme else detail)

    def bed(self, ok: bool, was: str, detail: str = "") -> None:
        self._melde(bool(ok), was, detail)

    def tabelle(self, datei: str, label: str, schluessel: str, erwartet: dict, fett=None) -> None:
        """Vergleicht jede Zahl einer Tabellenzeile mit dem Sollwert (gleiche Stellenzahl).
        fett: {zeile: index der fett gesetzten Datenzelle, gezählt ab der ersten Zelle nach dem Zeilenkopf}."""
        block = self.texte[datei].split("\\label{" + label + "}", 1)[1].split("\\end{tabular}", 1)[0]
        zeilen = dict(re.findall(r"\n\s*\$?(" + schluessel + r")\$?\s*&([^\n]*)", block))
        for key, soll in erwartet.items():
            roh = zeilen.get(str(key), "")
            ist = [x.replace("{,}", ".") for x in re.findall(r"-?\d+(?:\{,\}\d+)?", roh)]
            soll_s = [format(d, "f") for d in soll]
            ok = ist == soll_s
            detail = "" if ok else f"ist {ist}, soll {soll_s}"
            if fett is not None and ok:
                zellen = roh.split("&")
                fett_ist = [i for i, c in enumerate(zellen) if "\\mathbf" in c]
                ok = fett_ist == [fett[key]]
                detail = "" if ok else f"Fettdruck in Spalte {fett_ist}, soll [{fett[key]}]"
            self._melde(ok, f"{datei[:2]}: Tabelle {label}, Zeile {key}", detail)


# ------------------------------------------------------------------ Prüfungen
def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ohne-modelle", action="store_true", help="Abb. 7.2/7.5 nicht aus den Modellen nachrechnen")
    args = ap.parse_args()

    texte = {p.stem: p.read_text(encoding="utf-8") for p in (HERE / "kapitel").glob("*.tex")}
    pr = Protokoll(texte)
    cfg = Config()
    daten = lade_csv()
    snrs = SNR_LEVELS_DB

    def M(m, s, k="f1"):
        return daten[m][("mean", s)][k]

    def S(m, s, k="f1"):
        return daten[m][("std", s)][k]

    def r4(x):
        return rund(x, 4)

    # ---------------------------------------------------------------- CSV
    pr.abschnitt("CSV-Konsistenz: mean/std-Zeilen gegen die Seed-Einzelwerte")
    dm = ds = 0.0
    for m in MODELLE:
        for s in snrs:
            for k in METRIKEN:
                v = np.array([daten[m][(str(seed), s)][k] for seed in SEEDS])
                dm, ds = max(dm, abs(v.mean() - M(m, s, k))), max(ds, abs(v.std() - S(m, s, k)))
    pr.bed(dm <= 1e-4 + 1e-12 and ds <= 1.5e-4, "Mittelwert und Standardabweichung (n = 3, Populationsformel) "
           "passen zu den Einzelwerten", f"max. Abweichung {dm:.1e} bzw. {ds:.1e}")

    # ---------------------------------------------------------------- abgeleitete Größen
    T = {m: {s: implizite_schwelle(s, M(m, s, "recall")) for s in snrs} for m in MODELLE}
    RLA = {m: {s: 100 * (M(m, 30) - M(m, s)) / M(m, 30) for s in snrs} for m in MODELLE}
    ELA = {m: {s: (1 - M(m, s)) / M(m, 30) for s in snrs} for m in MODELLE}
    D2 = {s: mahalanobis_sq(s) for s in snrs}
    PE = {s: 100 * norm.sf(math.sqrt(D2[s]) / 2) for s in snrs}
    PEPI = {s: 100 * sum(fehlerflaechen(D2[s], 0.5)) for s in snrs}
    D2_0 = float(np.sum(DMU**2 / SIG**2))
    PEPI_0 = 100 * sum(fehlerflaechen(D2_0, 0.5))
    BAYES = {s: (float(bayes_point(s, 0.5)[2]), float(bayes_point(s, 1 / 3)[2])) for s in snrs}
    NST_HUBER = brentq(lambda p: float(huber_differenz(p)), 0.01, 0.49, xtol=1e-14)
    P_STERN = (R_TN - R_FP) / ((R_TP - R_FN) + (R_TN - R_FP))
    SN_20 = max(sn(30).max(), sn(20).max())
    mlp0 = make_models(0)["MLP"].named_steps["mlpclassifier"]
    rf0 = make_models(0)["RandomForest"]
    svc0 = make_models(0)["SVM_RBF"].named_steps["svc"]
    src_train = (HERE / "train_eval.py").read_text(encoding="utf-8")
    src_base = (HERE / "eval_baselines.py").read_text(encoding="utf-8")
    clip = float(re.search(r"clip_grad_norm_\(self\.online\.parameters\(\), ([\d.]+)\)", src_train).group(1))
    n_3d = int(re.search(r"^N_FLOWS = (\d+)", (HERE / "create_3d_plot.py").read_text(encoding="utf-8"), re.M).group(1))
    n_param = sum(p.numel() for p in QNet(N_FEATURES, N_ACTIONS, cfg.hidden).parameters())
    erste, schritte, syncs, voll = trainingszaehler(cfg)
    decay = max(1, int(cfg.eps_decay_frac * cfg.iterations))
    n_dqn = cfg.iterations * cfg.num_envs
    e_dqn = int(math.floor(math.log10(n_dqn)))
    e_base = int(math.floor(math.log10(N_TRAIN)))
    rw = lade_reward_tabelle()
    rw_diff = [float(rw[s]["DDQN"][0] - rw[s]["MLP"][0]) for s in snrs]
    rw_std = max(float(v[1]) for s in snrs for v in rw[s].values())
    rng = np.random.default_rng(0 + 12_345)
    x_tr, _ = sample_flows(rng, rng.choice(np.array(snrs, dtype=np.float64), N_TRAIN), N_TRAIN)
    xs = make_models(0)["SVM_RBF"].named_steps["standardscaler"].fit_transform(x_tr)
    gamma_scale = 1.0 / (N_FEATURES * xs.var())
    reward_satz = (r"$R_{\mathrm{TP}}=" + vz(R_TP) + r"$, $R_{\mathrm{TN}}=" + vz(R_TN) + r"$, $R_{\mathrm{FP}}="
                   + vz(R_FP) + r"$, $R_{\mathrm{FN}}=" + vz(R_FN) + "$")

    # ---------------------------------------------------------------- Kapitel 2 und 4
    pr.abschnitt("Kapitel 2 und 4: Generatormodell und Kanal")
    pr.text(K2, r"Bei $", Z(100 * (1 - PI), 0), r"\,\%$ Normalverkehr")
    pr.text(K2, r"bereits $", Z(100 * (1 - PI), 0), r"\,\%$ Accuracy")
    pr.text(K2, r"im hier verwendeten Modell $", Z(sn(5).max(), 2), r"$ und übersteigt die Klassenstreuung $\sigma=",
            Z(SIG, 2), "$")
    pr.bed(sn(5).max() > SIG, "Kap. 2: Rauschstandardabweichung bei 5 dB übersteigt sigma")
    pr.text(K4, r"Die Klassenmittelwerte liegen zwischen $", Z(MU_ALLE.min(), 2), r"$ und $", Z(MU_ALLE.max(), 2),
            r"$, die Streuung beträgt $\sigma=", Z(SIG, 2), "$")
    pr.text(K4, r"ausschließlich auf den $", Z(N_TRAIN, 0), "$ Trainingsflüssen")
    pr.text(K4, r"intern $", Z(100 * mlp0.validation_fraction, 0), r"\,\%$ seiner Trainingsflüsse")
    pr.bed(mlp0.early_stopping, "Kap. 4/6: MLP nutzt Early Stopping")
    pr.text(K4, "Startwert $1\\,000\\,003\\,s+")
    pr.bed("default_rng(seed * 1_000_003 + s)" in src_base and "default_rng(seed * 1_000_003 + int(snr_db))" in src_train,
           "Kap. 4: Test-Startwert 1 000 003 s + SNR in train_eval.py und eval_baselines.py")
    pr.text(K4, r"Mittelwertabständen von $", Z(np.abs(DMU).min(), 2), r"$ bis $", Z(np.abs(DMU).max(), 2), "$")
    pr.text(K4, "Bei vier Merkmalen liegt der Angriffsmittelwert über dem Normalwert, bei vier darunter")
    pr.bed((DMU > 0).sum() == 4 and (DMU < 0).sum() == 4, "Kap. 4: je vier Merkmale mit positivem/negativem dmu")
    pr.text(K4, r"nur um $", Z(np.abs(DMU).min(), 2), r"$ bis $", Z(np.abs(DMU).max(), 2), "$ und liegen")
    pr.text(K4, r"Der Angriffsanteil beträgt $\pi=", Z(PI, 1), "$")
    pr.text(K4, "des Angriffsanteils von ", Z(100 * IOT_DH_TEST[0] / IOT_DH_TEST[1], 0), r"\,\% im Testsplit")
    pr.text(K4, r"($\mu_{\text{Normal},1}=", Z(MU_NORMAL[0], 2), r"$, $\mu_{\text{DDoS},1}=", Z(MU_ATTACK[0], 2), "$)")
    pr.text(K4, r"bleibt $\sigma_{n,1}\approx", Z(sn(30)[0], 3), r"$ neben $\sigma=", Z(SIG, 2), "$")
    pr.text(K4, r"\sqrt{\sigma^2+\sigma_{n,1}^2}\approx", Z(seff(5)[0], 2), "$")
    pr.text(K4, r"dieselben $", Z(n_3d, 0, tausender=True), "$ Flüsse")

    # ---------------------------------------------------------------- Kapitel 5
    pr.abschnitt("Kapitel 5: Reward, Huber-Verlust, Kanal, Tabelle 5.1, Training, Tabellen 5.2/5.3")
    pr.text(K5, r"R_{\mathrm{TP}}=+", Z(R_TP, 1), r", & a=1,\ c=1")
    pr.text(K5, r"R_{\mathrm{TN}}=+", Z(R_TN, 1), r", & a=0,\ c=0")
    pr.text(K5, r"R_{\mathrm{FP}}=", Z(R_FP, 1), r", & a=1,\ c=0")
    pr.text(K5, r"R_{\mathrm{FN}}=", Z(R_FN, 1), r", & a=0,\ c=1")
    pr.text(K5, r"\textbf{Diskontierung:} $\gamma=", Z(cfg.gamma, 2), "$")
    pr.text(K5, r"liegen aber $", Z(R_TP - R_FP, 0), r"$ (Aktion DDoS: $" + vz(R_TP) + "$ oder $" + vz(R_FP)
            + r"$) bzw. $", Z(R_TN - R_FN, 1), r"$ (Aktion Normal: $" + vz(R_FN) + "$ oder $" + vz(R_TN) + "$)")
    pr.text(K5, r"mit der Nullstelle $p=3/7\approx", Z(NST_HUBER, 2), "$")
    pr.bed(abs(NST_HUBER - 3 / 7) < 1e-9, "Kap. 5: Nullstelle der Huber-Differenz ist 3/7", f"{NST_HUBER:.12f}")
    pr.text(K5, r"beträgt die Rauschvarianz $10^{", Z(-30 / 10, 0), r"}P_j$, bei $5$\,dB etwa $", Z(10 ** -0.5, 2),
            r"\,P_j$")
    pr.text(K5, r"etwa $", Z(100 * math.sqrt(1e-3), 0), r"\,\%$ (bei $30$\,dB) bis $", Z(100 * math.sqrt(10 ** -0.5), 0),
            r"\,\%$ (bei $5$\,dB)")
    pr.text(K5, r"mit $", Z(sn(5).min(), 2), r"$ bis $", Z(sn(5).max(), 2), r"$ je Merkmal über der intrinsischen "
            r"Klassenstreuung $\sigma=", Z(SIG, 2), "$")
    pr.text(K5, r"wächst von $\sigma^2=", Z(SIG**2, 2), "$")
    pr.text(K5, r"beträgt $\pm", Z(np.abs(DMU).min(), 2), r"$ bis $\pm", Z(np.abs(DMU).max(), 2), "$")
    pr.tabelle(K5, "tab:klassenueberlappung", r"\d+",
               {s: [rund(sn(s).max(), 3), rund(D2[s], 2), rund(math.sqrt(D2[s]), 2), rund(PE[s], 1), rund(PEPI[s], 1)]
                for s in snrs})
    pr.text(K5, r"weil das Rauschen ($\sigma_{n,j}\le", Z(SN_20, 3), "$)")
    pr.text(K5, r"Bei $5$\,dB sinkt $D$ auf $", Z(math.sqrt(D2[5]), 2), r"$, die effektive Streuung steigt je nach "
            r"Merkmal auf $", Z(seff(5).min(), 2), r"$ bis $", Z(seff(5).max(), 2), "$")
    pr.text(K5, "auf fast das Dreifache")
    pr.bed(round(PE[5] / PE[30]) == 3 and PE[5] / PE[30] < 3, "Kap. 5: P_e(5 dB)/P_e(30 dB) fast das Dreifache",
           f"{PE[5] / PE[30]:.3f}")
    pr.text(K5, r"von $", Z(PEPI[30], 1), r"\,\%$ bei $30$\,dB bis $", Z(PEPI[5], 1), r"\,\%$ bei $5$\,dB")
    fn30, fp30 = (100 * v for v in fehlerflaechen(D2[30], 0.5))
    fn5, fp5 = (100 * v for v in fehlerflaechen(D2[5], 0.5))
    fn5s, fp5s = (100 * v for v in fehlerflaechen(D2[5], 1 / 3))
    pr.text(K5, r"bei $30$\,dB $", Z(fn30, 1), r"\,\%$ verpasste Angriffe und $", Z(fp30, 1), r"\,\%$ Fehlalarme, "
            r"bei $5$\,dB $", Z(fn5, 1), r"$ und $", Z(fp5, 1), r"\,\%$")
    pr.text(K5, r"um $\ln 2\approx", Z(float(tau(0.5) - tau(1 / 3)), 2), "$ weiter links")
    pr.text(K5, r"sinkt der FN-Anteil auf $", Z(fn5s, 1), r"\,\%$, der FP-Anteil steigt auf $", Z(fp5s, 1), r"\,\%$")
    pr.text(K5, r"Der Gesamtfehler wächst dabei von $", Z(fn5 + fp5, 1), r"$ auf $", Z(fn5s + fp5s, 1), r"\,\%$")
    pr.bed(erwarteter_reward(D2[5], 1 / 3) > erwarteter_reward(D2[5], 0.5),
           "Kap. 5: erwartete Kosten bei 5 dB sinken mit p > 1/3",
           f"E[R] {erwarteter_reward(D2[5], 0.5):.4f} -> {erwarteter_reward(D2[5], 1 / 3):.4f}")
    pr.text(K5, r"erfüllt ($\sigma_{n,j}\le", Z(SN_20, 3), r"$), bei $5$\,dB mit $\sigma_{n,j}$ bis $",
            Z(sn(5).max(), 2), "$")
    pr.text(K5, r"innerhalb der ersten $", Z(100 * cfg.eps_decay_frac, 0), r"\,\%$ der Iterationen ($", Z(decay, 0), "$)")
    pr.text(K5, r"Adam mit Lernrate $10^{", Z(math.log10(cfg.lr), 0), "}$")
    pr.text(K5, r"alle $", Z(cfg.target_update, 0), "$ Lernschritte hart synchronisiert")
    pr.text(K5, r"je Iteration $", Z(cfg.num_envs, 0), r"$ Transitionen aufnimmt, erreicht er die Mindestfüllung von $",
            Z(cfg.learn_start, 0), r"$ Einträgen in Iteration $", Z(erste, 0), "$")
    pr.text(K5, r"damit $", Z(schritte, 0), r"$ Lernschritte mit je $", Z(cfg.batch_size, 0),
            r"$ Transitionen aus und synchronisiert das Zielnetz $", Z(syncs, 0), "$-mal")
    pr.text(K5, r"Ab Iteration $", Z(voll, 0), "$ ist der Puffer voll")
    pr.bed("agent.learn()" in src_train and "max(self.cfg.batch_size, self.cfg.learn_start)" in src_train
           and "self.size = min(self.size + n, self.cap)" in src_train,
           "Kap. 5: Zählweise entspricht train()/learn()/ReplayBuffer in train_eval.py")
    pr.text(K5, r"\Ein{$N=", Z(cfg.iterations, 0), r"$ Iterationen, $B_{\mathrm{f}}=", Z(cfg.num_envs, 0),
            r"$ Flüsse je Iteration, Mini-Batch $B=", Z(cfg.batch_size, 0), r"$, Lernstart $L=", Z(cfg.learn_start, 0),
            r"$, Zielnetz-Takt $C=", Z(cfg.target_update, 0), r"$, Lernrate $\alpha=10^{", Z(math.log10(cfg.lr), 0), "}$}")
    pr.text(K5, r"(Kapazität $", Z(cfg.buffer_size, 0), "$)")
    pr.text(K5, r"\max\bigl(", Z(cfg.eps_end, 2), r";\ ", Z(cfg.eps_start, 1), "-", Z(cfg.eps_start - cfg.eps_end, 2),
            r"\,i/(", Z(cfg.eps_decay_frac, 1), r"\,N)\bigr)")
    pr.text(K5, r"Gradientennorm $\le", Z(clip, 0), "$")
    pr.text(K5, "Netzstruktur & $", Z(N_FEATURES, 0), r"\to", Z(cfg.hidden, 0), r"\to", Z(cfg.hidden, 0), r"\to",
            Z(N_ACTIONS, 0), "$")
    pr.text(K5, r"Lernrate & $", Z(cfg.lr / 10 ** math.floor(math.log10(cfg.lr)), 0), r"\cdot10^{",
            Z(math.floor(math.log10(cfg.lr)), 0), "}$")
    pr.text(K5, r"Diskontierungsfaktor $\gamma$ & $", Z(cfg.gamma, 2), "$")
    pr.text(K5, "Batch-Größe & $", Z(cfg.batch_size, 0), "$")
    pr.text(K5, "Puffergröße & $", Z(cfg.buffer_size, 0), "$")
    pr.text(K5, r"Explorationsrate $\epsilon$ & $", Z(cfg.eps_start, 1), r" \to ", Z(cfg.eps_end, 2), "$")
    pr.text(K5, r"Linear über $", Z(100 * cfg.eps_decay_frac, 0), r"\,\%$ der Iterationen")
    pr.text(K5, "Lernstart & $", Z(cfg.learn_start, 0), "$ Einträge")
    pr.text(K5, "Zielnetz-Update & alle $", Z(cfg.target_update, 0), "$ Lernschritte")
    pr.text(K5, r"Gradientennorm begrenzt auf $", Z(clip, 0), "$")
    pr.text(K5, "Iterationen & $", Z(cfg.iterations, 0), "$ & Je $", Z(cfg.num_envs, 0), r"$ Flüsse, insgesamt $",
            Z(n_dqn / 10 ** e_dqn, 1), r"\cdot10^{", Z(e_dqn, 0), "}$ Flüsse")
    pr.text(K5, r"Klassenstreuung $\sigma$ & $", Z(SIG, 2), "$")
    pr.text(K5, r"Angriffsanteil $\pi$ & $", Z(PI, 1), "$")
    pr.text(K5, "Evaluierte SNR-Werte & $" + ",\\,".join(str(s) for s in snrs) + "$\\,dB")
    pr.text(K5, "$(" + ";\\,".join(vz(r) for r in (R_TP, R_TN, R_FP, R_FN)) + ")$")
    pr.text(K5, "Seeds & $" + ",\\,".join(str(s) for s in SEEDS) + "$")
    pr.text(K5, "Testflüsse & $", Z(cfg.eval_flows, 0), "$ je SNR und Seed")

    # ---------------------------------------------------------------- Kapitel 6
    pr.abschnitt("Kapitel 6: Baselines und Bayes-Fehler der Referenzstufe")
    pr.text(K6, r"das intern $", Z(100 * mlp0.validation_fraction, 0), r"\,\%$ seiner Trainingsflüsse")
    pr.text(K6, r"auf $", Z(cfg.eval_flows, 0), "$ frischen Flüssen")
    pr.text(K6, "trainieren aber überwacht auf ", Z(N_TRAIN, 0), " Flüssen")
    pr.text(K6, "besteht aus ", Z(rf0.n_estimators, 0), " Bäumen mit mindestens fünf Trainingsbeispielen je Blatt")
    pr.bed(rf0.min_samples_leaf == 5, "Kap. 6: min_samples_leaf = 5")
    pr.text(K6, r"RBF-Kernel mit $C=", Z(svc0.C, 0), r'$ und \texttt{gamma=\string"scale\string"}, nach der Standardisierung also '
            r"$\gamma\approx1/8$")
    pr.bed(svc0.gamma == "scale" and abs(gamma_scale - 1 / 8) < 1e-3, "Kap. 6: gamma='scale' ergibt 1/8",
           f"{gamma_scale:.6f}")
    pr.text(K6, "zwei verdeckte Schichten mit je ", Z(mlp0.hidden_layer_sizes[0], 0), " ReLU-Neuronen")
    pr.bed(tuple(mlp0.hidden_layer_sizes) == (cfg.hidden, cfg.hidden), "Kap. 6: MLP (128, 128) wie das Q-Netz")
    pr.text(K6, "über höchstens ", Z(mlp0.max_iter, 0), " Epochen")
    pr.text(K6, r"ohne Rauschen $", Z(PEPI_0, 2), r"\,\%$ und bei $30$\,dB $", Z(PEPI[30], 2), r"\,\%$")

    # ---------------------------------------------------------------- Kapitel 7: DDQN
    pr.abschnitt("Kapitel 7.1: DDQN über die SNR-Stufen, Tabelle 7.1, implizite Schwellen")
    pr.text(K7, "mit ", Z(cfg.iterations, 0), " Iterationen à ", Z(cfg.num_envs, 0), " parallel erzeugten")
    pr.text(K7, "auf je ", Z(cfg.eval_flows, 0), " frischen Flüssen pro SNR-Stufe")
    pr.tabelle(K7, "tab:ddqn_snr", r"\d+",
               {s: [r4(f(m_, s, k)) for k in METRIKEN for f, m_ in ((M, "DDQN"), (S, "DDQN"))] for s in snrs})
    pr.text(K7, r"fällt monoton von $", Z(M("DDQN", 30), 4), r"$ bei 30\,dB über $", Z(M("DDQN", 20), 4),
            r"$ (20\,dB) und $", Z(M("DDQN", 10), 4), r"$ (10\,dB) auf $", Z(M("DDQN", 5), 4), r"$ bei 5\,dB")
    pr.bed(all(M(m, a) > M(m, b) for m in MODELLE for a, b in zip(snrs, snrs[1:])),
           "Kap. 7: F1 fällt bei allen Verfahren monoton")
    pr.text(K7, r"verliert der Agent nur $", Z(100 * (M("DDQN", 30) - M("DDQN", 20)), 1),
            r"$ Prozentpunkte, zwischen 10 und 5\,dB dagegen $", Z(100 * (M("DDQN", 10) - M("DDQN", 5)), 1), "$")
    pr.text(K7, r"bei 5\,dB rund $", Z(10 ** -0.5, 2), r"\,P_j$")
    pr.text(K7, r"Klassenstreuung von $\sigma=", Z(SIG, 2), "$")
    pr.text(K7, r"Mittelwertdifferenzen von $", Z(np.abs(DMU).min(), 2), r"$ bis $", Z(np.abs(DMU).max(), 2), "$")
    std_max = max(S("DDQN", s, k) for s in snrs for k in METRIKEN)
    pr.text(K7, r"($\sigma\le ", Z(std_max, 4), "$ bei allen Metriken")
    fakt = {k: [S("DDQN", s, k) / S("DDQN", 30, k) for s in (10, 5)] for k in METRIKEN}
    pr.bed(all(round(v) == 2 for v in fakt["precision"]), "Kap. 7: Precision-Streuung bei 10/5 dB etwa das Zweifache",
           ", ".join(f"{v:.2f}" for v in fakt["precision"]))
    pr.bed(all(round(v) == 4 for v in fakt["f1"]), "Kap. 7: F1-Streuung bei 10/5 dB etwa das Vierfache",
           ", ".join(f"{v:.2f}" for v in fakt["f1"]))
    pr.text(K7, r"nur um rund $", Z(100 * (fakt["recall"][0] - 1), 0, einheit=10), r"$ bis $",
            Z(100 * (fakt["recall"][1] - 1), 0, einheit=10), r"\,\%$")
    pr.text(K7, r"am deutlichsten bei 5\,dB ($", Z(M("DDQN", 5, "recall"), 4), r"$ gegenüber $",
            Z(M("DDQN", 5, "precision"), 4), "$)")
    dpr = {s: M("DDQN", s, "recall") - M("DDQN", s, "precision") for s in snrs}
    pr.bed(min(dpr.values()) > 0 and max(dpr, key=dpr.get) == 5,
           "Kap. 7: Recall > Precision auf allen Stufen, Abstand am größten bei 5 dB")
    pr.text(K7, "Belohnungsstruktur " + reward_satz)
    pr.text(K7, r"Q(\mathbf{y},1)&=p\cdot " + tex(Decimal(repr(R_TP)).normalize()) + r"+(1-p)("
            + tex(Decimal(repr(R_FP)).normalize()) + ")=2p-1")
    pr.bed(R_TP - R_FP == 2 and R_FP == -1, "Kap. 7: Q(y,1) = 2p - 1 aus R_TP, R_FP")
    pr.text(K7, r"Q(\mathbf{y},0)&=p\cdot(" + tex(Decimal(repr(R_FN)).normalize()) + r")+(1-p)\cdot ", Z(R_TN, 1), "=",
            Z(R_TN, 1), "-", Z(R_TN - R_FN, 1), r"\,p")
    pr.text(K7, "also für $p>1/3$")
    pr.bed(abs(P_STERN - 1 / 3) < 1e-12, "Kap. 7: Q(y,1) = Q(y,0) bei p = 1/3", f"{P_STERN:.12f}")
    pr.text(K7, r"ergibt sich $t=", Z(T["DDQN"][30], 2), "$, $", Z(T["DDQN"][20], 2), "$, $", Z(T["DDQN"][10], 2),
            "$ und $", Z(T["DDQN"][5], 2), r"$ bei 30, 20, 10 und 5\,dB")
    pr.bed(min(T["DDQN"].values()) > 1 / 3, "Kap. 7: DDQN erreicht 1/3 auf keiner Stufe")
    pr.text(K7, r"Der Random Forest liegt bei $t=", Z(T["RF"][30], 2), r"$ bis $", Z(T["RF"][5], 2),
            r"$, die SVM bei $", Z(T["SVM"][30], 2), r"$ bis $", Z(T["SVM"][5], 2), r"$ und das MLP bei $",
            Z(T["MLP"][30], 2), r"$ bis $", Z(T["MLP"][5], 2), "$")
    pr.bed(all(T[m][a] > T[m][b] for m in MODELLE for a, b in zip(snrs, snrs[1:])),
           "Kap. 7: Schwellen fallen von 30 nach 5 dB bei allen Verfahren")
    pr.bed(all(T["RF"][s] > T["SVM"][s] > T["MLP"][s] > T["DDQN"][s] for s in snrs),
           "Kap. 7: Rangfolge der Schwellen RF > SVM > MLP > DDQN auf allen Stufen")
    pr.text(K7, r"schrumpft aber von $",
            Z(T["MLP"][30] - T["DDQN"][30], 2, leser=rund(T["MLP"][30], 3) - rund(T["DDQN"][30], 3)), r"$ auf $",
            Z(T["MLP"][5] - T["DDQN"][5], 2, leser=rund(T["MLP"][5], 3) - rund(T["DDQN"][5], 3)), "$")
    pr.text(K7, r"Die Sollgerade $", Z((R_TP - R_FP) + (R_TN - R_FN), 1), r"\,p-", Z(R_TN - R_FP, 1), "$")
    pr.text(K7, r"liegt die Nullstelle bei $3/7\approx", Z(NST_HUBER, 2), "$")
    pr.text(K7, r"Seeds 0, 1 und 2 ($", Z(len(SEEDS) * cfg.eval_flows, 0), "$ je Stufe)")
    pr.text(K7, r"verliert die Precision $", Z(100 * (M("DDQN", 30, "precision") - M("DDQN", 5, "precision")), 1),
            r"$ Prozentpunkte, der Recall $", Z(100 * (M("DDQN", 30, "recall") - M("DDQN", 5, "recall")), 1), "$")
    pr.text(K7, r"(Priorwahrscheinlichkeit $", Z(1 - PI, 1), "$)")

    # ---------------------------------------------------------------- Kapitel 7: Abb. 7.2 und 7.5 aus den Modellen
    mw = None
    if args.ohne_modelle:
        print("\n== Abbildung 7.2 und 7.5: übersprungen (--ohne-modelle)")
    else:
        pr.abschnitt("Kapitel 7: Abbildung 7.2 und 7.5 aus den gespeicherten Modellen (MLP gegen CSV geprüft)")
        mw = modellwerte(cfg, daten)
        pr.bed(mw is not None, "MLP aller Seeds reproduziert baseline_results.csv auf vier Stellen")
    if mw:
        t_dq, t_m, t_mix, t_hm = ({s: mw[(k, s)] for s in (30, 5)} for k in ("ddqn", "mlp", "mischung", "mischung_huber"))
        pr.text(K7, r"Ihr Median schneidet die Null bei $", Z(t_dq[30], 3), r"$ (30\,dB) und $", Z(t_dq[5], 3),
                r"$ (5\,dB)")
        pr.bed(all(abs(t_dq[s] - T["DDQN"][s]) < 0.02 for s in (30, 5)),
               "Kap. 7: Nulldurchgang bestätigt die Recall-Schwellen (Abstand < 0,02)",
               ", ".join(f"{t_dq[s]:.3f}/{T['DDQN'][s]:.3f}" for s in (30, 5)))
        pr.bed(abs(t_dq[30] - 3 / 7) < 0.02, "Kap. 7: bei 30 dB schneidet das Netz nahe 3/7", f"{t_dq[30]:.3f}")
        pr.text(K7, r"Nulldurchgang von $", Z(t_hm[30], 3), r"$ bei 30\,dB und $", Z(t_hm[5], 3),
                r"$ bei 5\,dB, die gemessenen Werte liegen um $",
                Z(t_hm[30] - t_dq[30], 2, leser=rund(t_hm[30], 3) - rund(t_dq[30], 3)), r"$ bzw. $",
                Z(t_hm[5] - t_dq[5], 2, leser=rund(t_hm[5], 3) - rund(t_dq[5], 3)), "$ darunter")
        pr.text(K7, r"überschreitet $1/2$ bei $p=", Z(t_mix[30], 3), r"$ bzw. $", Z(t_mix[5], 3), r"$, das MLP bei $",
                Z(t_m[30], 3), r"$ bzw. $", Z(t_m[5], 3), "$")
        pr.bed(abs(t_mix[30] - t_m[30]) < 0.005 and (0.5 - t_mix[5]) / (0.5 - t_m[5]) > 0.5,
               "Kap. 7: SNR-Mischung erklärt die MLP-Verschiebung bei 30 dB vollständig, bei 5 dB zum größeren Teil",
               f"Anteil bei 5 dB {(0.5 - t_mix[5]) / (0.5 - t_m[5]):.2f}")
        pr.text(K7, r"schwankt die Posterior bei 30\,dB zwischen $", Z(mw["post_min"], 2), r"$ und $",
                Z(mw["post_max"], 2), "$")

    # ---------------------------------------------------------------- Kapitel 7: Baselines
    pr.abschnitt("Kapitel 7.2: Baselines, Tabelle 7.2, mittlerer Reward")
    pr.text(K7, "Random Forest (", Z(rf0.n_estimators, 0), " Bäume)")
    pr.text(K7, "zwei verdeckten Schichten à ", Z(mlp0.hidden_layer_sizes[0], 0), " Neuronen")
    pr.text(K7, "überwacht auf ", Z(N_TRAIN, 0), " Flüssen mit pro Fluss gemischtem SNR")
    reihen = ("DDQN", "RF", "SVM", "MLP")
    pr.tabelle(K7, "tab:f1_vergleich", r"\d+", {s: [r4(f(m, s)) for m in reihen for f in (M, S)] for s in snrs},
               fett={s: max(range(4), key=lambda i: M(reihen[i], s)) for s in snrs})
    pr.bed(all(M("DDQN", s) > M("MLP", s) > M("SVM", s) > M("RF", s) for s in snrs),
           "Kap. 7/8: Rangfolge DDQN > MLP > SVM > RF auf allen Stufen")
    d_mlp = {s: 100 * (M("DDQN", s) - M("MLP", s)) for s in snrs}
    d_svm = [100 * (M("DDQN", s) - M("SVM", s)) for s in snrs]
    d_rf = [100 * (M("DDQN", s) - M("RF", s)) for s in snrs]
    pr.text(K7, r"beträgt er $", Z(d_mlp[30], 2), r"$ (30\,dB), $", Z(d_mlp[20], 2), r"$ (20\,dB), $", Z(d_mlp[10], 2),
            r"$ (10\,dB) und $", Z(d_mlp[5], 2), r"$ Prozentpunkte (5\,dB)")
    pr.text(K7, r"Die SVM liegt zwischen $", Z(min(d_svm), 2), r"$ und $", Z(max(d_svm), 2),
            r"$, der Random Forest zwischen $", Z(min(d_rf), 2), r"$ und $", Z(max(d_rf), 2), "$ Prozentpunkten zurück")
    pr.text(K7, r"($\sigma_{\mathrm{MLP}}=", Z(S("MLP", 10), 4), "$)")
    pr.bed(d_mlp[10] / 100 < S("DDQN", 10) + S("MLP", 10), "Kap. 7: Fehlerbalken von DDQN und MLP überlappen bei 10 dB")
    pr.text(K7, r"Beim Random Forest betragen Precision und Recall bei 30\,dB $", Z(M("RF", 30, "precision"), 4),
            r"$ und $", Z(M("RF", 30, "recall"), 4), r"$, beim DDQN $", Z(M("DDQN", 30, "precision"), 4), r"$ und $",
            Z(M("DDQN", 30, "recall"), 4), "$")
    t_base30 = [T[m][30] for m in ("RF", "SVM", "MLP")]
    pr.text(K7, r"($t=", Z(min(t_base30), 2), r"$ bis $", Z(max(t_base30), 2), "$, Abschnitt")
    pr.text(K7, r"bei 30\,dB $", Z(rw[30]["DDQN"][0], 3), r"$ gegenüber $", Z(rw[30]["MLP"][0], 3), "$ beim MLP")
    pr.bed(all(rw[s]["DDQN"][0] == max(v[0] for v in rw[s].values()) for s in snrs),
           "Kap. 7/8: DDQN hat auf allen Stufen den höchsten mittleren Reward")
    pr.text(K7, r"Der Abstand zum MLP von rund $", Z(min(rw_diff), 3), r"$ bis $", Z(max(rw_diff), 3), "$ je Fluss")
    pr.text(K7, r"Bei Seed-Streuungen bis $", Z(rw_std, 3), "$")
    pr.text(K7, r"rund $", Z(n_dqn / 10 ** e_dqn, 1), r"\cdot 10^{", Z(e_dqn, 0), "}$ Trainingsflüsse (",
            Z(cfg.iterations, 0), r" Iterationen $\times$ ", Z(cfg.num_envs, 0), r"), die Baselines nur $",
            Z(N_TRAIN / 10 ** e_base, 0), r"\cdot 10^{", Z(e_base, 0), "}$")
    pr.text(K7, "mit Angriffsanteil $", Z(PI, 1), "$")

    # ---------------------------------------------------------------- Kapitel 7: RLA/ELA und Bayes-Referenz
    pr.abschnitt("Kapitel 7.3: Tabelle 7.4 (RLA/ELA aus Tabelle 7.2), Tabelle 7.5, Bayes-Abstände")
    zeile = {"DDQN": "DDQN", "MLP": "MLP", "SVM": "SVM (RBF)", "RF": "Random Forest"}
    pr.tabelle(K7, "tab:rla_ela", r"DDQN|MLP|SVM \(RBF\)|Random Forest",
               {zeile[m]: [rund(RLA[m][s], 2) for s in (20, 10, 5)] + [rund(ELA[m][s], 3) for s in snrs]
                for m in MODELLE})
    pr.bed(all(RLA["DDQN"][s] == min(RLA[m][s] for m in MODELLE) for s in (20, 10, 5))
           and all(ELA["DDQN"][s] == min(ELA[m][s] for m in MODELLE) for s in snrs),
           "Kap. 7: DDQN-Zeile (fett) hat in jeder Spalte den kleinsten Wert")
    pr.bed(all(ELA["DDQN"][s] < ELA["MLP"][s] < ELA["SVM"][s] < ELA["RF"][s] for s in snrs),
           "Kap. 7: ELA ordnet auf jeder Stufe DDQN vor MLP, SVM und RF")
    pr.text(K7, r"($", Z(RLA["SVM"][20], 2), r"$ gegenüber $", Z(RLA["MLP"][20], 2), r"\,\%$)")
    pr.bed(RLA["SVM"][20] < RLA["MLP"][20] and M("SVM", 30) < M("MLP", 30) and M("SVM", 20) < M("MLP", 20),
           "Kap. 7: SVM verliert bei 20 dB relativ weniger als das MLP trotz niedrigerem F1")
    f30 = [M(m, 30) for m in MODELLE]
    pr.text(K7, r"nur $", Z(max(f30) - min(f30), 2, leser=rund(max(f30), 3) - rund(min(f30), 3)),
            r"$ auseinanderliegen ($", Z(min(f30), 3), r"$ bis $", Z(max(f30), 3), "$)")
    r10 = [RLA[m][10] for m in ("DDQN", "MLP", "SVM")]
    r10_druck = [rund(v, 2) for v in r10]
    pr.text(K7, r"in der RLA nur $", Z(max(r10) - min(r10), 2, leser=max(r10_druck) - min(r10_druck)),
            r"$ Prozentpunkte auseinander, der Random Forest $",
            Z(RLA["RF"][10] - RLA["DDQN"][10], 1, leser=rund(RLA["RF"][10], 2) - rund(RLA["DDQN"][10], 2)),
            "$ Prozentpunkte hinter dem DDQN")
    r5 = [RLA[m][5] for m in MODELLE]
    pr.text(K7, r"zwischen $", Z(min(r5), 2), r"$ und $", Z(max(r5), 2), r"\,\%$ ihrer Ausgangsleistung")
    pr.text(K7, r"von $", Z(PEPI[30], 1), r"$ auf $", Z(PEPI[5], 1), r"\,\%$ (Tabelle")
    pr.tabelle(K7, "tab:bayes_grenze", r"\d+",
               {s: [r4(BAYES[s][0]), r4(M("MLP", s)), r4(M("SVM", s)), r4(M("RF", s)), r4(BAYES[s][1]),
                    r4(M("DDQN", s))] for s in snrs})
    pr.bed(all(abs(M(m, s) - BAYES[s][0]) < 0.01 for m in ("MLP", "SVM") for s in snrs)
           and all(abs(M("DDQN", s) - BAYES[s][1]) < 0.01 for s in snrs),
           "Kap. 7/8: DDQN, MLP und SVM liegen innerhalb eines Prozentpunkts der Bayes-Referenz")
    rf_gap = [100 * (BAYES[s][0] - M("RF", s)) for s in snrs]
    rf_gap_druck = [100 * (float(r4(BAYES[s][0])) - M("RF", s)) for s in snrs]
    gap_lo = Z(min(rf_gap), 1, leser=min(rf_gap_druck))
    gap_hi = Z(max(rf_gap), 1, leser=max(rf_gap_druck))
    pr.text(K7, r"der Random Forest $", gap_lo, r"$ bis $", gap_hi, "$ Prozentpunkte darunter")
    pr.text(K7, r"und $", Z(cfg.eval_flows, 0), "$ Testflüsse je Stufe")

    pr.abschnitt("Kapitel 7.4/7.5: Einordnung und Zusammenfassung")
    pr.text(K7, r"nur um etwa $", Z(winkel_grad(5), 0), r"^\circ$ ($", Z(winkel_grad(20), 1), r"^\circ$ bei $20$\,dB, $",
            Z(winkel_grad(10), 1), r"^\circ$ bei $10$\,dB)")
    pr.text(K7, r"impliziten Schwelle von $", Z(T["RF"][30], 2), r"$ bei 30\,dB")
    pr.text(K7, r"dieselbe Struktur $", Z(N_FEATURES, 0), r"\to", Z(cfg.hidden, 0), r"\to", Z(cfg.hidden, 0), "$")
    t_dqmin, t_dqmax = min(T["DDQN"].values()), max(T["DDQN"].values())
    pr.text(K7, r"gemessen entscheidet es bei $t=", Z(t_dqmin, 2), r"$ bis $", Z(t_dqmax, 2), "$")
    pr.text(K7, r"nur den Anteil $1/", Z(N_FEATURES, 0), "$ seiner Leistung")
    pr.text(K7, r"$\sqrt{", Z(N_FEATURES, 0), r"}\approx", Z(math.sqrt(N_FEATURES), 1), "$")
    pr.text(K7, r"F1-Score von $", Z(M("DDQN", 30), 4), r"$ und hält bei 10\,dB noch $", Z(M("DDQN", 10), 4),
            r"$. Bei 5\,dB fällt er auf $", Z(M("DDQN", 5), 4), "$")
    pr.text(K7, r"implizite Schwelle des Agenten bei $", Z(t_dqmin, 2), r"$ bis $", Z(t_dqmax, 2), "$")
    pr.bed(max(d_mlp.values()) < 1, "Kap. 7: Abstand DDQN-MLP unter einem Prozentpunkt")
    pr.text(K7, r"($", Z(ELA["DDQN"][5], 3), r"$ bei 5\,dB gegenüber $", Z(ELA["RF"][5], 3), "$ beim Random Forest)")
    pr.text(K7, r"fällt um $", gap_lo, r"$ bis $", gap_hi, "$ Prozentpunkte ab")
    pr.text(K7, r"Bis 10\,dB verliert der DDQN $", Z(RLA["DDQN"][10], 2), r"\,\%$ seiner Ausgangsleistung, bei 5\,dB $",
            Z(RLA["DDQN"][5], 2), r"\,\%$")

    # ---------------------------------------------------------------- Kapitel 8
    pr.abschnitt("Kapitel 8: Fazit und Ausblick")
    pr.text(K8, "Reward-Matrix " + reward_satz)
    t_mmin, t_mmax = min(T["MLP"].values()), max(T["MLP"].values())
    pr.text(K8, r"bei $t=", Z(t_dqmin, 2), r"$ bis $", Z(t_dqmax, 2), r"$, die des MLP bei $", Z(t_mmin, 2),
            r"$ bis $", Z(t_mmax, 2), "$")
    pr.text(K8, r"bei 30\,dB mit $", Z(M("DDQN", 30, "recall"), 4), r"$ gegenüber $", Z(M("DDQN", 30, "precision"), 4),
            "$")
    pr.text(K8, r"fällt monoton von $", Z(M("DDQN", 30), 4), r"$ bei 30\,dB über $", Z(M("DDQN", 10), 4),
            r"$ bei 10\,dB auf $", Z(M("DDQN", 5), 4), r"$ bei 5\,dB")
    pr.text(K8, r"zwischen $", Z(min(r5), 2), r"$ und $", Z(max(r5), 2), r"\,\%$ ihres F1-Scores")
    pr.text(K8, r"von $", Z(PEPI[30], 1), r"$ auf $", Z(PEPI[5], 1), r"\,\%$ steigt")
    pr.text(K8, r"der Random Forest $", gap_lo, r"$ bis $", gap_hi, "$ Prozentpunkte darunter")
    pr.bed(M("DDQN", 30, "precision") - M("DDQN", 5, "precision") > M("DDQN", 30, "recall") - M("DDQN", 5, "recall"),
           "Kap. 8: Precision sinkt stärker als der Recall")
    pr.text(K8, r"beträgt aber nur $", Z(min(d_mlp.values()), 2), r"$ bis $", Z(max(d_mlp.values()), 2),
            "$ Prozentpunkte")
    pr.text(K8, r"um $", Z(min(rw_diff), 3), r"$ bis $", Z(max(rw_diff), 3), r"$ über dem des MLP, bei "
            r"Seed-Streuungen bis $", Z(rw_std, 3), "$")
    pr.text(K8, r"Der DDQN sah $", Z(n_dqn / 10 ** e_dqn, 1), r"\cdot10^{", Z(e_dqn, 0), r"}$ Trainingsflüsse, "
            r"die Baselines $", Z(N_TRAIN / 10 ** e_base, 0), r"\cdot10^{", Z(e_base, 0), "}$")
    pr.text(K8, r"(implizite Schwelle $", Z(t_dqmin, 2), r"$ bis $", Z(t_dqmax, 2), "$ statt $1/3$)")
    pr.text(K8, r"rund $", Z(100 * IOT_DH_TEST[0] / IOT_DH_TEST[1], 0), r"\,\%$ Angriffsanteil")
    pr.text(K8, r"Huber-Parameter $\beta\ge", Z(max(R_TP - R_FP, R_TN - R_FN), 1), "$")
    pr.text(K8, r"mit $", Z(n_param, 0), "$ Parametern gegen Random Forest (", Z(rf0.n_estimators, 0), " Bäume)")

    # ---------------------------------------------------------------- Anhang A
    pr.abschnitt("Anhang A: Text und skriptgenerierte Tabellen")
    if KA in texte:
        pr.text(KA, r"je $", Z(cfg.eval_flows, 0), "$ Testflüsse")
        pr.text(KA, "alle ", Z(len(SEEDS) * len(snrs) * len(METRIKEN), 0), " DDQN-Werte")
        import anhang_tabellen
        for name, inhalt in anhang_tabellen.erzeuge(daten).items():
            datei = HERE / "anhang" / name
            pr.bed(datei.exists() and datei.read_text(encoding="utf-8") == inhalt,
                   f"anhang/{name} entspricht der Ausgabe von anhang_tabellen.py")
    else:
        pr.bed(False, "kapitel/anhang_a.tex fehlt")

    print("\nNur verwiesen, hier nicht nachgerechnet:")
    print("  Tabelle 7.3 (mittlerer Reward): reward_tabelle.py prüft P/R/F1 aller Modelle gegen beide CSV-Dateien.")
    print("  MSE-Kontrolllauf (Nulldurchgang 0,30 bis 0,32): kontrolle_mse_verlust.py.")
    print("  Literaturwerte (Kap. 2, 3, 7.4): gegen die Quell-PDFs geprüft.")
    print(f"\nErgebnis: {pr.n_ok} Prüfungen bestanden, {len(pr.fehler)} Abweichungen.")
    for f in pr.fehler:
        print(f"  ABWEICHUNG: {f}")
    return 1 if pr.fehler else 0


if __name__ == "__main__":
    sys.exit(main())
