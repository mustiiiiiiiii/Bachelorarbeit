"""Tabellen für Anhang A, direkt aus den CSV-Dateien erzeugt (nicht von Hand ändern).

anhang/tab_prf1.tex       Tab. A.1  Precision, Recall und F1-Score aller Verfahren je SNR-Stufe (Mittelwert ± Std)
anhang/tab_seeds.tex      Tab. A.2  Einzelwerte je Seed
anhang/tab_schwellen.tex  Tab. A.3  implizite Schwellen (Bayes-Schwelle mit gleichem Recall), dreistellig
anhang/tab_zuordnung.tex  Tab. A.4  Zuordnung jeder Abbildung, Tabelle und des Algorithmus zu Skript und Daten

Vor dem Schreiben wird geprüft, dass jedes fig:/tab:/alg:-Label der Kapitel 1 bis 8 genau einmal in Tab. A.4
vorkommt, die Anhang-Labels in kapitel/anhang_a.tex definiert sind und alle genannten Skripte und Dateien existieren.
Aufruf: python anhang_tabellen.py
"""

import re
import sys

from nachweis_zahlen import HERE, METRIKEN, MODELLE, NAMEN, implizite_schwelle, lade_csv, rund, tex
from train_eval import SEEDS, SNR_LEVELS_DB

KOPF = "% erzeugt von anhang_tabellen.py, nicht von Hand ändern"
KURZ = {"DDQN": "DDQN", "RF": "RF", "SVM": "SVM", "MLP": "MLP"}


def zahl(x, stellen: int = 4) -> str:
    return tex(rund(x, stellen))


def tt(name: str) -> str:
    """Datei- oder Skriptname in Schreibmaschinenschrift (die Spaltenbreite in Tab. A.4 fasst jeden Namen)."""
    return r"\texttt{" + name.replace("_", "\\_") + "}"


def tab_prf1(daten: dict) -> str:
    z = [KOPF, r"\begin{tabular}{llccc}", r"\toprule",
         r"\textbf{Verfahren} & \textbf{SNR in dB} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} \\"]
    for m in MODELLE:
        z.append(r"\midrule")
        for j, s in enumerate(SNR_LEVELS_DB):
            zellen = [f"${zahl(daten[m][('mean', s)][k])} \\pm {zahl(daten[m][('std', s)][k])}$" for k in METRIKEN]
            z.append((NAMEN[m] if j == 0 else "") + f" & {s} & " + " & ".join(zellen) + r" \\")
    return "\n".join(z + [r"\bottomrule", r"\end{tabular}", ""])


def tab_seeds(daten: dict) -> str:
    z = [KOPF, r"\begin{tabular}{@{}lc*{9}{c}@{}}", r"\toprule",
         " & & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{\textbf{{Seed {s}}}}}" for s in SEEDS) + r" \\",
         "".join(rf"\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(len(SEEDS))),
         r"\textbf{Verfahren} & \textbf{SNR} & " + " & ".join(["P & R & F1"] * len(SEEDS)) + r" \\"]
    for m in MODELLE:
        z.append(r"\midrule")
        for j, s in enumerate(SNR_LEVELS_DB):
            zellen = [f"${zahl(daten[m][(str(seed), s)][k])}$" for seed in SEEDS for k in METRIKEN]
            z.append((KURZ[m] if j == 0 else "") + f" & {s} & " + " & ".join(zellen) + r" \\")
    return "\n".join(z + [r"\bottomrule", r"\end{tabular}", ""])


def tab_schwellen(daten: dict) -> str:
    z = [KOPF, r"\begin{tabular}{lccccc}", r"\toprule",
         r" & \multicolumn{4}{c}{\textbf{Implizite Schwelle $t$}} & \\", r"\cmidrule(lr){2-5}",
         r"\textbf{Verfahren} & " + " & ".join(f"{s}\\,dB" for s in SNR_LEVELS_DB) + r" & \textbf{Soll-Schwelle} \\",
         r"\midrule"]
    for m in MODELLE:
        t = [implizite_schwelle(s, daten[m][("mean", s)]["recall"]) for s in SNR_LEVELS_DB]
        soll = "$1/3$" if m == "DDQN" else "$1/2$"
        z.append(NAMEN[m] + " & " + " & ".join(f"${zahl(v, 3)}$" for v in t) + f" & {soll} " + r"\\")
    return "\n".join(z + [r"\bottomrule", r"\end{tabular}", ""])


def zuordnung():
    """-> (Zeilen, genannte Dateien). Zeile: (Labels, Element-Text oder None, Inhalt, Skript, Daten)."""
    dateien = []

    def d(name: str) -> str:
        dateien.append(name)
        return tt(name)

    beide = "beide CSV-Dateien"
    zeilen = [
        (("fig:paketstrom", "fig:generator_ablauf", "fig:blockschaltbild", "fig:qnet_architektur", "fig:backup",
          "fig:eval_protokoll"), None, "Schemata, als TikZ im Quelltext gesetzt", "--", "keine Messdaten"),
        (("tab:rw_robustheit", "tab:ki_vergleich"), None, "Ergebnisse der Literatur", "--",
         "Quellen im Literaturverzeichnis"),
        (("fig:feature_raum_3d",), None, r"Flüsse im Teilraum $(y_1,y_2,y_4)$", d("create_3d_plot.py"),
         "Generator in " + d("train_eval.py")),
        (("tab:klassenueberlappung",), None, r"Abstand $D$ und Bayes-Fehler je SNR-Stufe",
         d("nachweis_zahlen.py") + " (Prüfung)", d("bayes_referenz.py")),
        (("fig:llr_signalraum",), None, r"Klassendichten der Prüfgröße $\Lambda$", d("llr_signalraum_plot.py"),
         d("bayes_referenz.py")),
        (("alg:training", "tab:dqn_hyperparameters", "tab:system_parameters"), None,
         "Trainingsschleife und Konfiguration", d("train_eval.py") + ", " + d("nachweis_zahlen.py") + " (Prüfung)",
         r"Klasse \texttt{Config}"),
        (("tab:ddqn_snr", "fig:snr_robustheit"), None, "DDQN je SNR-Stufe",
         d("train_eval.py") + ", " + d("generate_plot.py"), d("evaluation_results.csv")),
        (("fig:entscheidungsfunktion",), None, "Entscheidungsfunktion gegenüber der Bayes-Posterior",
         d("entscheidungsfunktion_plot.py"), d("modelle/") + ", " + d("baseline_results.csv")),
        (("tab:f1_vergleich", "fig:baseline_vergleich"), None, "F1-Score aller Verfahren",
         d("eval_baselines.py") + ", " + d("generate_plot.py"), beide),
        (("tab:reward",), None, "mittlerer Reward je Fluss", d("reward_tabelle.py"), d("modelle/") + ", " + beide),
        (("tab:rla_ela",), None, "RLA und ELA", d("nachweis_zahlen.py") + " (Prüfung)",
         r"Tabelle~\ref{tab:f1_vergleich}"),
        (("tab:bayes_grenze", "fig:pr_ebene"), None, "Bayes-Referenz und gemessene Arbeitspunkte",
         d("pr_ebene_plot.py"), d("bayes_referenz.py") + ", " + beide),
        (("fig:entscheidungsgrenzen",), None, "Schnitt durch den Merkmalsraum", d("entscheidungsgrenzen_plot.py"),
         d("modelle/ddqn_seed0.pt") + ", " + d("baseline_results.csv")),
        (("tab:anh_prf1", "tab:anh_seeds", "tab:anh_schwellen", "tab:anh_zuordnung"), None,
         "Messwerte, Schwellen und Zuordnung", d("anhang_tabellen.py"), beide),
        ((), "Fließtext aller Kapitel", "abgeleitete Zahlen in Text und Tabellen", d("nachweis_zahlen.py"),
         beide + ", Code, " + d("modelle/")),
        ((), r"Abschnitt~\ref{sec:reward_recall}", "Kontrolllauf mit quadratischem Verlust",
         d("kontrolle_mse_verlust.py"), r"eigener Lauf mit Seed~0"),
        ((), r"Gewichte in \texttt{modelle/}", "DDQN-Agenten der Seeds 0, 1 und 2",
         d("modelle_nachtrainieren.py"), "reproduziert " + d("evaluation_results.csv")),
    ]
    return zeilen, dateien


def element(labels) -> str:
    """Labels -> 'Abbildungen~\\ref{a}, \\ref{b} und~\\ref{c}, Tabelle~\\ref{d}', Gruppen in Reihenfolge."""
    namen = {"fig": ("Abbildung", "Abbildungen"), "tab": ("Tabelle", "Tabellen"), "alg": ("Algorithmus", "Algorithmen")}
    gruppen = {}
    for lab in labels:
        gruppen.setdefault(lab.split(":")[0], []).append(lab)
    teile = []
    for art, labs in gruppen.items():
        refs = [rf"\ref{{{x}}}" for x in labs]
        liste = refs[0] if len(refs) == 1 else ", ".join(refs[:-1]) + " und~" + refs[-1]
        teile.append(namen[art][len(refs) > 1] + "~" + liste)
    return ", ".join(teile)


def pruefe_zuordnung(zeilen, dateien) -> list:
    probleme = []
    haupt = []
    for p in sorted((HERE / "kapitel").glob("0[1-8]_*.tex")):
        haupt += re.findall(r"\\label\{((?:fig|tab|alg):[^}]+)\}", p.read_text(encoding="utf-8"))
    genannt = [lab for z in zeilen for lab in z[0]]
    for lab in haupt:
        if genannt.count(lab) != 1:
            probleme.append(f"{lab} kommt {genannt.count(lab)}-mal in Tab. A.4 vor")
    anhang = HERE / "kapitel" / "anhang_a.tex"
    anhang_labels = re.findall(r"\\label\{([^}]+)\}", anhang.read_text(encoding="utf-8")) if anhang.exists() else []
    for lab in genannt:
        if lab not in haupt and lab not in anhang_labels:
            probleme.append(f"{lab} ist weder im Hauptteil noch in kapitel/anhang_a.tex definiert")
    for name in dict.fromkeys(dateien):
        if not (HERE / name).exists():
            probleme.append(f"{name} existiert nicht")
    return probleme


def tab_zuordnung() -> str:
    zeilen, dateien = zuordnung()
    probleme = pruefe_zuordnung(zeilen, dateien)
    if probleme:
        raise ValueError("Tab. A.4 unvollständig: " + "; ".join(probleme))
    z = [KOPF, r"\begin{tabular}{@{}L{2.8cm}L{4.6cm}L{7.4cm}@{}}", r"\toprule",
         r"\textbf{Element} & \textbf{Inhalt} & \textbf{Skript und Datengrundlage} \\", r"\midrule"]
    for labels, text, inhalt, skript, quelle in zeilen:
        dritte = quelle if skript == "--" else skript + r"\newline Daten: " + quelle
        z.append(f"{text or element(labels)} & {inhalt} & {dritte} " + r"\\")
    return "\n".join(z + [r"\bottomrule", r"\end{tabular}", ""])


def erzeuge(daten=None) -> dict:
    daten = daten or lade_csv()
    return {"tab_prf1.tex": tab_prf1(daten), "tab_seeds.tex": tab_seeds(daten),
            "tab_schwellen.tex": tab_schwellen(daten), "tab_zuordnung.tex": tab_zuordnung()}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ziel = HERE / "anhang"
    ziel.mkdir(exist_ok=True)
    try:
        tabellen = erzeuge()
    except ValueError as e:
        print(e)
        return 1
    for name, inhalt in tabellen.items():
        (ziel / name).write_text(inhalt, encoding="utf-8")
        print(f"Gespeichert: anhang/{name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
