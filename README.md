# Begleitarchiv zur Bachelorarbeit

**Entdeckung von Netzwerkangriffen durch Künstliche Intelligenz**
Mustafa Al-Obaidi, TU Ilmenau, Fachgebiet Kommunikationsnetze (Prof. Dr. Jochen Seitz)

Das Archiv enthält alle Skripte, Messdaten und Modellgewichte, aus denen die Ergebnisse der Arbeit entstehen, sowie den
LaTeX-Quelltext. Welche Abbildung und Tabelle von welchem Skript stammt, steht in Tabelle A.4 der Arbeit.

## Inhalt

| Pfad | Inhalt |
|---|---|
| `Bachelorarbeit.pdf` | die Arbeit |
| `train_eval.py` | Generatormodell, AWGN-Kanal, DDQN-Training und Evaluation (Seeds 0, 1, 2) |
| `eval_baselines.py` | Random Forest, SVM und MLP auf denselben Testflüssen |
| `bayes_referenz.py` | analytische Bayes-Referenz (Mahalanobis-Abstand, Schwellen, Posterior) |
| `modelle_nachtrainieren.py` | trainiert die DDQN-Agenten exakt nach und speichert die Gewichte |
| `*_plot.py`, `plot_stil.py` | Abbildungen der Arbeit |
| `reward_tabelle.py`, `anhang_tabellen.py` | skriptgenerierte Tabellen (Tab. 7.3, A.1–A.4) |
| `nachweis_zahlen.py` | prüft alle abgeleiteten Zahlen in Text und Tabellen gegen Daten, Code und Modelle |
| `kontrolle_mse_verlust.py` | Kontrolllauf mit quadratischem Verlust (Abschnitt 7.1.1) |
| `evaluation_results.csv`, `baseline_results.csv` | Messdaten (Precision, Recall, F1 je Seed und SNR) |
| `modelle/ddqn_seed{0,1,2}.pt` | Gewichte der drei DDQN-Agenten |
| `main.tex`, `kapitel/`, `anhang/`, `verzeichnisse.tex`, `literatur.bib`, `*.pdf`-Abbildungen, `reward_tabelle.tex` | LaTeX-Quelltext |
| `requirements.txt` | Python-Pakete mit den verwendeten Versionen |

## Schnellprüfung (ohne Training, ca. 30 s)

```
python -m pip install -r requirements.txt
python nachweis_zahlen.py
```

Das Skript rechnet jede abgeleitete Zahl der Arbeit nach (203 Prüfungen) und endet mit Exit-Code 0, wenn alle Werte
mit dem LaTeX-Quelltext übereinstimmen. `--ohne-modelle` überspringt die Prüfungen, die PyTorch-Modelle laden.

## Vollständige Reproduktion

In dieser Reihenfolge, jeweils `python <skript>`:

1. `train_eval.py` – DDQN, schreibt `evaluation_results.csv` (ca. 2 min je Seed auf einer RTX 2060, `--device cpu` möglich)
2. `eval_baselines.py` – schreibt `baseline_results.csv`
3. `modelle_nachtrainieren.py` – schreibt `modelle/`, speichert nur bei exakter Übereinstimmung mit der CSV
4. Plotskripte (`generate_plot.py`, `create_3d_plot.py`, `llr_signalraum_plot.py`, `entscheidungsfunktion_plot.py`,
   `pr_ebene_plot.py`, `entscheidungsgrenzen_plot.py`)
5. `reward_tabelle.py`, `anhang_tabellen.py`
6. `nachweis_zahlen.py`

Verwendete Umgebung: Python 3.14.5, PyTorch 2.11.0 (CUDA 12.8), NVIDIA GeForce RTX 2060. Mit dieser Umgebung werden alle
Werte der CSV-Dateien auf vier Stellen reproduziert. Auf anderer Hardware oder ohne CUDA können einzelne Werte in der
letzten Stelle abweichen. Die Plotskripte setzen Beschriftungen mit LaTeX, dafür muss eine TeX-Distribution installiert sein.

PDF bauen: `pdflatex main`, `biber main`, zweimal `pdflatex main`.

## Nicht enthalten

Die zitierten Quellen-PDFs (urheberrechtlich geschützt, Angaben im Literaturverzeichnis) und die Rohdaten von
DDoS-AT-2022 und IoT-DH. Die Experimente verwenden ein parametrisches Flussmodell (Kapitel 4), keine Rohdaten.
