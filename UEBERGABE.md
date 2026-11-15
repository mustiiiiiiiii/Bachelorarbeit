# Übergabe: Stand der Bachelorarbeit

Stand: 2026-11-15, Branch `main` (lokale Sitzung in VS Code nach der Cloud-Sitzung bis `374a34a`).
Diese Notiz fasst den Arbeitsstand zusammen, damit eine neue Sitzung
(z. B. Claude Code in VS Code) ohne den Gesprächsverlauf weiterarbeiten kann.
Stilregeln für alle Texte: `CLAUDE.md`.

## Erledigt

- **Literatur:** `literatur.bib` hat 29 Einträge, alle im Text zitiert und gegen die PDFs in `QuellenLiteratur/` geprüft. Al-Gethami 2021 wurde bewusst aussortiert
  (Label- statt Merkmalsrauschen, Hindawi).
- **Methodische Primärquellen (lokale Sitzung):** Sutton & Barto 2018 (MDP-Notation, Kap. 5), Breiman 2001 (Kap. 3),
  Cortes & Vapnik 1995 (Kap. 3), Elkan 2001 (Schwelle p* nach Gl. (2) = 1/3, Kap. 5 und 7), Kingma & Ba 2015 (Adam, Kap. 5),
  Chawla et al. 2002 (SMOTE, Kap. 2), Henderson et al. 2018 (Seed-Varianz, Kap. 7), Pedregosa et al. 2011 (scikit-learn, Kap. 6).
  DOIs über Crossref bestätigt. Ohne DOI: Sutton (Buch, ISBN), Kingma (arXiv), Pedregosa (JMLR), Elkan (IJCAI).
  Elkan S. 973–977 laut ijcai.org (Proceedings 2001). Henderson ohne Seiten: Crossref/OpenAlex führen keine,
  AAAI und DBLP blocken automatisierte Abrufe.
- **Zotero-Export:** Ein roher Zotero-Export der bib (fehlende Keys, Dubletten mit `a`-Suffix, unvollständige Felder)
  wurde verworfen. Maßgeblich ist allein die geprüfte `literatur.bib`, Zotero dient nur als PDF-Ablage.
  `literatur_zotero*.bib` steht in der `.gitignore`. Künftige Exporte nur mit Better BibTeX und festen Citation Keys.
- **Repo bereinigt:** Testdateien (`mwe*.tex`, `test_dup*.tex`), `buildlog.txt`, ein toter Discord-Link (`.url`) und ein
  alter KI-Kontext-Prompt entfernt. Die Aufgabenstellung von Prof. Seitz heißt jetzt `Aufgabenstellung_Bachelorarbeit.pdf`.
- **Volltextabgleich:** Arjoune 2020 nutzt Linkqualitätsmerkmale (BPR, PDR, RSS, CCA), **kein SNR-Merkmal**.
  Caminero 2019 (AE-RL) besteht aus zwei DQN-Agenten. Merzouk-, Zhang- und Apruzzese-Zahlen stammen aus den Volltexten.
- **Ran 2024:** Bd. 20, H. 8, S. 10405–10418, Aug. 2024, über Crossref (DOI 10.1109/TII.2024.3396527) bestätigt;
  das PDF ist das akzeptierte Manuskript.
- **Konsistenz:** Forschungsfragen FF1–FF3 (Kap. 1) passen zum synthetischen Gauß-Flussmodell; Fazit (Kap. 8)
  beantwortet FF1–FF3 entlang Kap. 7. FF1 unterscheidet Soll-Schwelle 1/3 (Elkan) und implizite Schwelle 0,38–0,43. Kap. 4 beschreibt nur die tatsächlich ausgeführte Vorverarbeitung.
- **Abgleich Kap. 5–7 mit Code** (`train_eval.py`, `eval_baselines.py`, CSVs): alle Hyperparameter und Messwerte geprüft.

## Nachgerechnete Größen (nicht aus neuen Trainingsläufen)

| Größe | Quelle / Methode | Stelle |
|---|---|---|
| Bayes-Fehler bei π = 0,3: 6,2 / 6,6 / 10,4 / 16,2 % | analytisch aus Generatormodell | Tab. `tab:klassenueberlappung` |
| Bayes-optimaler F1 je Schwelle | analytisch (LLR ~ N(±D²/2, D²)) | Tab. `tab:bayes_grenze` |
| Mittlerer Reward je Fluss | aus P/R je Lauf + rekonstruierter Zahl der Angriffsflüsse (deterministische Test-RNGs) | Kap. 7, `sec:baseline_vergleich` |
| RLA / ELA | aus mittleren F1-Werten der CSVs | Tab. `tab:rla_ela` |
| 19 969 Lernschritte, 79 Zielnetz-Syncs | aus `Config` in `train_eval.py` | Kap. 5, `sec:training_optimierung` |
| Bayes-PR-Kurven, Bayes-Punkte p > 1/2 und p > 1/3 | geschlossen aus LLR ~ N(±D²/2, D²), `pr_ebene_plot.py` | Abb. `fig:pr_ebene` (Kap. 7) |
| Implizite Schwelle t (Bayes-Schwelle mit gleichem Recall): DDQN 0,43 / 0,42 / 0,38 / 0,38; RF 0,68–0,50; SVM 0,61–0,47; MLP 0,54–0,41 | Nullstellensuche auf der Bayes-Recall-Kurve, mittlere P/R aus den CSVs | Kap. 7, `sec:reward_recall` |

## Abbildungen

- `pr_ebene_plot.py` → `pr_ebene_plot.pdf` (Abb. 7.3): Precision-Recall-Ebene je SNR mit Bayes-Grenze, Bayes-Arbeitspunkten,
  Elkan-Pfeil und Modellpunkten (Mittelwert ± Std). Importiert die Generatorparameter aus `train_eval.py`, Stil wie `generate_plot.py`.
  Bayes-F1 aus dem Skript stimmt auf vier Stellen mit Tab. `tab:bayes_grenze` überein.
- **ELA-Höhenlinien (nach Sáez et al. 2016, Fig. 1): geprüft und verworfen, weil A₀ der Modelle zu eng beieinanderliegt.**
  Die A₀-Werte (F1 bei 30 dB) liegen nur 0,02 auseinander (0,874–0,894). Die Trajektorien werden dadurch zu senkrechten Strichen,
  und die ELA-Isolinien verlaufen fast waagrecht. Die ELA ordnet damit praktisch nach dem F1-Score der gestörten Stufe,
  was Abb. 7.2 schon zeigt. Die Erkenntnis steht stattdessen als Satz nach Tab. `tab:rla_ela` in Kap. 7.

## Build

- `pdflatex → biber → pdflatex → pdflatex`, Ergebnis zuletzt: 50 Seiten, 0 Fehler, 0 Warnungen (LaTeX und biber), 0 Overfull, 0 Underfull.
- `main.tex` enthält Spaltentyp `L{…}` (linksbündig, Paket `array`), URL-Umbrüche für biblatex und `\emergencystretch`.
- `main.bcf`, `main.run.xml` und `main.pdf` sind nicht versioniert (Build-Artefakte, lokal kompilieren).

## Offen

1. Optional: Seitenzahlen für Henderson 2018 (AAAI-18) im Browser auf ojs.aaai.org oder dblp.org prüfen und nachtragen.

## Arbeitsweise

- Nur an einem Ort arbeiten (lokal **oder** Cloud), vorher `git pull`.
- Nach jeder Änderung vollständig kompilieren und das Log auf Fehler, undefinierte Verweise sowie Over-/Underfull prüfen.
