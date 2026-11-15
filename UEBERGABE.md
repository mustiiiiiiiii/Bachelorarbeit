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
  Henderson und Elkan ohne Seitenzahlen, weil nur Preprints vorliegen.
- **Zotero-Export:** Ein roher Zotero-Export der bib (fehlende Keys, Dubletten mit `a`-Suffix, unvollständige Felder)
  wurde verworfen und liegt unversioniert als `literatur_zotero.bib` vor. Maßgeblich ist die geprüfte `literatur.bib`.
- **Volltextabgleich:** Arjoune 2020 nutzt Linkqualitätsmerkmale (BPR, PDR, RSS, CCA), **kein SNR-Merkmal**.
  Caminero 2019 (AE-RL) besteht aus zwei DQN-Agenten. Merzouk-, Zhang- und Apruzzese-Zahlen stammen aus den Volltexten.
- **Ran 2024:** Bd. 20, H. 8, S. 10405–10418 nur über zitierende Arbeiten bestätigt; das PDF ist das akzeptierte
  Manuskript. Einmal auf IEEE Xplore (Dok. 10528868) gegenprüfen.
- **Konsistenz:** Forschungsfragen FF1–FF3 (Kap. 1) passen zum synthetischen Gauß-Flussmodell; Fazit (Kap. 8)
  beantwortet FF1–FF3 entlang Kap. 7. Kap. 4 beschreibt nur die tatsächlich ausgeführte Vorverarbeitung.
- **Abgleich Kap. 5–7 mit Code** (`train_eval.py`, `eval_baselines.py`, CSVs): alle Hyperparameter und Messwerte geprüft.

## Nachgerechnete Größen (nicht aus neuen Trainingsläufen)

| Größe | Quelle / Methode | Stelle |
|---|---|---|
| Bayes-Fehler bei π = 0,3: 6,2 / 6,6 / 10,4 / 16,2 % | analytisch aus Generatormodell | Tab. `tab:klassenueberlappung` |
| Bayes-optimaler F1 je Schwelle | analytisch (LLR ~ N(±D²/2, D²)) | Tab. `tab:bayes_grenze` |
| Mittlerer Reward je Fluss | aus P/R je Lauf + rekonstruierter Zahl der Angriffsflüsse (deterministische Test-RNGs) | Kap. 7, `sec:baseline_vergleich` |
| RLA / ELA | aus mittleren F1-Werten der CSVs | Tab. `tab:rla_ela` |
| 19 969 Lernschritte, 79 Zielnetz-Syncs | aus `Config` in `train_eval.py` | Kap. 5, `sec:training_optimierung` |

## Build

- `pdflatex → biber → pdflatex → pdflatex`, Ergebnis zuletzt: 48 Seiten, 0 Fehler, 0 Warnungen (LaTeX und biber), 0 Overfull, 0 Underfull.
- `main.tex` enthält Spaltentyp `L{…}` (linksbündig, Paket `array`), URL-Umbrüche für biblatex und `\emergencystretch`.
- `main.bcf` und `main.run.xml` sind nicht mehr versioniert (werden beim Kompilieren erzeugt). `main.pdf` ist noch versioniert.

## Offen

1. Ran 2024 Metadaten auf IEEE Xplore bestätigen.
2. Optional: Seitenzahlen für Henderson 2018 (AAAI) und Elkan 2001 (IJCAI) aus den Verlagsfassungen nachtragen.
3. Optional: `main.pdf` ebenfalls aus Git nehmen, wenn nur noch lokal kompiliert wird.

## Arbeitsweise

- Nur an einem Ort arbeiten (lokal **oder** Cloud), vorher `git pull`.
- Nach jeder Änderung vollständig kompilieren und das Log auf Fehler, undefinierte Verweise sowie Over-/Underfull prüfen.
