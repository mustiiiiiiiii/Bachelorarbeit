# Übergabe: Stand der Bachelorarbeit

Stand: Commit nach `374a34a` (Branch `claude/dreamy-gauss-muiook`, `main` per Fast-Forward identisch).
Diese Notiz fasst zusammen, was in der Cloud-Sitzung erledigt wurde, damit eine neue Sitzung
(z. B. Claude Code in VS Code) ohne den Gesprächsverlauf weiterarbeiten kann.
Stilregeln für alle Texte: `CLAUDE.md`.

## Erledigt

- **Literatur:** `literatur.bib` hat 21 Einträge, alle im Text zitiert. 14 davon wurden neu ergänzt
  und gegen die PDFs in `QuellenLiteratur/` geprüft. Al-Gethami 2021 wurde bewusst aussortiert
  (Label- statt Merkmalsrauschen, Hindawi).
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

- `pdflatex → biber → pdflatex → pdflatex`, Ergebnis zuletzt: 47 Seiten, 0 Fehler, 0 Overfull, 0 Underfull.
- `main.tex` enthält Spaltentyp `L{…}` (linksbündig, Paket `array`), URL-Umbrüche für biblatex und `\emergencystretch`.
- `main.bcf` und `main.run.xml` sind nicht mehr versioniert (werden beim Kompilieren erzeugt). `main.pdf` ist noch versioniert.

## Offen

1. **Methodische Primärquellen ergänzen** (Vorschlag, noch nicht umgesetzt; DOIs vorher prüfen):
   Sutton & Barto 2018 (MDP, Kap. 5), Breiman 2001 (Random Forest), Cortes & Vapnik 1995 (SVM),
   Elkan 2001 (Kostenschwelle p > 1/3, Kap. 5/7), Kingma & Ba 2015 (Adam), Chawla et al. 2002 (SMOTE, Kap. 2),
   Henderson et al. 2018 (Seed-Varianz in Deep RL, Kap. 7), Pedregosa et al. 2011 (scikit-learn).
2. Ran 2024 Metadaten auf IEEE Xplore bestätigen.
3. Optional: `main.pdf` ebenfalls aus Git nehmen, wenn nur noch lokal kompiliert wird.

## Arbeitsweise

- Nur an einem Ort arbeiten (lokal **oder** Cloud), vorher `git pull`.
- Nach jeder Änderung vollständig kompilieren und das Log auf Fehler, undefinierte Verweise sowie Over-/Underfull prüfen.
