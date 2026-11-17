# Übergabe: Stand der Bachelorarbeit

Stand: 2026-11-17, Branch `main` (lokale Sitzung in VS Code nach der Cloud-Sitzung bis `374a34a`).
**Status: inhaltlich und technisch endgültig abgeschlossen.** Offen sind nur die manuellen Formalia und zwei Betreuer-Fragen
(Abschnitt „Offen“).
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
  Elkan S. 973–977 laut ijcai.org (Proceedings 2001). Henderson ohne Seiten, weil der Verlag keine führt:
  Die AAAI-Artikelseite, Crossref und OpenAlex nennen nur Bd. 32, H. 1 und die DOI 10.1609/aaai.v32i1.11694.
  Die oft zitierte Angabe „S. 3207–3214“ ist nicht belegt und bewusst nicht eingetragen.
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

## Nachgerechnete Größen

Bis auf die als solche markierten Zeilen ohne neue Trainingsläufe. **Jede abgeleitete Zahl in Text und Tabellen prüft
`nachweis_zahlen.py`** (203 Prüfungen mit Kontextmuster gegen den LaTeX-Quelltext, Exit-Code 1 bei Abweichung, Option
`--ohne-modelle`). Rechenregel: Basis sind die abgedruckten Mittelwerte (CSV-mean-Zeilen = Tab. 7.1/7.2/A.1) bzw. Code und
Modelle, gerundet wird genau einmal kaufmännisch. Steht eine Differenz gerundeter Werte im Text, muss sie sich auch aus den
gedruckten Werten ergeben (Leserprobe).

| Größe | Quelle / Methode | Stelle |
|---|---|---|
| Bayes-Fehler bei π = 0,3: 6,2 / 6,6 / 10,4 / 16,2 % | analytisch aus Generatormodell | Tab. `tab:klassenueberlappung` |
| Bayes-optimaler F1 je Schwelle | analytisch (LLR ~ N(±D²/2, D²)) | Tab. `tab:bayes_grenze` |
| Mittlerer Reward je Fluss, 4 Modelle × 4 SNR-Stufen (30 dB: DDQN 0,509, MLP 0,497, SVM 0,484, RF 0,465) | **reproduzierte Modelle**, direkt aus den Entscheidungen auf den Testflüssen, `reward_tabelle.py` (prüft vorher alle 144 P/R/F1-Werte beider CSVs, erzeugt `reward_tabelle.tex`) | Tab. `tab:reward` (7.3) |
| RLA / ELA | aus den mittleren F1-Werten der Tab. 7.2 (nicht aus Seed-Einzelwerten), `nachweis_zahlen.py` | Tab. `tab:rla_ela` |
| 19 969 Lernschritte, 79 Zielnetz-Syncs | aus `Config` in `train_eval.py` | Kap. 5, `sec:training_optimierung` |
| Bayes-PR-Kurven, Bayes-Punkte p > 1/2 und p > 1/3 | geschlossen aus LLR ~ N(±D²/2, D²), `pr_ebene_plot.py` | Abb. `fig:pr_ebene` (Kap. 7) |
| Implizite Schwelle t (Bayes-Schwelle mit gleichem Recall): DDQN 0,43 / 0,42 / 0,38 / 0,38; RF 0,68–0,50; SVM 0,61–0,47; MLP 0,54–0,41 | Nullstellensuche auf der Bayes-Recall-Kurve, mittlere P/R aus den CSVs; dreistellig in Tab. A.3 (`anhang_tabellen.py`) | Kap. 7, `sec:reward_recall`, Tab. `tab:anh_schwellen` |
| FN/FP-Anteile (bezogen auf alle Flüsse) bei p > 1/2 und p > 1/3: 30 dB 3,5/2,7 % bzw. 2,3/4,4 %; 5 dB 10,2/6,0 % bzw. 6,3/11,5 % | geschlossen auf der LLR, `llr_signalraum_plot.py` (Summe bei 1/2 = P_eᵖ aus Tab. 5.1) | Abschnitt 5.4.1, Abb. `fig:llr_signalraum` |
| Nullstelle der Q-Differenz bei Huber-Verlust (β = 1): 3/7 ≈ 0,43 (MSE: 1/3) | geschlossen, `bayes_referenz.huber_differenz` | Kap. 5, `sec:dqn_architektur` |
| Modellbasierte Nulldurchgänge (im Text dreistellig): DDQN 0,416 (30 dB) / 0,369 (5 dB), MLP 0,536 / 0,400; ohne SNR-Kenntnis erwartet: Huber 0,453 / 0,396, MLP-Schwelle 0,535 / 0,442 | **neue Trainingsläufe, exakt reproduziert** (`modelle_nachtrainieren.py`), Auswertung `entscheidungsfunktion_plot.py` | Abb. `fig:entscheidungsfunktion`, `sec:reward_recall` |
| MSE-Kontrolllauf Seed 0: Nulldurchgang 0,30–0,32; P/R/F1 bei 30 dB 0,8442/0,9286/0,8844 | **neuer Trainingslauf, nur Diagnose** (`kontrolle_mse_verlust.py`), ändert keine CSV | 7.1.1, 7.5, 8 (FF1) |
| Bayes-Posterior entlang der DDQN-Grenze im Schnitt (y₁, y₄): 0,36–0,43 (30 dB), 0,30–0,41 (5 dB) | **reproduzierte Modelle** (Seed 0), Konsolenausgabe von `entscheidungsgrenzen_plot.py` | Abb. `fig:entscheidungsgrenzen` (7.5), 7.4 |

## Abbildungen

- `pr_ebene_plot.py` → `pr_ebene_plot.pdf` (Abb. 7.4): Precision-Recall-Ebene je SNR mit Bayes-Grenze, Bayes-Arbeitspunkten,
  Elkan-Pfeil und Modellpunkten (Mittelwert ± Std). Importiert die Generatorparameter aus `train_eval.py`, Stil wie `generate_plot.py`.
  Bayes-F1 aus dem Skript stimmt auf vier Stellen mit Tab. `tab:bayes_grenze` überein.
- **ELA-Höhenlinien (nach Sáez et al. 2016, Fig. 1): geprüft und verworfen, weil A₀ der Modelle zu eng beieinanderliegt.**
  Die A₀-Werte (F1 bei 30 dB) liegen nur 0,02 auseinander (0,874–0,894). Die Trajektorien werden dadurch zu senkrechten Strichen,
  und die ELA-Isolinien verlaufen fast waagrecht. Die ELA ordnet damit praktisch nach dem F1-Score der gestörten Stufe,
  was der F1-Vergleich (jetzt Abb. 7.3) schon zeigt. Die Erkenntnis steht stattdessen als Satz nach Tab. `tab:rla_ela` in Kap. 7.
- `bayes_referenz.py`: gemeinsame analytische Bayes-Referenz (D², Schwellen τ(t), Bayes-PR-Punkte, LLR, Posterior mit
  und ohne SNR-Kenntnis, Huber-Sollkurve). Genutzt von `pr_ebene_plot.py`, `llr_signalraum_plot.py`,
  `entscheidungsfunktion_plot.py`, `kontrolle_mse_verlust.py`.
- `llr_signalraum_plot.py` → `llr_signalraum_plot.pdf` (Abb. 5.4): Signalraum auf der LLR bei 30/5 dB nach Proakis.
- `entscheidungsfunktion_plot.py` → `entscheidungsfunktion_plot.pdf` (Abb. 7.2): gelernte Q-Differenz bzw. MLP-Wahrscheinlichkeit
  über der Bayes-Posterior. Braucht `modelle/ddqn_seed{0,1,2}.pt` aus `modelle_nachtrainieren.py`. Das MLP wird neu gefittet
  und gegen `baseline_results.csv` geprüft, bei Abweichung bricht das Skript ab.
- TikZ: Abb. 5.3 (Backup-Diagramme, Sutton-Notation) in Kap. 5.3, Abb. 4.1 oben mit AWGN-Kanal als Addierer ⊕ (Proakis-Notation).
- `entscheidungsgrenzen_plot.py` → `entscheidungsgrenzen_plot.pdf` (Abb. 7.5): Schnitt (y₁, y₄) durch den Merkmalsraum mit
  RF-Regionen, Grenzen von DDQN und MLP, Bayes-Grenzen bei 1/2, 1/3 und 3/7 und Normalenvektor **w**. Modelle aus Seed 0, RF und MLP
  werden gegen `baseline_results.csv` geprüft.
- `reward_tabelle.py` → `reward_tabelle.tex` (Tab. 7.3, per `\input` eingebunden, **nicht von Hand ändern**).
- `anhang_tabellen.py` → `anhang/tab_prf1.tex`, `tab_seeds.tex`, `tab_schwellen.tex`, `tab_zuordnung.tex` (Tab. A.1–A.4 in
  `kapitel/anhang_a.tex`, **nicht von Hand ändern**). Bricht ab, wenn ein `fig:`/`tab:`/`alg:`-Label aus Kap. 1–8 in Tab. A.4
  fehlt oder ein genanntes Skript nicht existiert. `nachweis_zahlen.py` prüft zusätzlich, ob die Dateien aktuell sind.
- `nachweis_zahlen.py`: Nachweis aller abgeleiteten Zahlen (siehe „Nachgerechnete Größen“). Nicht nachgerechnet, nur verwiesen:
  Tab. 7.3 (`reward_tabelle.py` prüft sich selbst), MSE-Kontrolllauf, Literaturwerte.
- `plot_stil.py`: gemeinsamer Stil aller Matplotlib-Abbildungen (LaTeX-Satz in Computer Modern wie der Fließtext,
  Dezimalkomma). Genutzt von `generate_plot.py`, `pr_ebene_plot.py`, `llr_signalraum_plot.py`, `entscheidungsfunktion_plot.py`,
  `entscheidungsgrenzen_plot.py`. `create_3d_plot.py` setzt usetex selbst.

## Build

- `pdflatex → biber → pdflatex → pdflatex`, Ergebnis zuletzt: 63 PDF-Seiten (VIII + 56), 0 Fehler, 0 Warnungen (LaTeX und biber),
  0 Overfull, 0 Underfull, keine undefinierten Verweise. PDF-Seite = gedruckte Seite + 9.
- Anhang A über `\appendix` und `\include{kapitel/anhang_a}` nach dem Literaturverzeichnis. `\@pnumwidth` = 2em, damit „VIII“
  im Inhaltsverzeichnis passt. Tab. 5.2/5.3 stehen mit `[!t]` oben auf S. 31 (sonst Gleitseite und fast leere Kapitelendseite).
- `main.tex` enthält Spaltentyp `L{…}` (linksbündig, Paket `array`), URL-Umbrüche für biblatex und `\emergencystretch`.
- `main.bcf`, `main.run.xml` und `main.pdf` sind nicht versioniert (Build-Artefakte, lokal kompilieren).
- Zusatzpakete: `longtable` (Verzeichnisse in `verzeichnisse.tex`, eigene hyperref-Anker über `\theHtable`) und
  `algorithm2e` mit den Optionen `german, ruled, vlined, linesnumbered, algochapter` (Algorithmus 5.1, deutsche Schlüsselwörter
  `\Ein`, `\Aus`, `\Fuer`, `\Wenn`).

## Offen

**Inhaltlich und technisch endgültig abgeschlossen (2026-11-16).** Quellen geprüft, Aufgabenstellung vollständig abgedeckt,
Kap. 7 und Fazit konsistent, alle abgeleiteten Zahlen skriptgeprüft (`nachweis_zahlen.py`: 203 Prüfungen, 0 Abweichungen),
Build sauber (63 Seiten, 0 Fehler, 0 Warnungen, 0 Over-/Underfull).
Offen sind die zwei Betreuer-Fragen (Abschnitt „Offene Fragen an den Betreuer“) und die Formalia unten.
Hinweis: Henderson 2018 bleibt ohne Seitenzahlen (siehe Literatur), das Zitat ist über die DOI vollständig.

### Vor der Abgabe (manuell)

1. **Deckblatt:** In `main.tex` gibt es eine eigene `titlepage` (Z. 91–115). Sie ist gegen die Vorgaben des Fachgebiets
   Kommunikationsnetze bzw. der Fakultät EI der TU Ilmenau zu prüfen (Titel laut `Aufgabenstellung_Bachelorarbeit.pdf`,
   Betreuer, Matrikelnummer, Datum).
2. **Eigenständigkeitserklärung:** Fehlt noch im LaTeX-Quelltext. Wortlaut und Position gemäß gültiger Prüfungsordnung
   der TU Ilmenau einfügen und unterschreiben. Eine ggf. geforderte Erklärung zum Einsatz von KI-Werkzeugen gehört dazu.
3. **Letzter Textdurchgang:** Erledigt am 2026-11-17 (alle .tex-Dateien vollständig im Quelltext gelesen, chktex,
   Musterprüfungen; Befunde siehe „Sprach- und Typografie-Durchgang“). LTeX+ wurde bewusst nicht eingesetzt. Optional vor der
   Abgabe: ein Durchgang mit LanguageTool nur für die Kommasetzung in langen Sätzen, dort prüft ein Werkzeug systematischer.
4. **Kurzfassung/Abstract:** fehlt noch. Prüfen, ob die Prüfungsordnung eine deutsche und/oder englische Kurzfassung verlangt.
5. Danach vollständig kompilieren und das PDF für die Abgabe erzeugen (`main.pdf` ist nicht versioniert).

## Qualitäts- und Visualisierungsphase (ab 2026-11-15)

Vorgehen: Einzelseiten-Check von Seite 1 bis 50 (physische PDF-Seiten), Befunde je Seite hier eintragen, gebündelt beheben,
danach Build-Kontrolle. Neue Abbildungen nur mit nachweisbarem Mehrwert gegenüber Text und Tabelle (Kriterium wie bei
den verworfenen ELA-Höhenlinien), Vorschau vor dem Einbinden.

### Befunde Seiten 1–5 (Deckblatt, Verzeichnisse)

| Seite | Befund | Vorschlag |
|---|---|---|
| alle | **Seitennummerierung:** `\setcounter{page}{2}` in `main.tex` lässt Kap. 1 auf S. 2 beginnen, eine S. 1 gibt es nicht. | Zähler auf 1 setzen |
| alle | **Kapitelanfangsseiten ohne Seitenzahl:** `\clearpairofpagestyles` leert auch den Stil `plain`, also bleiben Verzeichnis- und Kapitelanfänge unnummeriert. | Seitenzahl für `plain` setzen, z. B. `\cfoot*{\pagemark}` oder `\ohead*{\pagemark}` |
| 1 | Deckblatt: Fachgebiet Kommunikationsnetze fehlt (steht in der Aufgabenstellung), kein Uni-Logo, Abgabedatum 14. April 2027 prüfen | gegen Vorgaben des Fachgebiets abgleichen |
| 2–3 | Kapitel 3 heißt englisch „Related Work“, alle anderen Kapitel sind deutsch | z. B. „Stand der Forschung“ |
| 2–3 | Kurzfassung/Abstract, Abkürzungsverzeichnis (DDQN, AWGN, SNR, RLA, ELA, MDP …) und Eigenständigkeitserklärung fehlen | Anforderungen der Prüfungsordnung klären, dann ergänzen |
| 4–5 | Abbildungs- und Tabellenverzeichnis übernehmen die volle Caption (bis 5 Zeilen, inkl. Farblegenden und Wertungen). Nur Abb. 4.1 und 7.3 haben eine Kurzform. | Kurzcaptions `\caption[…]{…}` für alle Abbildungen und Tabellen |
| – | `kapitel/07_evaluation_ergebnisse.tex` beginnt mit einem UTF-8-BOM | entfernen (harmlos, aber unsauber) |

**Behoben:** Die Seitenzählung beginnt bei 1, und Kapitelanfänge sowie Verzeichnisse tragen eine Seitenzahl (`\ohead*{\pagemark}`).
Alle Abbildungen und Tabellen haben Kurzcaptions, jeder Verzeichniseintrag passt jetzt auf eine Zeile. Kap. 3 heißt „Stand der Forschung“.
Das Deckblatt nennt das Fachgebiet Kommunikationsnetze, das BOM ist entfernt.
**Offen:** Logo und Abgabedatum auf dem Deckblatt, Kurzfassung/Abstract, Abkürzungsverzeichnis und Eigenständigkeitserklärung
hängen von den Vorgaben der Prüfungsordnung ab.

### Neue Abbildungen

- **Abb. 2.1 „Vom Paketstrom zum Merkmalsvektor“** (TikZ, Kap. 2.1.1): Bots und Clients → Messpunkt → Paketstrom → Flussaggregation
  → **x** ∈ ℝ⁸ → AWGN → Detektor → Entscheidung. Ein gestrichelter Rahmen markiert die reale Messkette, ein Generatormodell-Knoten zeigt,
  was diese Arbeit ersetzt. Sie verbindet die Grundlagen mit Abb. 4.1 und 5.1.

### Kap. 1 (PDF-Seiten 6–7) – behoben

- DDoS-AT-2022 und IoT-DH heißen jetzt „Referenzdatensätze“ (Kap. 1 und Titel von Abschnitt 4.1), weil die Experimente sie nicht auswerten.
- AWGN, SNR, SVM, MLP, DRL, RLA und ELA werden beim ersten Auftreten in Kap. 1 aufgelöst.
- FF2 spricht jetzt von „kanalbedingter Klassenüberlappung“. FF3 führt RLA und ELA ein, das Hurenkind „cy?“ ist weg.
- „Der Preis“ wurde zu „Die Kehrseite“, „schlägt“ zu „abschneidet“. Der Titel von Kap. 5 in der Übersicht ist angeglichen.
- `\mbox{DDoS-AT-2022}` und `\mbox{IoT-DH}` stehen in allen Kapiteln. Die Kapitelübersicht ist linksbündig gesetzt.
- In `main.tex` sind Hurenkinder und Schusterjungen global unterbunden (`\widowpenalty`, `\clubpenalty`, `\displaywidowpenalty` = 10000).

### Kap. 2 (PDF-Seiten 8–12) – behoben

- Gl. 2.1 schreibt die Dimension aus (σ²ₙ,₈), weil `d` in Kap. 5/8 für das Done-Flag steht. Gl. 2.2 unterscheidet die Dichten p_x und p_y.
- „Beide Werte“ heißt jetzt „Beide Ergebnisse“ (Raza et al. liefern eine Rangfolge, keinen Wert).
- CNN, SDN und k-NN werden beim ersten Auftreten aufgelöst.
- Die Floskel-Einleitung von 2.4 ist gestrichen. Die Überschriften lauten jetzt „2.4 Maschinelles Lernen: Training, Evaluierung und
  Klassenungleichgewicht“ und „2.4.2 Klassenungleichgewicht“.
- Bala (1,4 Tbit/s) und Zhu & Wu (17 Datensätze, C4.5) sind gegen die PDFs geprüft.

### Kap. 3 (PDF-Seiten 13–20) – behoben

- Das Double-Q-Argument in 3.6 ist eingeschränkt: Es greift in der Ein-Schritt-Formulierung nicht (Verweis auf 5.2.1, 5.3 und Kap. 8).
- Die Kostenasymmetrie überwachter Verfahren ist nach Elkan als gleichwertig eingeordnet. Der DDQN bindet sie direkt ins Lernziel ein.
- Die Merzouk-Angabe „sechsseitiges Kurzpapier“ ist entfernt (PDF hat 7 Seiten, kein Short-Paper-Vermerk). Bei der Online-Anpassung
  steht in Text und Tab. 3.2 jetzt „grundsätzlich möglich, hier nicht untersucht“.
- Der einzelne Unterabschnitt 3.4.1 ist aufgelöst. Die Überschriften lauten jetzt „3.2 Klassische Verfahren des maschinellen Lernens“
  und „3.2.2 Support Vector Machines“.
- IDS, DQN, PG, AC, RBF, C-LSTM, A3C, PPO und DDPG werden beim ersten Auftreten aufgelöst. FP/FN stehen als Fußzeile unter Tab. 3.2.
  ML und DL sind in Kap. 1–3 ausgeschrieben bzw. aufgelöst.
- „Der Preis …“ ist ersetzt, 99,97 % im Mathe-Modus gesetzt.

### Kap. 4 (PDF-Seiten 21–25) – behoben

- **Rechenfehler korrigiert:** Für Merkmal 1 bei 5 dB gilt σ_eff = √(0,04 + 0,0613) ≈ 0,32 (vorher 0,34), bei 30 dB σₙ,₁ ≈ 0,014 (vorher 0,015).
  Die TikZ-Kurve in Abb. 4.1 (b) nutzt jetzt Varianz 0,1013 und Amplitude 1,254 (vorher 0,1137/1,18).
- Abschnitt 4.5 nennt die Soll-Schwellen (1/3 bzw. 1/2) und verweist für die gemessenen impliziten Schwellen auf 7.1.1.
- Die Überschriften lauten jetzt „4.3 Merkmalsextraktion“ und „4.5 Umgang mit dem Klassenungleichgewicht“. „Normierte Merkmale“
  ist präzisiert, der verwaiste Verweissatz nach Abb. 4.2 integriert.
- **Abb. 4.2:** `create_3d_plot.py` erzeugt die Grafik jetzt in Textbreite (6,3 × 3,3 in), damit die Schrift in Originalgröße erscheint
  (Achsen 9 pt, Ticks 7,5 pt). Die z-Beschriftung ist nicht mehr abgeschnitten.
- Gegen Code und Saif-PDF geprüft: Mittelwertabstände, P_j-Formel, Test-Seeds, 20 000 Trainingsflüsse, Skalierung, Early Stopping,
  IoT-DH-Angaben (10 000 Einträge, 80/20, 98 %, 748/2000).

### Kap. 5 (PDF-Seiten 26–33) – behoben

- **Abb. 5.1 neu gezeichnet:** Ein Gymnasium-Rahmen umschließt Generator, AWGN und Reward-Matrix. a und c führen in die Reward-Matrix,
  r läuft gestrichelt eindeutig in den Replay-Puffer.
- σ_eff bei 5 dB „0,30 bis 0,37 je Merkmal“ (vorher pauschal 0,32). In 5.2/5.2.1 steht jetzt die Soll-Schwelle 1/3 mit Verweis auf
  die gemessenen 0,38–0,43.
- Der Verweis „Kapitel 3.6“ heißt jetzt „Abschnitt 3.6“. Die Gridworld-Reste (Gitter, Bewegungsaktion) sind gestrichen.
  „Kostenmatrix“ heißt in Kap. 5/7/8 einheitlich „Reward-Matrix“.
- GPU neutral formuliert („standardmäßig CUDA-GPU“), weil nicht belegt ist, auf welchem Gerät die Läufe liefen.
- Alle Hyperparameter und abgeleiteten Zahlen stimmen mit `train_eval.py` überein.

### Kap. 6 (PDF-Seiten 34–36) – behoben

- **Abb. 6.1:** Aus dem gemischten SNR führen zwei parallele Trainingsstränge (DDQN; RF/SVM/MLP), der irreführende Pfeil
  DDQN → Baselines ist weg. Die Überschriften stehen auf gleicher Höhe.
- Bishop-Regularisierung nur für differenzierbare Modelle (Q-Netz, MLP), nicht für den RF, wie in 2.4.4.
- „1 − Precision“ ist jetzt als Anteil der Fehlalarme unter allen Alarmen formuliert (vorher ungenau „Fehlalarmrate“).
- TP, FP, FN, F1 und die Reward-Indizes R_TP … R_FN stehen in der ganzen Arbeit aufrecht. RLA in %, ELA als Verhältniszahl.
- Kurzformen SVM und MLP, Floskeln (Nominal-/Robustheitsbedingung, Referenzrahmen) gestrichen.
- Gegen den Code geprüft: gamma=scale (≈ 1/8), MLP 2 × 128/Adam/200/Early Stopping, `numpy.std` in Populationsform,
  Bayes-Fehler 6,15 % ohne Rauschen und 6,20 % bei 30 dB.

### Kap. 7 (PDF-Seiten 37–44) – behoben

- Die alte Schwellenaussage in 7.2, 7.4 und 7.5 lautet jetzt „Soll-Schwelle“, jeweils mit den gemessenen impliziten Schwellen.
- RF-Absatz: Die Bayes-Normale Σ⁻¹Δμ dreht sich gegenüber 30 dB um 0,6°/4,3°/7,8° (20/10/5 dB). Die schiefe „weil“-Begründung ist ersetzt.
- `A_0` ist durch F1₃₀ ersetzt. Statt „Belastbar …“ steht „Konsistent ist lediglich die Rangfolge“. Die Zusammenfassung nennt die
  konkreten Verluste (7,1 % bei 10 dB, 18,8 % bei 5 dB) statt „Robustheit bis 10 dB“.
- Einheiten nach DIN: „SNR in dB“, „RLA in %“ in allen Tabellen und Plots. Gl. 7.1 ist mit `align` gesetzt.
- **Plots:** `generate_plot.py`, `pr_ebene_plot.py` und `create_3d_plot.py` setzen das deutsche Dezimalkomma. Alle Abbildungen sind neu erzeugt.

### Kap. 8 (PDF-Seiten 45–47) – behoben

- FF1: Die Aktion ist das vorhergesagte Label. Die Reward-Matrix senkt die Soll-Schwelle, Elkans Formel liefert denselben Wert.
  In FF2 steht „kanalbedingte Klassenüberlappung“, in FF3 „Konsistent über alle Stufen ist die Rangfolge“.
- Der „Vorzug“ der Kostenasymmetrie ist ehrlich eingeordnet: Er ist konzeptionell, nicht empirisch belegt (Elkan-Gleichwertigkeit,
  implizite Schwelle 0,38–0,43). Schlussabsatz: „beste Mittelwerte, kein signifikanter Vorsprung“.
- Ran et al. korrekt wiedergegeben: zwei DQN zur Erkennung von FDIA und adversarialen Beispielen.
- Ausblick: Multi-Agenten-Punkt gestrichen (jetzt fünf Punkte). Punkt Betriebskennzahlen konkretisiert (Latenz, Ressourcen
  des Q-Netzes mit 17 922 Parametern gegen RF/SVM). d = 0 für nichtterminale Schritte, PER bleibt ohne Zitat.

**Einzelseiten-Check bis zum Literaturverzeichnis abgeschlossen.**

### Literaturverzeichnis und globaler Umbruch – behoben

- **Literaturverzeichnis (29 Einträge, alle zitiert):** URL und Abrufdatum bei Einträgen mit DOI entfernt, lokale Zotero-Pfade entfernt,
  ISSN global unterdrückt (`\AtEveryBibitem{\clearfield{issn}}`). Tagesdaten auf Monat gekürzt, AAAI-Heftnummern entfernt
  (vorher „Bd. 30. 1.“), Merzouk als „Art. 31“. Verzeichnis im Flattersatz (`\appto{\bibsetup}{\raggedright}`), DOIs brechen
  nicht mehr mitten in Zahlenfolgen um.
- **K2-7:** Durch die Textänderungen erledigt. Nach Messung des Weißraums aller Seiten (Kontaktabzug) bleiben außerhalb von
  Kapitelenden höchstens etwa 26 mm, unauffällig bei `\raggedbottom`. `\raggedbottom` bleibt bewusst (Standard für einseitigen Satz).
- **K5-8:** Tab. 5.2 und 5.3 stehen in einer gemeinsamen Gleitumgebung `[tbp]` auf einer Seite direkt nach ihrem Verweis.
- Hurenkinder und Schusterjungen sind global unterbunden (Penalties 10000). Die Gesamtsicht aller 50 Seiten ist unauffällig.

### Visualisierungen aus `compressedPDFs/screenshots.odt` (2026-11-16)

- 147 Screenshots aus 20 Quellen gesichtet, nur Visualisierungs-*Typen* übernommen. Umgesetzt: A Entscheidungsfunktion
  (Abb. 7.2), C LLR-Signalraum (Abb. 5.4), D Backup-Diagramme (Abb. 5.3), E Kanalnotation (Abb. 4.1). B (Lernkurve) auf Wunsch
  nicht umgesetzt. Verworfen: Konfusionsmatrizen, Balkenvergleiche, t-SNE, Architektur-/Infografiken, Rechenzeitdiagramm.
- **Reproduktion:** Die DDQN-Läufe wurden mit identischen Seeds auf CUDA neu trainiert (RTX 2060, ca. 2 min je Seed).
  Alle 36 P/R/F1-Werte aus `evaluation_results.csv` stimmen auf 4 Stellen, die Gewichte liegen in `modelle/`.
  Das MLP reproduziert `baseline_results.csv` ebenfalls exakt.
- **Befund Huber-Verlust:** `smooth_l1_loss` (β = 1) schätzt bei Reward-Abständen von 2 bzw. 2,5 nicht den Erwartungswert. Die
  Nullstelle der Q-Differenz liegt bei 3/7 statt 1/3. Das erklärt den Großteil der impliziten Schwelle 0,38–0,43, der Rest
  (Abfall mit dem SNR) kommt von der SNR-Mischung. Ein MSE-Kontrolllauf landet bei 0,30–0,32. Text angepasst in 5.2.1, 5.3
  (Herleitung als Gleichung), 7.1.1, 7.4, 7.5 und 8 (FF1, Einschränkungen, Ausblick „Verlustfunktion“).
- Abbildungsnummern in Kap. 7 verschoben: Entscheidungsfunktion 7.2, F1-Vergleich 7.3, PR-Ebene 7.4.
- `compressedPDFs/` (Screenshot-Sammlung, fremde Abbildungen) bleibt unversioniert und steht in der `.gitignore`.

### Methodisch-didaktischer Audit Kap. 1–8 (2026-11-16)

- **Neu:** Abschnitt 2.5 „Grundlagen des Reinforcement Learning“ (MDP, Return, Bellman-Gleichung, Q-Learning, DQN, Double DQN
  mit Gl. 2.6/2.7, Ein-Schritt-Fall als kontextueller Bandit mit Spielautomat-Analogie). 5.3 verweist darauf, statt die Ziele
  doppelt herzuleiten. Detektor-Analogie (Schwelle: Entdeckung gegen Fehlalarm) in 2.4.3.
- **Neu:** Algorithmus 5.1 (Trainingsschleife), Zeile für Zeile gegen `train()`/`learn()` geprüft. Die Prosa in 5.5 ist gekürzt.
- **Neu:** Tab. 7.3 (mittlerer Reward je Fluss) und Abb. 7.5 (Entscheidungsgrenzen im Schnitt). Abkürzungs- und
  Symbolverzeichnis im Vorspann (S. V–VII).
- **Typografie:** Alle Plots in Computer Modern (vorher DejaVu), Zahlenwerte unverändert.
- **Symbole bereinigt:** Lernrate α (η ist bei Bishop die Rauschvarianz), Laufindex im Algorithmus m, „Zielnetz“ statt „Target-Netz“.
- **Tabellennummern in Kap. 7 verschoben:** Mittlerer Reward 7.3, RLA/ELA 7.4, Bayes-Referenz 7.5. Abbildungen 7.1–7.4 unverändert,
  neu ist Abb. 7.5.
- **Bewusst weggelassen** (kein Erkenntnisgewinn oder irreführend): 3D-Decision-Regions, CNN-/Topologie-Diagramme,
  Konfusionsmatrizen, Box-/Kreis-/Balkendiagramme, Spiel- und Physik-Analogien, weitere Herleitungen, externe Zeichenwerkzeuge.

### Master-Audit und Umsetzung (2026-11-16)

Prüfung in fünf Dimensionen (Aufgabenstellung, Datenintegrität, Quellen, Stil, Verweise), danach Umsetzung der Punkte 1–3:
- **Aufgabenstellung:** Neuer Unterabschnitt 2.2.1 „Präventive und reaktive Gegenmaßnahmen“ nach Mirkovic & Reiher [4]
  (präventiv: Angriffs- und DoS-Prävention; reaktiv: Erkennung, Quellenidentifikation, Ratenbegrenzung, Filterung,
  Rekonfiguration; Einsatzort; Einordnung der Arbeit als Erkennungsstufe, hoher Recall spricht für Ratenbegrenzung).
  Abschnitt 2.2 heißt jetzt „Gegenmaßnahmen und klassische Angriffserkennung“. In 1.2 ordnet ein Absatz die vier Teilaufgaben
  den Kapiteln zu, die Aufbau-Liste nennt Anhang A. Der Ausblick (Kap. 8, Punkt 2) verweist auf 2.2.1.
- **Zahlen korrigiert:** Tab. 7.4 einheitlich aus Tab. 7.2 gerechnet, dadurch DDQN-ELA 10 dB 0,191 → 0,190, MLP-ELA 5 dB
  0,315 → 0,314, RF-ELA 5 dB 0,347 → 0,346 (auch 7.6). Vorher waren mehrere ELA-Werte aus Seed-Einzelwerten gerechnet.
  RLA im Text 18,75 / 20,22 % statt doppelt gerundeter 18,8 / 20,2 % (7.4, 7.6, 8), 7,13 % statt 7,1 %.
  Schwellenlücke DDQN–MLP bei 5 dB 0,04 statt 0,03 (Verweis auf Tab. A.3). Absatz zu Abb. 7.2 dreistellig, damit
  „um 0,04 bzw. 0,03 darunter“ nachrechenbar ist; „nahe dieser Stelle“ statt „dort“ (0,416 gegenüber 3/7).
  Recall-Streuung „rund 30 bis 40 %“ (+32 / +41 %).
- **Kleinigkeiten:** Kurzcaption Abb. 7.5 ohne „aller Verfahren“ (SVM fehlt), „Botnetz einer Mirai-Variante“, „verwundbare Geräte“
  (Bala: *vulnerable*), SARSA im Abkürzungsverzeichnis.
- **Anhang A „Messdaten und Reproduzierbarkeit“:** Tab. A.1 P/R/F1 aller Verfahren (Mittel ± Std), A.2 Einzelwerte je Seed,
  A.3 implizite Schwellen dreistellig, A.4 Zuordnung Element → Skript → Daten, Reproduktionsreihenfolge, Softwareversionen.
  Verweise darauf aus 5.6, 7.1, 7.1.1 und 7.4.
- **Punkt 4 (Stil und Beleg), umgesetzt:** Drei rhetorische Stellen geglättet: „Die Kehrseite:“ (Kap. 1) als Nebensatz,
  „Genau diesen Effekt korrigiert die ELA“ → „Die ELA gleicht diesen Effekt aus, weil …“, „Auffälliger als die Rangfolge ist
  die Spannweite.“ → „In der Robustheit unterscheiden sich die Modelle nur wenig.“ Die Schwellenanalogie in 2.4.3 ist mit
  Mirkovic & Reiher belegt (NBS-2, Nachteil „Threshold setting“: niedrige Schwelle → viele Fehlalarme, hohe Schwelle →
  geringere Empfindlichkeit, im PDF wörtlich geprüft).

### Sprach- und Typografie-Durchgang (2026-11-17)

Alle .tex-Dateien (Kap. 1–8, Anhang, `main.tex`, `verzeichnisse.tex`) vollständig gelesen, dazu chktex und Musterprüfungen
(Striche, Anführungszeichen, Einheiten, Abkürzungen, Doppelwörter, Leerzeichen vor `\ref`/`\autocite`). Keine Tippfehler gefunden.
- **Satzfehler im PDF:** `\texttt{gamma="scale"}` erschien als „gamma=ßcale"“, weil babel-ngerman `"s` als ß setzt.
  Jetzt `\texttt{gamma=\string"scale\string"}` (6.1). **Regel: kein `"` im Quelltext, auch nicht in `\texttt`.**
- **Rechtschreibung/Grammatik:** „gaußsche/-s/-n“ statt „Gaußsche“ (3.2.3, 5.4.1, Symbolverzeichnis; Duden seit 2006),
  „des Angriffsanteils von 37 %“ (4.4), SDN-Satz umformuliert (2.3.1), „liefern die Ergebnisse nicht“ statt unklarem „sie“
  (7.6), „Kleiner ist besser.“ nach Doppelpunkt (6.2.1).
- **Vereinheitlicht:** Reward-Vektor, „aufwendig“, „Botnetz-…“, durchgekoppelt „Intrusion-Detection-Systems“ und
  „Network-Intrusion-Detection-Systemen“, keine Sätze mit Formelzeichen am Anfang („Der Strafterm $\Omega$“, „Die Kovarianz $\boldsymbol\Sigma$“).
- **Typografie:** `~\autocite` an 110 Stellen im Fließtext und in Bildunterschriften. In Tabellenzellen (Tab. 3.1/3.2)
  bleibt das normale Leerzeichen. Vorher begannen zwei Zeilen mit einer Quellennummer („[1].“ S. 3, „[2].“ S. 4). Die dadurch
  in 2.1.3 entstandene Zusatzzeile hätte 2.3–2.5 je eine Seite verschoben. Ausgeglichen mit „bis zum Ausfall des Dienstes“
  (statt „vollständigen Ausfall“), Seitenumbrüche per Vergleichsbuild identisch mit dem Stand davor.
- `nachweis_zahlen.py`: zwei Suchmuster angepasst (gamma, 37 %), weiterhin 203/203.
- Fehlalarme von chktex (Leerzeichen vor `\label`, Satzabstände unter `\frenchspacing`, `\allowbreak`, Intervall `[0,1)`)
  bewusst nicht geändert.
- **Kommaprüfung mit LanguageTool (LTeX+ 18.7, de-DE, CLI auf alle 9 Kapiteldateien):** keine echten Kommafehler. Gemeldet
  wurden nur Fehlalarme (`\,\%` vor Satzzeichen, „Angriffe wie UDP-Floods“ ohne Nebensatz). Einzige echte Grammatikmeldung:
  Genitiv „eines Double Deep Q-Networks“ (Kap. 1), korrigiert. `.vscode/settings.json` setzt `ltex.language` auf `de-DE`;
  ohne diese Einstellung prüft LTeX+ die .tex-Dateien als Englisch.

### Komplett-Audit vor der Abgabe (2026-11-17)

Alle .tex-Dateien vollständig gelesen; 122 `\autocite` (140 Schlüsselnennungen) automatisch gegen alle 29 Quell-PDFs
abgeglichen (Zahlen, Fachbegriffe) und jede Literaturzahl im PDF-Kontext nachgelesen; Plagiatsabgleich (gemeinsame
7-Wort-Folgen mit allen Quellen: nur Fachbegriffe); Formeln numerisch nachgerechnet (Huber-Minimierer, Elkans p*,
TikZ-Dichten Abb. 4.1); Implementierungsaussagen gegen `train_eval.py`; Stilstatistik; Sichtprüfung aller 63 Seiten.
- **Umgesetzt:** Wiederholungen gestrichen (AE-RL-Absatz 3.4, π = 0,3/37 % in 4.5, Füllsatz 5.3, doppelte Einordnung in der
  Überschrift von Tab. 3.2). Belege korrigiert: Entropie-Merkmale kommen in DDoS-AT-2022, IoT-DH und Rasas eigener
  Merkmalsauswahl nicht vor (jetzt Paketrate, Zeitabstände, Flag-Zähler; Entropie nur noch als Ansatz aus Rasas
  Literaturübersicht, 3.2.3 und 4.3), „frühe … NetFlow“ unbelegt (3.2), Henderson gilt für TRPO (7.1). XAI ausgeschrieben
  („erklärbare Künstliche Intelligenz“). Doppelpunkt-Konstruktionen in 2.4.3, 7 und 8 aufgelöst.
- **Bestätigt:** Alle Literaturzahlen stehen wörtlich in den PDFs. IoT-DH 37 %: Saif et al. haben die Konfusionsmatrix
  (Tab. 3) vertauscht beschriftet; maßgeblich sind Tab. 2 (Support Klasse 1 = 748) und die Labeldefinition.
- Layout per Vergleichsbuild unverändert: 63 Seiten, alle Kapitel- und Abschnittsanfänge auf denselben Seiten.

### Offene Fragen an den Betreuer (Prof. Seitz)

- **Aufgabenstellung vs. synthetisches Modell:** Die Aufgabenstellung verlangt, „an einem Data Set, das den Netzverkehr mit einem
  DDoS-Angriff beschreibt, die Erkennung des Angriffs zu demonstrieren“. Die Arbeit evaluiert ausschließlich auf einem
  parametrischen Gauß-Flussmodell, das sich an DDoS-AT-2022 und IoT-DH orientiert (Begründung: kontrollierbares SNR,
  analytische Bayes-Grenze). Die Arbeit legt das offen (4.1, Kap. 8 Ausblick). Vor der Abgabe klären, ob das akzeptiert wird
  oder ob ein ergänzender Lauf auf einem der Originaldatensätze erwartet wird.
- **Zahl der Seeds (n = 3):** Alle Mittelwerte und Streuungen beruhen auf drei Seeds. Die Abstände zwischen DDQN und MLP
  (0,25–0,62 Prozentpunkte F1, 0,011–0,016 Reward je Fluss) liegen in der Größenordnung der Seed-Streuung, die Arbeit nennt
  sie deshalb ausdrücklich nicht signifikant. Henderson et al. 2018 zeigen, dass schon 2 × 5 Seeds verschiedene Ergebnisse
  liefern können. Ein Lauf mit 10 Seeds wäre technisch schnell machbar (DDQN ca. 2 min je Seed auf der RTX 2060, Baselines
  wenige Minuten), würde aber alle Tabellen und Zahlen in Kap. 7 und 8 ändern. Vor der Abgabe klären, ob das erwartet wird.

### Vorgemerkt für das Ende des Seiten-Checks

- K2-7 und K5-8 erledigt (siehe Abschnitt „Literaturverzeichnis und globaler Umbruch“).

### Nächste Seiten

Inhaltliche und technische Arbeit endgültig abgeschlossen, Master-Audit einschließlich Punkt 4 und Sprachdurchgang
umgesetzt. Vor der Abgabe bleiben nur die manuellen Formalia (siehe „Vor der Abgabe“) und die Klärung der zwei Betreuer-Fragen.

## Arbeitsweise

- Nur an einem Ort arbeiten (lokal **oder** Cloud), vorher `git pull`.
- Nach jeder Änderung vollständig kompilieren und das Log auf Fehler, undefinierte Verweise sowie Over-/Underfull prüfen.
- Nach jeder Änderung an Zahlen, Tabellen oder Abbildungen `anhang_tabellen.py` und `nachweis_zahlen.py` laufen lassen
  (0 Abweichungen, sonst nicht committen).
