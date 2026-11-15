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

Inhaltlich und technisch keine offenen Punkte: Quellen geprüft, Kap. 7 und Fazit konsistent, Build sauber (50 Seiten, 0 Fehler, 0 Warnungen).
Hinweis: Henderson 2018 bleibt ohne Seitenzahlen (siehe Literatur), das Zitat ist über die DOI vollständig.

### Vor der Abgabe (manuell)

1. **Deckblatt:** In `main.tex` gibt es eine eigene `titlepage` (Z. 78–101). Sie ist gegen die Vorgaben des Fachgebiets
   Kommunikationsnetze bzw. der Fakultät EI der TU Ilmenau zu prüfen (Titel laut `Aufgabenstellung_Bachelorarbeit.pdf`,
   Betreuer, Matrikelnummer, Datum).
2. **Eigenständigkeitserklärung:** Fehlt noch im LaTeX-Quelltext. Wortlaut und Position gemäß gültiger Prüfungsordnung
   der TU Ilmenau einfügen und unterschreiben. Eine ggf. geforderte Erklärung zum Einsatz von KI-Werkzeugen gehört dazu.
3. **Letzter Textdurchgang:** Korrekturlesen auf Rechtschreibung, Stil (`CLAUDE.md`) und Abbildungs- und Tabellenverweise.
   Ein vollständiges Korrekturlesen fand in den Sitzungen nicht statt.
4. Danach vollständig kompilieren und das PDF für die Abgabe erzeugen (`main.pdf` ist nicht versioniert).

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

### Offene Frage an den Betreuer (Prof. Seitz)

- **Aufgabenstellung vs. synthetisches Modell:** Die Aufgabenstellung verlangt, „an einem Data Set, das den Netzverkehr mit einem
  DDoS-Angriff beschreibt, die Erkennung des Angriffs zu demonstrieren“. Die Arbeit evaluiert ausschließlich auf einem
  parametrischen Gauß-Flussmodell, das sich an DDoS-AT-2022 und IoT-DH orientiert (Begründung: kontrollierbares SNR,
  analytische Bayes-Grenze). Die Arbeit legt das offen (4.1, Kap. 8 Ausblick). Vor der Abgabe klären, ob das akzeptiert wird
  oder ob ein ergänzender Lauf auf einem der Originaldatensätze erwartet wird.

### Vorgemerkt für das Ende des Seiten-Checks

- K2-7 und K5-8 erledigt (siehe Abschnitt „Literaturverzeichnis und globaler Umbruch“).

### Nächste Seiten

Technische Qualitätsphase abgeschlossen. Offen sind nur die Betreuer-Frage und die manuellen Formalia (siehe „Vor der Abgabe“).

## Arbeitsweise

- Nur an einem Ort arbeiten (lokal **oder** Cloud), vorher `git pull`.
- Nach jeder Änderung vollständig kompilieren und das Log auf Fehler, undefinierte Verweise sowie Over-/Underfull prüfen.
