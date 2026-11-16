"""Erzeugt das Begleitarchiv Bachelorarbeit_Begleitarchiv.zip aus den versionierten Dateien plus main.pdf.

Nicht ins Archiv: Quellen-PDFs (QuellenLiteratur/), Arbeitsnotizen (UEBERGABE.md, CLAUDE.md), .github/, Themenblatt.
Voraussetzung: main.pdf ist aktuell gebaut. Aufruf: python abgabe_archiv.py
"""

import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).parent
ZIEL = HERE / "Bachelorarbeit_Begleitarchiv.zip"
AUSSCHLUSS_PRAEFIX = ("QuellenLiteratur/", ".github/")
AUSSCHLUSS = {"UEBERGABE.md", "CLAUDE.md", ".gitignore", "13112025_Themenblatt_Bachelorarbeit_Mustafa_Al_Obaidi.pdf"}


def main() -> int:
    if not (HERE / "main.pdf").exists():
        print("main.pdf fehlt, zuerst die Arbeit bauen.")
        return 1
    dateien = subprocess.run(["git", "ls-files"], cwd=HERE, capture_output=True, text=True, check=True).stdout.split("\n")
    dateien = sorted({d for d in dateien if d and d not in AUSSCHLUSS and not d.startswith(AUSSCHLUSS_PRAEFIX)}
                     | {"README.md", "requirements.txt"})
    with zipfile.ZipFile(ZIEL, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(HERE / "main.pdf", "Bachelorarbeit/Bachelorarbeit.pdf")
        for d in dateien:
            z.write(HERE / d, f"Bachelorarbeit/{d}")
    print(f"{ZIEL.name}: {len(dateien) + 1} Dateien, {ZIEL.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
