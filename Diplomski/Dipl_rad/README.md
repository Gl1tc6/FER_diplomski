# Upute za generiranje `dipl.pdf`

Projekt koristi **pdfLaTeX** (ne XeLaTeX/LuaLaTeX), klasičan **BibTeX** (poziva `fer_plainnat_HR.bst` / `fer_IEEEtran_HR.bst`) te paket `pdfx` za PDF/A izlaz (zato postoji `dipl.xmpdata`).

## Potrebni alati

- **Kompletna LaTeX distribucija** (potrebno je puno paketa: `fer.cls`, `pdfx`, `xltabular`, `babel` s hrvatskim jezikom, `mathptmx`, `tgtermes`, `titlesec`, `hyperref`, `listings`...):
  - Linux: **TeX Live** (najbolje `texlive-full` ili barem `texlive-lang-croatian` + `texlive-latex-extra` + `texlive-fonts-extra`)
  - Windows: **MiKTeX** (automatski instalira pakete koji nedostaju) ili **TeX Live**
- (Preporučeno) **`latexmk`** — automatski odradi sve potrebne prolaze (pdflatex → bibtex → pdflatex → pdflatex) u jednoj naredbi
- (Opcionalno) Editor: **TeXstudio** ili **VS Code** s ekstenzijom *LaTeX Workshop* — nije nužno, može se raditi i iz terminala

## Linux

Instalirajte potrebne pakete:

```bash
sudo apt update
sudo apt install texlive-full latexmk
# ili minimalnije, ako ne želite texlive-full (par GB):
# sudo apt install texlive-latex-extra texlive-lang-croatian texlive-fonts-extra texlive-bibtex-extra latexmk
```

Otvorite terminal u mapi projekta i pokrenite kompajliranje:

```bash
cd /home/ante/Documents/GitHub/FER_diplomski/Diplomski/Dipl_rad
latexmk -pdf dipl.tex
```

Ako radije koristite ručni niz naredbi umjesto `latexmk`:

```bash
pdflatex dipl.tex
bibtex dipl
pdflatex dipl.tex
pdflatex dipl.tex
```

Rezultat je datoteka `dipl.pdf` u istoj mapi.

## Windows

1. Preuzmite i instalirajte **MiKTeX** s adrese https://miktex.org/download. Tijekom instalacije odaberite opciju "Install missing packages on-the-fly = Yes".
2. Otvorite **MiKTeX Console** i pokrenite provjeru ažuriranja (Check for updates).
3. Otvorite **Command Prompt** ili **PowerShell** u mapi projekta i pokrenite:

```bash
cd C:\putanja\do\Diplomski\Dipl_rad
latexmk -pdf dipl.tex
```

Ako `latexmk` nije dostupan (MiKTeX ga obično ponudi za instalaciju na upit, pa ga jednostavno prihvatite), koristite ručni niz naredbi:

```bash
pdflatex dipl.tex
bibtex dipl
pdflatex dipl.tex
pdflatex dipl.tex
```

Prilikom prvog pokretanja MiKTeX će vjerojatno tražiti potvrdu za instalaciju paketa kojih nema (npr. `pdfx`, `xltabular`, `mathptmx`, hrvatski `babel` uzorci) — dopustite instalaciju kada se to od vas zatraži.

## Napomene

- Sve sljedeće datoteke moraju ostati u **istoj mapi**: `dipl.tex`, sva poglavlja (`1uvod.tex` ... `7zakljucak.tex`), `fer.cls`, `literatura.bib`, `fer_plainnat_HR.bst`/`fer_IEEEtran_HR.bst`, `dipl.xmpdata`, `hr_0036540817_94.pdf` (zadatak koji se uključuje preko `\zadatak{}`) te mapa `slike/` sa slikama.
- Naziv izlazne datoteke odgovara imenu `.tex` datoteke koju kompajlirate — zato je potrebno kompajlirati baš `dipl.tex`, a ne pojedina poglavlja, kako bi izlaz bio `dipl.pdf`.
- Ako BibTeX javi grešku da ne može pronaći `.bst` stil, provjerite da ste naredbu pokrenuli iz mape projekta (ne iz podmape), jer su `.bst` datoteke lokalne, a ne instalirane u distribuciju.
- Prvo kompajliranje zna biti sporije jer `pdfx`/PDF-A dio i hrvatski `babel` prijelomi zahtijevaju dodatne pakete — svaki naredni build je brži.
