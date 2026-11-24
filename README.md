# Atsiliepimų sentimentų analizė

Projektą sudaro dvi dalys:

1. `train_models.py` – ankstesnis mokslinis eksperimentas, lyginantis TF-IDF + Logistic Regression ir TF-IDF + LinearSVC pagal žvaigždučių įvertinimus.
2. `app.py` – interaktyvi programa, į kurią galima įkelti CSV arba Excel failą ir klasifikuoti atsiliepimus lietuvių bei anglų kalbomis.

## Programos galimybės
- CSV, XLSX ir XLS failų įkėlimas.
- Automatinis galimo atsiliepimų stulpelio atpažinimas; jį galima pasirinkti ir rankiniu būdu.
- Klasifikavimas į teigiamus, neigiamus ir neutralius atsiliepimus.
- Lietuvių ir anglų kalbų palaikymas su daugiakalbiu GLiClass modeliu.
- Rezultatų statistika, filtravimas ir atsisiuntimas CSV / Excel formatais.
- Žvaigždučių įvertinimai nėra būtini.

## Paleidimas Windows
Rekomenduojama Python 3.10 arba naujesnė versija.

1. Išarchyvuokite projektą.
2. Atidarykite CMD arba PowerShell projekto aplanke.
3. Sukurkite virtualią aplinką:
   ```powershell
   python -m venv .venv
   ```
4. Aktyvuokite:
   ```powershell
   .venv\Scripts\activate
   ```
5. Įdiekite bibliotekas:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
6. Paleiskite:
   ```powershell
   streamlit run app.py
   ```
7. Naršyklėje atidarykite terminale rodomą `Local URL`, jei puslapis neatsidaro automatiškai.

## Pirmasis paleidimas
Pirmą kartą paleidus analizę, daugiakalbis modelis atsisiunčiamas iš Hugging Face, todėl reikalingas interneto ryšys. Modelio failai išsaugomi lokaliai.

Modelis: `knowledgator/gliclass-multilang-edge`. Jo modelio kortelėje nurodytas lietuvių kalbos palaikymas. Modelis gali klysti, ypač su sarkazmu, ironija, mišriomis nuomonėmis ir labai trumpais tekstais. Svarbius rezultatus reikėtų patikrinti žmogui.

## Įkeliamo failo struktūra
Faile turi būti atsiliepimų tekstų stulpelis. Programa atpažįsta įprastus pavadinimus: `content`, `review`, `review_text`, `text`, `comment`, `feedback`, `atsiliepimas`, `komentaras`, `tekstas`. Jei pavadinimas kitoks, stulpelį galima pasirinkti programoje.

## Senasis mokymo eksperimentas
Norėdami pakartoti ankstesnį modelių palyginimą:
```powershell
python train_models.py
```
Šiai mokymo daliai CSV faile būtini `content` ir `score` stulpeliai, o `score` turi būti 1–5 žvaigždučių įvertinimas. Ši dalis skirta eksperimentui, o ne naujų failų analizei per programos langą.

## Failai
- `app.py` – interaktyvi programa.
- `train_models.py` – ankstesnis modelių mokymo ir palyginimo kodas.
- `requirements.txt` – bibliotekos.
- `Candy Crush Saga.csv` – pradinis duomenų pavyzdys.
- `Instrukcija_kaip_uzbaigti.docx` – pradinė instrukcija.
- `pavyzdys_dvikalbis.csv` – bandomieji lietuviški ir angliški atsiliepimai.
