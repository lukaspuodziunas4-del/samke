# Atsiliepimų sentimentų analizė

## Projekto paskirtis
Programa klasifikuoja Google Play žaidimų atsiliepimus į tris sentimentų klases:
- 1–2 žvaigždutės – Neigiamas
- 3 žvaigždutės – Neutralus
- 4–5 žvaigždutės – Teigiamas

## Du metodai
**Baseline:** TF-IDF (žodžiai) + Logistic Regression.

**Proposed:** TF-IDF žodžių 1–2 gramai + simbolių 3–5 gramai + LinearSVC.

## Paleidimas Windows
1. Įsidiekite Python 3.10 arba naujesnį.
2. Atidarykite terminalą projekto aplanke.
3. Įvykdykite:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   python train_models.py
   ```
4. Rezultatai bus aplanke `results`.

## Duomenų įkėlimas
Į `data` aplanką įdėkite CSV failus. Kiekviename faile turi būti stulpeliai `content` ir `score`.
Programa automatiškai perskaitys visus aplanke esančius CSV failus. Taigi galima pridėti ir kitų 9 žaidimų duomenis.

## Eksperimentų metodika
Duomenys padalijami į 80 % mokymo ir 20 % testavimo dalį. Naudojamas `stratify`, kad klasių proporcijos išliktų panašios. Abu modeliai mokomi su ta pačia mokymo imtimi ir vertinami su ta pačia testavimo imtimi.

Vertinimo metrikos: Accuracy, weighted Precision, weighted Recall ir weighted F1-score. Taip pat sukuriamos klaidų matricos.

## Svarbus apribojimas
Sentimentų etiketės automatiškai sudaromos pagal žvaigždučių įvertinimą. Žvaigždutės ne visada sutampa su tikra teksto emocija. Ataskaitoje šį apribojimą būtina paminėti. Jeigu turite galimybę, dalį atsiliepimų verta rankiniu būdu patikrinti ir pataisyti etiketes.

## Failų struktūra
- `data/` – pradiniai CSV duomenys
- `train_models.py` – abiejų metodų mokymas ir eksperimentas
- `requirements.txt` – bibliotekos
- `results/` – metrikos, grafikai, paruošti duomenys ir išsaugoti modeliai
