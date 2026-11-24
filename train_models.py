"""
Atsiliepimų sentimentų analizė – Baseline ir Proposed metodų palyginimas.

Duomenys: CSV failai su stulpeliais:
- content: atsiliepimo tekstas
- score: įvertinimas 1–5 žvaigždutėmis

Etiketės:
1–2 = neigiamas, 3 = neutralus, 4–5 = teigiamas.
"""

from pathlib import Path
import argparse
import re
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

LABELS = ["Neigiamas", "Neutralus", "Teigiamas"]

def clean_text(value):
    """Paprastas teksto valymas, išsaugant žodžius ir skyrybą."""
    if pd.isna(value):
        return ""
    text = str(value).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_data(data_dir):
    files = sorted(Path(data_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"Aplanke {data_dir} nerasta CSV failų.")

    frames = []
    for file in files:
        df = pd.read_csv(file)
        if "content" not in df.columns or "score" not in df.columns:
            print(f"Praleidžiamas {file.name}: nėra content arba score stulpelio.")
            continue
        part = df[["content", "score"]].copy()
        part["zaidimas"] = file.stem
        frames.append(part)

    if not frames:
        raise ValueError("CSV failuose nerasta reikalingų stulpelių: content ir score.")

    data = pd.concat(frames, ignore_index=True)
    data["content"] = data["content"].apply(clean_text)
    data["score"] = pd.to_numeric(data["score"], errors="coerce")
    data = data.dropna(subset=["content", "score"])
    data = data[data["content"].str.len() > 0]
    data = data[data["score"].between(1, 5)]
    data["score"] = data["score"].astype(int)

    data["sentiment"] = data["score"].map(
        lambda x: "Neigiamas" if x <= 2 else ("Neutralus" if x == 3 else "Teigiamas")
    )
    # Pašalinami tik visiškai vienodi teksto ir etiketės įrašai.
    data = data.drop_duplicates(subset=["content", "sentiment"]).reset_index(drop=True)
    return data

def evaluate(name, model, X_train, X_test, y_train, y_test, out_dir):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="weighted", zero_division=0
    )
    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(
        y_test, predictions, labels=LABELS, target_names=LABELS,
        zero_division=0, output_dict=True
    )
    pd.DataFrame(report).transpose().to_csv(
        out_dir / f"{name.lower()}_classification_report.csv",
        encoding="utf-8-sig"
    )

    cm = confusion_matrix(y_test, predictions, labels=LABELS)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"{name}: klaidų matrica")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name.lower()}_confusion_matrix.png", dpi=160)
    plt.close(fig)

    joblib.dump(model, out_dir / f"{name.lower()}_model.joblib")

    return {
        "Metodas": name,
        "Accuracy": accuracy,
        "Precision (weighted)": precision,
        "Recall (weighted)": recall,
        "F1-score (weighted)": f1,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data", help="Aplankas su CSV failais")
    parser.add_argument("--output", default="results", help="Rezultatų aplankas")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(args.data)
    if data["sentiment"].nunique() < 2:
        raise ValueError("Reikia bent dviejų skirtingų sentimentų klasių.")

    data.to_csv(out_dir / "paruošti_duomenys.csv", index=False, encoding="utf-8-sig")
    data["sentiment"].value_counts().reindex(LABELS, fill_value=0).rename_axis(
        "Sentimentas"
    ).reset_index(name="Kiekis").to_csv(
        out_dir / "klasiu_pasiskirstymas.csv", index=False, encoding="utf-8-sig"
    )

    X = data["content"]
    y = data["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # BASELINE: vieno žodžio TF-IDF požymiai + Logistic Regression.
    baseline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 1), min_df=2, max_features=50000,
            sublinear_tf=True, strip_accents="unicode"
        )),
        ("classifier", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ))
    ])

    # PROPOSED: žodžių 1–2 gramų ir simbolių 3–5 gramų TF-IDF požymiai
    # + LinearSVC. Tai leidžia atsižvelgti į frazes ir žodžių formas / rašybą.
    proposed = Pipeline([
        ("features", FeatureUnion([
            ("word_tfidf", TfidfVectorizer(
                analyzer="word", ngram_range=(1, 2), min_df=2,
                max_features=60000, sublinear_tf=True,
                strip_accents="unicode"
            )),
            ("char_tfidf", TfidfVectorizer(
                analyzer="char", ngram_range=(3, 5), min_df=2,
                max_features=60000, sublinear_tf=True
            ))
        ])),
        ("classifier", LinearSVC(class_weight="balanced", random_state=42))
    ])

    rows = []
    rows.append(evaluate("Baseline", baseline, X_train, X_test, y_train, y_test, out_dir))
    rows.append(evaluate("Proposed", proposed, X_train, X_test, y_train, y_test, out_dir))

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "metodu_palyginimas.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8, 5))
    metrics = ["Accuracy", "Precision (weighted)", "Recall (weighted)", "F1-score (weighted)"]
    x = range(len(metrics))
    width = 0.35
    ax.bar([i - width/2 for i in x], results.loc[0, metrics], width, label="Baseline")
    ax.bar([i + width/2 for i in x], results.loc[1, metrics], width, label="Proposed")
    ax.set_xticks(list(x))
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1-score"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Reikšmė")
    ax.set_title("Baseline ir Proposed metodų palyginimas")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "metodu_palyginimo_grafikas.png", dpi=160)
    plt.close(fig)

    # Paprasta rezultatų suvestinė tekstiniam ataskaitos panaudojimui.
    best = results.loc[results["F1-score (weighted)"].idxmax()]
    summary = f"""EKSPERIMENTO SUVESTINĖ

Duomenų įrašų po paruošimo: {len(data)}
Mokymo imtis: {len(X_train)}
Testavimo imtis: {len(X_test)}
Atsitiktinė būsena (random_state): 42
Testavimo dalis: 20 %

Etikečių sudarymas pagal žvaigždutes:
1–2 = Neigiamas; 3 = Neutralus; 4–5 = Teigiamas.

Geriausias metodas pagal weighted F1: {best['Metodas']}
Accuracy: {best['Accuracy']:.4f}
Precision (weighted): {best['Precision (weighted)']:.4f}
Recall (weighted): {best['Recall (weighted)']:.4f}
F1-score (weighted): {best['F1-score (weighted)']:.4f}

DĖMESIO: etiketės sukurtos pagal žvaigždučių įvertinimą, todėl tai yra
įvertinimu paremtas sentimentų pakaitinis žymėjimas. Žvaigždutės ne visada
tiksliai atspindi teksto emociją (pvz., tekstas gali būti teigiamas, o
įvertinimas – žemas). Rezultatus reikia interpretuoti atsižvelgiant į šį
apribojimą.
"""
    (out_dir / "eksperimento_suvestine.txt").write_text(summary, encoding="utf-8")

    print(results.to_string(index=False))
    print(f"\nRezultatai išsaugoti aplanke: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
