from __future__ import annotations
import io
import pandas as pd
import streamlit as st
import torch
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer

MODEL_ID = "knowledgator/gliclass-multilang-edge"
LABELS = ["positive", "negative", "neutral"]
LABEL_LT = {"positive": "Teigiamas", "negative": "Neigiamas", "neutral": "Neutralus"}
PROMPT = """
Classify the sentiment of this customer review. Choose the best matching sentiment:
- positive: satisfied, praises something, likes it, or describes a good experience.
  Lithuanian: teigiamas atsiliepimas, žmogus patenkintas, giria arba aprašo gerą patirtį.
- negative: dissatisfied, complains, criticizes something, or describes a bad experience.
  Lithuanian: neigiamas atsiliepimas, žmogus nepatenkintas, skundžiasi arba kritikuoja.
- neutral: factual, mixed without a clear overall sentiment, or no clear positive/negative opinion.
  Lithuanian: neutralus atsiliepimas, faktinė informacija arba nėra aiškios teigiamos ar neigiamos nuomonės.
The review can be written in Lithuanian or English. Judge the meaning, not the language.
"""
TEXT_COLUMN_NAMES = {
    "content", "review", "reviews", "review_text", "text", "comment", "comments",
    "feedback", "message", "body", "description", "atsiliepimas", "atsiliepimai",
    "tekstas", "komentaras", "komentarai", "nuomone", "nuomonė", "aprasymas", "aprašymas",
}

@st.cache_resource(show_spinner=False)
def load_classifier():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Force FP32 weights to avoid Float/BFloat16 matrix multiplication errors.
    model = GLiClassModel.from_pretrained(MODEL_ID)
    model = model.to(device=device, dtype=torch.float32)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    classifier = ZeroShotClassificationPipeline(
        model, tokenizer, classification_type="multi-label", device=device
    )
    return classifier, device

def read_uploaded_file(uploaded_file, file_name):
    suffix = file_name.lower().rsplit(".", 1)[-1]
    raw = uploaded_file.getvalue()
    if suffix == "csv":
        try:
            return pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding="cp1257")
    if suffix in {"xlsx", "xls"}:
        return pd.read_excel(io.BytesIO(raw))
    raise ValueError("Palaikomi tik CSV, XLSX ir XLS failai.")

def find_text_column(columns):
    normalized = {str(c).strip().lower(): c for c in columns}
    for name in TEXT_COLUMN_NAMES:
        if name in normalized:
            return normalized[name]
    return None

def classify_reviews(texts, classifier, batch_size=8):
    labels_out, scores_out = [], []
    for start in range(0, len(texts), batch_size):
        predictions = classifier(
            texts[start:start + batch_size], LABELS, prompt=PROMPT, threshold=0.0
        )
        for item in predictions:
            if not item:
                labels_out.append("Neaišku")
                scores_out.append(0.0)
                continue
            best = max(item, key=lambda result: float(result.get("score", 0.0)))
            key = str(best.get("label", "")).lower()
            labels_out.append(LABEL_LT.get(key, "Neaišku"))
            scores_out.append(float(best.get("score", 0.0)))
    return labels_out, scores_out

st.set_page_config(page_title="Atsiliepimų analizė", page_icon="💬", layout="wide")
st.title("💬 Atsiliepimų sentimentų analizė")
st.write("Įkelkite CSV arba Excel failą. Programa išanalizuos atsiliepimus lietuvių ir anglų kalbomis.")

with st.sidebar:
    st.header("Apie programą")
    st.markdown(
        "- **Teigiamas** – geri įspūdžiai ir pagyrimai.\n"
        "- **Neigiamas** – skundai ir neigiami įspūdžiai.\n"
        "- **Neutralus** – faktai arba neaiški / mišri nuomonė."
    )
    st.caption("Pirmą kartą modelis atsisiunčiamas iš interneto.")

uploaded = st.file_uploader("Pasirinkite atsiliepimų failą", type=["csv", "xlsx", "xls"])
if uploaded is not None:
    try:
        df = read_uploaded_file(uploaded, uploaded.name)
        if df.empty:
            st.error("Įkeltame faile nėra duomenų.")
            st.stop()
        st.subheader("1. Įkelti duomenys")
        st.caption(f"Rasta {len(df):,} eilučių ir {len(df.columns)} stulpelių.")
        st.dataframe(df.head(8), use_container_width=True)
        detected = find_text_column(df.columns)
        options = list(df.columns)
        default_index = options.index(detected) if detected in options else 0
        text_column = st.selectbox(
            "Pasirinkite stulpelį, kuriame yra atsiliepimų tekstai:",
            options=options, index=default_index
        )
        if st.button("🔍 Analizuoti atsiliepimus", type="primary", use_container_width=True):
            texts = df[text_column].fillna("").astype(str).str.strip().tolist()
            valid_indices = [i for i, value in enumerate(texts) if value]
            if not valid_indices:
                st.error("Pasirinktame stulpelyje nerasta atsiliepimų tekstų.")
                st.stop()
            with st.spinner("Kraunamas daugiakalbis AI modelis ir analizuojami atsiliepimai..."):
                classifier, device = load_classifier()
                valid_labels, valid_scores = classify_reviews(
                    [texts[i] for i in valid_indices], classifier
                )
            all_labels = ["Tuščias" if not value else "" for value in texts]
            all_scores = [0.0] * len(texts)
            for idx, label, score in zip(valid_indices, valid_labels, valid_scores):
                all_labels[idx], all_scores[idx] = label, score
            result = df.copy()
            result["Sentimentas"] = all_labels
            result["Modelio įvertinimas"] = all_scores
            st.session_state["sentiment_result"] = result
    except Exception as exc:
        st.error(f"Nepavyko perskaityti arba išanalizuoti failo: {exc}")

if "sentiment_result" in st.session_state:
    result = st.session_state["sentiment_result"]
    st.subheader("2. Analizės rezultatai")
    counts = result["Sentimentas"].value_counts()
    total = len(result)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Iš viso įrašų", f"{total:,}")
    c2.metric("Teigiami", f"{int(counts.get('Teigiamas', 0)):,}")
    c3.metric("Neigiami", f"{int(counts.get('Neigiamas', 0)):,}")
    c4.metric("Neutralūs", f"{int(counts.get('Neutralus', 0)):,}")
    summary = (
        result["Sentimentas"].value_counts()
        .reindex(["Teigiamas", "Neigiamas", "Neutralus", "Tuščias"], fill_value=0)
        .rename_axis("Sentimentas").reset_index(name="Kiekis")
    )
    summary["Procentai"] = (summary["Kiekis"] / total * 100).round(1)
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.bar_chart(summary.set_index("Sentimentas")["Kiekis"])
    st.subheader("Surūšiuoti atsiliepimai")
    chosen = st.multiselect(
        "Filtruoti pagal sentimentą:",
        ["Teigiamas", "Neigiamas", "Neutralus", "Tuščias"],
        default=["Teigiamas", "Neigiamas", "Neutralus"],
    )
    st.dataframe(result[result["Sentimentas"].isin(chosen)], use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Atsisiųsti CSV", data=result.to_csv(index=False).encode("utf-8-sig"),
        file_name="surusiuoti_atsiliepimai.csv", mime="text/csv", use_container_width=True
    )
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        result.to_excel(writer, index=False, sheet_name="Atsiliepimai")
        summary.to_excel(writer, index=False, sheet_name="Suvestine")
    st.download_button(
        "⬇️ Atsisiųsti Excel", data=excel_buffer.getvalue(),
        file_name="surusiuoti_atsiliepimai.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
    st.caption(
        "Modelio įvertinimas nėra garantuotas tikslumas. Sarkazmas, ironija, mišrios nuomonės "
        "ir labai trumpi tekstai gali būti klasifikuojami netiksliai."
    )
