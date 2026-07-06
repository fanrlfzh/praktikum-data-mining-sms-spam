import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)

st.set_page_config(
    page_title="SMS Spam Detection System",
    page_icon="🛡",
    layout="wide"
)

CONTOH_PESAN = {
    "Pilih contoh pesan": "",
    "🚨 Spam - Cash Prize": "URGENT! You have won a £1000 cash prize. Call now to claim your reward.",
    "🚨 Spam - Free Voucher": "Congratulations! You have been selected for a FREE £500 gift voucher. Claim now.",
    "🚨 Spam - Competition": "FREE entry into our weekly competition. Text WIN to 80085 now.",
    "✅ Ham - Meeting": "Hey, are we still meeting for lunch tomorrow?",
    "✅ Ham - Grocery": "I will be home around 7 pm. Do you need anything from the store?",
    "✅ Ham - Birthday": "Happy birthday! Hope you have a wonderful day.",
}

@st.cache_data
def load_default_dataset():
    df = pd.read_csv(
        "SMSSpamCollection", sep="\t", header=None,
        names=["label", "message"], encoding="utf-8"
    )
    return df


def load_uploaded_dataset(uploaded_file):
    raw = uploaded_file.read()
    # Coba format asli SMSSpamCollection (tab-separated, tanpa header)
    try:
        df = pd.read_csv(
            io.BytesIO(raw), sep="\t", header=None,
            names=["label", "message"], encoding="utf-8"
        )
        if df["label"].isin(["ham", "spam"]).mean() > 0.9:
            return df
    except Exception:
        pass
    # Coba format CSV umum (kolom v1/v2 atau label/message)
    df = pd.read_csv(io.BytesIO(raw), encoding="latin-1")
    cols_lower = [c.lower() for c in df.columns]
    df.columns = cols_lower
    if "v1" in df.columns and "v2" in df.columns:
        df = df.rename(columns={"v1": "label", "v2": "message"})
    df = df[["label", "message"]].dropna()
    return df


@st.cache_resource
def train_model(df_json):
    df = pd.read_json(io.StringIO(df_json))
    df["label_encoded"] = df["label"].map({"ham": 0, "spam": 1})

    X = df["message"]
    y = df["label_encoded"]

    vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
    X_vectorized = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y, test_size=0.2, random_state=42, stratify=y
    )

    model = MultinomialNB(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "vectorizer": vectorizer,
        "model": model,
        "y_test": y_test,
        "y_pred": y_pred,
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "vocab_size": X_vectorized.shape[1],
    }


def predict_message(text, vectorizer, model):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    return pred, prob


st.sidebar.title("🛡 SMS Spam Detection")
st.sidebar.markdown("**TF-IDF Vectorizer + Multinomial Naive Bayes**")
st.sidebar.divider()

st.sidebar.subheader("📁 Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload dataset SMS (.csv)", type=["csv", "txt"],
    help="Format: kolom 'label' (ham/spam) dan 'message', atau format SMSSpamCollection asli."
)

if uploaded_file is not None:
    try:
        df = load_uploaded_dataset(uploaded_file)
        st.sidebar.success(f"Dataset berhasil dimuat: {df.shape[0]} baris")
    except Exception as e:
        st.sidebar.error(f"Gagal membaca dataset: {e}")
        st.sidebar.info("Menggunakan dataset bawaan sebagai gantinya.")
        df = load_default_dataset()
else:
    df = load_default_dataset()
    st.sidebar.info("Menggunakan dataset bawaan (SMSSpamCollection, 5.574 pesan)")

df = df.dropna(subset=["label", "message"])
df = df[df["label"].isin(["ham", "spam"])].reset_index(drop=True)

# Latih model (cache berdasarkan isi dataset)
result = train_model(df.to_json())
vectorizer = result["vectorizer"]
model = result["model"]
y_test = result["y_test"]
y_pred = result["y_pred"]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

st.markdown(
    """
    <div style="background:linear-gradient(135deg,#0A1F44,#1B4DB3,#3E8EF0);
    padding:26px 28px;color:#FFFFFF;border-radius:6px;margin-bottom:10px;">
    <span style="font-size:11px;letter-spacing:3px;text-transform:uppercase;">Sistem deteksi pesan</span>
    <h1 style="margin:8px 0 0;font-weight:600;">🛡 SMS Spam Detection System</h1>
    <p style="margin-top:6px;font-size:14px;font-style:italic;">TF-IDF Vectorizer + Multinomial Naive Bayes</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Kartu metrik
m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", f"{acc*100:.2f}%")
m2.metric("Precision", f"{prec*100:.2f}%")
m3.metric("Recall", f"{rec*100:.2f}%")
m4.metric("F1-Score", f"{f1*100:.2f}%")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Dataset", "📈 Visualisasi", "✅ Evaluasi Model", "🔍 Prediksi SMS"]
)

# ---------------- TAB 1: DATASET ----------------
with tab1:
    st.subheader("Ringkasan Dataset")
    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah Data", df.shape[0])
    c2.metric("Jumlah Ham", int((df["label"] == "ham").sum()))
    c3.metric("Jumlah Spam", int((df["label"] == "spam").sum()))

    st.write("**Preview Data**")
    st.dataframe(df.head(10), use_container_width=True)

    st.write("**Missing Values**")
    st.dataframe(df.isnull().sum().rename("jumlah_missing"))

    colh, cols = st.columns(2)
    with colh:
        st.write("**Contoh SMS Ham**")
        st.dataframe(
            df[df["label"] == "ham"][["message"]].sample(
                min(5, (df["label"] == "ham").sum()), random_state=1
            ),
            use_container_width=True,
        )
    with cols:
        st.write("**Contoh SMS Spam**")
        st.dataframe(
            df[df["label"] == "spam"][["message"]].sample(
                min(5, (df["label"] == "spam").sum()), random_state=1
            ),
            use_container_width=True,
        )

with tab2:
    st.subheader("Visualisasi Data")

    colv1, colv2 = st.columns(2)

    with colv1:
        st.write("**Distribusi Label SMS (Ham vs Spam)**")
        fig, ax = plt.subplots(figsize=(5, 4))
        df["label"].value_counts().plot(
            kind="bar", color=["steelblue", "tomato"], edgecolor="black", ax=ax
        )
        ax.set_xlabel("Label")
        ax.set_ylabel("Jumlah")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        st.pyplot(fig)

    with colv2:
        st.write("**Hasil Evaluasi Model**")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
        values = [acc, prec, rec, f1]
        bars = ax2.bar(
            metrics_names, values,
            color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"], edgecolor="black"
        )
        ax2.set_ylim(0, 1.1)
        for bar, val in zip(bars, values):
            ax2.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.4f}", ha="center", fontsize=10
            )
        ax2.set_ylabel("Score")
        st.pyplot(fig2)

    st.write("**Confusion Matrix**")
    cm = confusion_matrix(y_test, y_pred)
    cm_flipped = cm[::-1, ::-1]
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm_flipped, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Spam", "Ham"], yticklabels=["Spam", "Ham"], ax=ax3
    )
    ax3.set_title("Confusion Matrix - Naive Bayes")
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")
    st.pyplot(fig3)

with tab3:
    st.subheader("Evaluasi Performa Model")

    st.write(f"- Jumlah fitur TF-IDF (vocabulary): **{result['vocab_size']}**")
    st.write(f"- Data latih: **{result['X_train_shape'][0]}** | Data uji: **{result['X_test_shape'][0]}**")

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Accuracy", f"{acc:.4f}")
    e2.metric("Precision", f"{prec:.4f}")
    e3.metric("Recall", f"{rec:.4f}")
    e4.metric("F1-Score", f"{f1:.4f}")

    st.write("**Classification Report**")
    report = classification_report(
        y_test, y_pred, target_names=["Ham", "Spam"], output_dict=True
    )
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

with tab4:
    st.subheader("Coba Klasifikasikan Pesan SMS")

    colp1, colp2 = st.columns(2)

    with colp1:
        st.write("**01 · Input SMS**")
        pilihan = st.selectbox("Contoh pesan:", list(CONTOH_PESAN.keys()))
        default_text = CONTOH_PESAN[pilihan] if pilihan != "Pilih contoh pesan" else ""
        text_input = st.text_area(
            "Masukkan isi SMS yang ingin dianalisis...",
            value=default_text, height=180
        )
        predict_btn = st.button("🔍 Analisis SMS", type="primary")

    with colp2:
        st.write("**02 · Prediction Result**")
        if predict_btn:
            if not text_input.strip():
                st.warning("Masukkan pesan SMS terlebih dahulu sebelum menjalankan analisis.")
            else:
                pred, prob = predict_message(text_input, vectorizer, model)
                p_ham = prob[0] * 100
                p_spam = prob[1] * 100
                confidence = max(p_ham, p_spam)

                if pred == 1:
                    st.error(f"🚨 **SPAM** — Confidence: {confidence:.1f}%")
                else:
                    st.success(f"✅ **HAM** — Confidence: {confidence:.1f}%")

                st.write("Probabilitas Spam")
                st.progress(int(p_spam))
                st.caption(f"{p_spam:.2f}%")

                st.write("Probabilitas Ham")
                st.progress(int(p_ham))
                st.caption(f"{p_ham:.2f}%")

                st.caption(
                    f"📏 Panjang: {len(text_input)} karakter | 🔤 Kata: {len(text_input.split())} kata"
                )
        else:
            st.info("Pilih contoh pesan atau ketik pesan SMS, lalu klik **Analisis SMS**.")

st.divider()
st.caption("SMS Spam Detection System · TF-IDF + Multinomial Naive Bayes · Dibuat untuk UAS Data Mining")