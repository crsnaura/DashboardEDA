# Streamlit dashboard untuk Analisis Eksploratif: Parkir UPN "Veteran" Jawa Timur
# Template interaktif & modern — siap di-deploy ke Streamlit Cloud

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
from sklearn.feature_extraction.text import CountVectorizer

st.set_page_config(page_title="Dashboard Parkir UPN - EDA", layout="wide")

# ---------- Helper functions ----------
@st.cache_data
def load_data_from_path(path: str):
    # Load Excel/CSV file with automatic detection
    if path.lower().endswith(".xlsx") or path.lower().endswith('.xls'):
        return pd.read_excel(path)
    elif path.lower().endswith('.csv'):
        return pd.read_csv(path)
    else:
        raise ValueError("Unsupported file type. Upload .xlsx, .xls or .csv")


def preprocess_df(df: pd.DataFrame):
    df = df.copy()
    # Standardize column names: strip and lower
    df.columns = [c.strip() for c in df.columns]

    # Try to detect Likert numeric columns: those that contain only integers 1-5 or convertible
    numeric_cols = []
    for c in df.columns:
        try:
            ser = pd.to_numeric(df[c], errors='coerce')
            # consider numeric if more than half non-null numeric and values inside 1-5 for Likert
            if ser.notna().sum() >= max(5, int(0.3 * len(ser))):
                numeric_cols.append(c)
        except Exception:
            continue

    # Simple conversion for numeric-like columns
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    return df, numeric_cols


@st.cache_data
def top_words(text_series, n=30):
    vec = CountVectorizer(stop_words='english')
    cleaned = text_series.fillna("").astype(str)
    X = vec.fit_transform(cleaned)
    s = np.asarray(X.sum(axis=0)).ravel()
    terms = np.array(vec.get_feature_names_out())
    top_idx = np.argsort(s)[::-1][:n]
    return pd.DataFrame({"word": terms[top_idx], "count": s[top_idx]})


# ---------- Layout ----------
st.sidebar.title("Kontrol Dashboard")
app_mode = st.sidebar.selectbox("Pilih Halaman:", ["Overview", "Deskriptif", "Korelasi", "Regresi Linear Berganda", "Teks (essay)", "Download & Petunjuk"]) 

# Upload / load data
st.sidebar.markdown("---")
st.sidebar.subheader("Data input")
uploaded = st.sidebar.file_uploader("Upload file Excel (.xlsx) atau CSV jika ada", type=['xlsx', 'xls', 'csv'])
use_example = st.sidebar.checkbox("Gunakan file contoh: Responden.xlsx (jika ada di repo)")

# default filename to try (when deploying, pastikan file ada di repo)
default_filename = "Responden.xlsx"

if uploaded is not None:
    try:
        df = load_data_from_path(uploaded.name)  # Note: streamlit passes a file-like object; load_data handles paths
        uploaded.seek(0)
        df = pd.read_excel(uploaded) if uploaded.name.lower().endswith(('xls','xlsx')) else pd.read_csv(uploaded)
    except Exception:
        st.error("Gagal memuat file yang diupload. Pastikan format .xlsx atau .csv dan kolom rapi.")
        st.stop()

elif use_example:
    try:
        df = load_data_from_path(default_filename)
    except Exception:
        st.error(f"Tidak menemukan {default_filename} di repo. Upload file lewat sidebar atau pastikan nama file benar.")
        st.stop()
else:
    st.info("Upload file data kamu di sidebar atau centang 'Gunakan file contoh' jika file sudah ada di repo.")
    st.stop()

# Preprocess
df_raw = df.copy()
df, numeric_cols = preprocess_df(df_raw)

# ------------------ Overview ------------------
if app_mode == "Overview":
    st.title("Overview — Dashboard Analisis Parkir")
    st.markdown("**Deskripsi singkat:** Dashboard ini menampilkan analisis eksploratif data survei preferensi dan kondisi lahan parkir di kampus UPN 'Veteran' Jawa Timur. \nGunakan sidebar untuk pindah halaman dan meng-upload dataset kamu.\n")

    st.subheader("Informasi dataset")
    st.write(f"Jumlah baris: **{df.shape[0]:,}** — Jumlah kolom: **{df.shape[1]:,}**")
    st.write("Daftar kolom:")
    st.dataframe(pd.DataFrame({'kolom': df.columns}))

    st.subheader("Preview data (5 baris pertama)")
    st.dataframe(df.head())

    st.subheader("Ringkasan singkat tipe data")
    st.write(df.dtypes)

# ------------------ Descriptive ------------------
elif app_mode == "Deskriptif":
    st.title("Analisis Deskriptif")

    with st.expander("Pilih kolom demografis untuk breakdown"):
        cat_cols = [c for c in df.columns if df[c].dtype == 'object' or df[c].dtype.name == 'category']
        sel_cat = st.multiselect('Pilih kategori (mis. fakultas, program studi, kendaraan)', cat_cols, default=cat_cols[:2])

    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Distribusi kategori")
        for c in sel_cat:
            fig = px.histogram(df, x=c, title=f"Distribusi: {c}", marginal='box')
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Statistik numerik (Kolom Likert)")
        if len(numeric_cols)>0:
            stats = df[numeric_cols].describe().T
            st.dataframe(stats)
        else:
            st.write("Tidak ada kolom numerik yang terdeteksi.")

    st.subheader("Jenis kendaraan")
    if 'Kendaraan apa yang biasanya Anda gunakan untuk ke kampus?' in df.columns:
        colname = 'Kendaraan apa yang biasanya Anda gunakan untuk ke kampus?'
        fig = px.pie(df, names=colname, title='Proporsi jenis kendaraan')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('Kolom jenis kendaraan tidak ditemukan dengan nama persis. Pilih di sidebar Deskriptif.')

# ------------------ Correlation ------------------
elif app_mode == "Korelasi":
    st.title("Analisis Korelasi")
    st.markdown("Menampilkan korelasi antar variabel numerik (Likert). Gunakan analisis ini untuk melihat hubungan linier sederhana antar skor.")

    if len(numeric_cols) < 2:
        st.warning("Tidak cukup variabel numerik untuk analisis korelasi.")
    else:
        cols_for_corr = st.multiselect("Pilih variabel numerik untuk korelasi", numeric_cols, default=numeric_cols[:8])
        if len(cols_for_corr) >= 2:
            corr = df[cols_for_corr].corr()
            fig = px.imshow(corr, text_auto=True, title='Matriks Korelasi (Pearson)')
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Tabel korelasi")
            st.dataframe(corr)

            # Show scatter matrix for selected
            if st.checkbox("Tampilkan scatter matrix (pairwise)"):
                fig2 = px.scatter_matrix(df[cols_for_corr].dropna(), dimensions=cols_for_corr)
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Pilih minimal 2 variabel untuk melihat korelasi.")

# ------------------ Regression ------------------
elif app_mode == "Regresi Linear Berganda":
    st.title("Regresi Linear Berganda")
    st.markdown("Pilih satu variabel dependen (numerik) dan beberapa variabel independen numerik. Disarankan minimal 3 variabel independen sesuai kebutuhan tugas.")

    if len(numeric_cols) < 2:
        st.warning("Tidak cukup variabel numerik untuk regresi.")
    else:
        dep = st.selectbox("Pilih variabel dependen (Y)", options=numeric_cols)
        indep = st.multiselect("Pilih variabel independen (X) — minimal 3", options=[c for c in numeric_cols if c!=dep])
        test_size = st.slider("Proporsi data test", 0.1, 0.5, 0.25)

        if len(indep) >= 1:
            # Prepare data
            sub = df[[dep] + indep].dropna()
            X = sub[indep].values
            y = sub[dep].values

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            # Fit sklearn linear regression
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_test)

            st.subheader("Hasil model (sklearn)")
            coef = pd.Series(lr.coef_, index=indep)
            st.write("Koefisien:")
            st.dataframe(coef.rename('coef').to_frame())
            st.write(f"Intercept: {lr.intercept_:.4f}")
            st.write(f"R^2 (test): {r2_score(y_test, y_pred):.4f}")
            st.write(f"MSE (test): {mean_squared_error(y_test, y_pred):.4f}")

            # statsmodels OLS untuk ringkasan jika user mau
            if st.checkbox("Tampilkan ringkasan statistik (statsmodels OLS)"):
                X_const = sm.add_constant(sub[indep])
                model = sm.OLS(sub[dep], X_const, missing='drop').fit()
                st.text(model.summary())

            # Plot actual vs predicted
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=y_test, mode='markers', name='Actual'))
            fig.add_trace(go.Scatter(y=y_pred, mode='markers', name='Predicted'))
            fig.update_layout(title='Actual vs Predicted (test set)', xaxis_title='Index', yaxis_title=dep)
            st.plotly_chart(fig, use_container_width=True)

            # Guidance re: minimal predictors
            if len(indep) < 3:
                st.warning("Disarankan memilih setidaknya 3 variabel independen jika tugas mensyaratkan >2 X.")
        else:
            st.info("Pilih minimal 1 variabel independen untuk memodelkan. Untuk tugas, pilih 3 atau lebih.")

# ------------------ Text analysis (essay) ------------------
elif app_mode == "Teks (essay)":
    st.title("Analisis Jawaban Essay")
    st.markdown("Analisis sederhana berupa frekuensi kata (top words) dan visualisasi. Ini membantu merangkum keluhan/kendala/solusi dari responden.")

    text_cols = [c for c in df.columns if df[c].dtype == 'object']
    sel_text = st.multiselect("Pilih kolom essay/text", text_cols, default=[c for c in text_cols if 'kendala' in c.lower() or 'solusi' in c.lower()][:1])

    if sel_text:
        for c in sel_text:
            st.subheader(f"Top words — {c}")
            tw = top_words(df[c].astype(str), n=30)
            fig = px.bar(tw, x='word', y='count', title=f'Top kata di kolom: {c}')
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Contoh jawaban (5 baris acak)")
            st.write(df[c].dropna().sample(min(5, df[c].dropna().shape[0])))
    else:
        st.info("Pilih minimal satu kolom teks untuk dianalisis.")

# ------------------ Download & Petunjuk ------------------
elif app_mode == "Download & Petunjuk":
    st.title("Petunjuk Deploy & Download")
    st.markdown(
        "1. Pastikan file `app.py` dan `requirements.txt` ada di repository GitHub kamu.\n"
        "2. Jika ingin data disertakan di repo, tambahkan `Responden.xlsx` ke repo agar Streamlit Cloud dapat membukanya.\n"
        "3. Di Streamlit Cloud (share.streamlit.io) buat New app → konek ke repo → pilih `app.py`.\n"
        "4. Jika terjadi error, buka Logs di Streamlit Cloud untuk melihat pesan kesalahan."
    )

    st.subheader("Download sample data (jika mau)")
    st.write("Gunakan tombol di bawah ini untuk mendownload ringkasan data sebagai CSV (hasil preprocess).")
    st.download_button("Download ringkasan numeric", df[numeric_cols].describe().to_csv().encode('utf-8'), file_name='ringkasan_numeric.csv')


# EOF
