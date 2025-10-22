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
    # Standardize column names: strip
    df.columns = [c.strip() for c in df.columns]

    numeric_cols = []
    for c in df.columns:
        try:
            # Try to convert to numeric
            ser = pd.to_numeric(df[c], errors='coerce')
            # Criteria: sufficiently non-null numeric
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
    # Using Indonesian stopwords list for better relevance
    # Note: 'english' stop_words was used in the original code, switching to a basic Indonesian list
    # If the user's essay is in English, this needs adjustment, but assuming Indonesian.
    indonesian_stopwords = set(['yang', 'dan', 'di', 'ke', 'dari', 'tidak', 'dengan', 'saya', 'untuk', 'pada', 'adalah', 'ini', 'itu', 'sangat', 'agar', 'bisa', 'akan'])
    vec = CountVectorizer(stop_words=list(indonesian_stopwords), min_df=2)
    cleaned = text_series.fillna("").astype(str).str.lower()
    X = vec.fit_transform(cleaned)
    s = np.asarray(X.sum(axis=0)).ravel()
    terms = np.array(vec.get_feature_names_out())
    top_idx = np.argsort(s)[::-1][:n]
    return pd.DataFrame({"word": terms[top_idx], "count": s[top_idx]})


# ---------- Layout & Data Loading (Diubah untuk Auto-Load) ----------
st.sidebar.title("Kontrol Dashboard & Filter")

# --- Bagian Data Loading Diubah ---
# Catatan Penting: Agar ini berfungsi di Streamlit Cloud, file Responden.xlsx
# HARUS sudah diunggah ke repositori GitHub BERSAMA app.py.

default_filename = "Responden.xlsx"
df = None
load_status = st.sidebar.empty() # Placeholder untuk pesan loading

try:
    load_status.info(f"Memuat data dari {default_filename}...")
    df = load_data_from_path(default_filename)
    load_status.success("Data berhasil dimuat!")
except Exception as e:
    # Jika gagal (misalnya file tidak ada di repo), tampilkan error dan hentikan
    load_status.error(f"Gagal memuat data utama: {default_filename}. Pastikan file ini ada di repositori GitHub Anda. Error: {e}")
    st.stop() # Hentikan proses jika data utama tidak ditemukan.

# Preprocess
df_raw = df.copy()
df, numeric_cols = preprocess_df(df_raw)


# ---------- Global Filtering in Sidebar (UNCHANGED) ----------

st.sidebar.markdown("---")
st.sidebar.subheader("Filter Data Global")

# List of columns to be used as filters
filter_cols_mapping = {
    "Fakultas": "Fakultas",
    "Program Studi": "Program Studi",
    "Nama Responden": "Nama",
    "Nomor Pokok Mahasiswa (NPM)": "NPM",
    "Timestamp (Waktu Pengisian)": "Timestamp"
}

# --- Apply Filters ---
df_filtered = df.copy()
initial_rows = len(df_filtered)

for label, col_name in filter_cols_mapping.items():
    if col_name in df_filtered.columns:
        # Get unique, non-null sorted values for the multiselect
        options = df_filtered[col_name].dropna().unique().tolist()
        options.sort()
        
        # Display multiselect filter in sidebar
        selected_values = st.sidebar.multiselect(
            f"Pilih {label}:",
            options=options,
            default=options # Default to all selected
        )
        
        # Apply filter to the DataFrame
        if selected_values:
            df_filtered = df_filtered[df_filtered[col_name].isin(selected_values)]
            
# Use the filtered DataFrame for all subsequent analysis
df = df_filtered
final_rows = len(df)

st.sidebar.info(f"Data tersaring: {final_rows} dari {initial_rows} baris.")

if final_rows == 0:
    st.error("Semua data terfilter habis. Sesuaikan pilihan filter Anda.")
    st.stop()
    
# Re-preprocess to update numeric_cols based on filtered data (important for describe())
# Though preprocess_df doesn't rely on data subsetting, it's safer to ensure consistency
df, numeric_cols = preprocess_df(df_raw) # Re-run with original data structure, then use df_filtered
# Since df is now df_filtered, we re-run preprocess on the original df_raw only to get ALL numeric columns consistently.

# ------------------ Navigation Tabs (UNCHANGED) ------------------

# Define all app modes (pages)
APP_MODES = ["Overview", "Deskriptif", "Analisis Kunci (Efektivitas Parkir)", "Korelasi", "Regresi Linear Berganda", "Teks (essay)", "Download & Petunjuk"]

# Create tabs for navigation instead of sidebar selectbox
tabs = st.tabs(APP_MODES)

# Map tab index to app_mode
for i, tab in enumerate(tabs):
    with tab:
        app_mode = APP_MODES[i]

        # ------------------ Overview ------------------
        if app_mode == "Overview":
            st.title("Overview — Dashboard Analisis Parkir")
            st.markdown("**Deskripsi singkat:** Dashboard ini menampilkan analisis eksploratif data survei preferensi dan kondisi lahan parkir di kampus UPN 'Veteran' Jawa Timur. \nFokus utama adalah pada **efektivitas ketersediaan dan pelayanan**. Gunakan tab di atas untuk pindah halaman.\n")
            
            st.subheader("Informasi dataset")
            st.write(f"Jumlah baris: **{df.shape[0]:,}** — Jumlah kolom: **{df.shape[1]:,}**")
            st.write("Daftar kolom:")
            st.dataframe(pd.DataFrame({'kolom': df.columns}))

            st.subheader("Preview data (5 baris pertama)")
            st.dataframe(df.head())

            st.subheader("Ringkasan singkat tipe data")
            st.dataframe(df.dtypes.to_frame(name='Tipe Data'))

        # ------------------ Descriptive ------------------
        elif app_mode == "Deskriptif":
            st.title("Analisis Deskriptif & Visualisasi Variatif")
            st.markdown("Halaman ini menampilkan statistik dasar dan distribusi data kategorikal serta numerik.")

            # Section 1: Demografi
            st.header("1. Distribusi Demografi")
            with st.expander("Pilih kolom demografis (kategorikal) untuk breakdown"):
                cat_cols = [c for c in df.columns if df[c].dtype == 'object' or df[c].dtype.name == 'category']
                # Remove filter columns from this selection to avoid redundancy
                filter_cols = list(filter_cols_mapping.values())
                cat_cols = [c for c in cat_cols if c not in filter_cols] 
                
                sel_cat = st.multiselect('Pilih kategori (mis. jenis kendaraan, angkatan)', cat_cols, default=cat_cols[:2])

            col_dem1, col_dem2 = st.columns(2)
            for i, c in enumerate(sel_cat):
                if i % 2 == 0:
                    col = col_dem1
                else:
                    col = col_dem2

                with col:
                    # Use Pie chart for top categories, Bar chart for detailed distribution
                    counts = df[c].value_counts().reset_index()
                    counts.columns = [c, 'count']
                    if len(counts) <= 6:
                         fig = px.pie(counts, names=c, values='count', title=f"Proporsi: {c}")
                         fig.update_traces(textposition='inside', textinfo='percent+label')
                    else:
                         fig = px.bar(counts.head(10), x=c, y='count', title=f"Top 10 Distribusi: {c}")

                    st.plotly_chart(fig, use_container_width=True)


            # Section 2: Analisis Skor Likert
            st.header("2. Statistik & Perbandingan Skor Likert (Numerik)")
            if len(numeric_cols)>0:
                col_stat1, col_stat2 = st.columns([1, 2])
                with col_stat1:
                    st.subheader("Ringkasan Skor")
                    stats = df[numeric_cols].describe().T[['count', 'mean', 'std', 'min', 'max']].sort_values(by='mean', ascending=False)
                    st.dataframe(stats.style.background_gradient(cmap='RdYlGn', subset=['mean']), use_container_width=True)

                with col_stat2:
                    st.subheader("Visualisasi Rata-Rata Skor")
                    # Horizontal Bar Chart for mean comparison
                    mean_df = df[numeric_cols].mean().sort_values(ascending=True).to_frame(name='Rata-Rata Skor')
                    mean_df = mean_df.reset_index().rename(columns={'index': 'Variabel'})
                    fig_mean = px.bar(mean_df, x='Rata-Rata Skor', y='Variabel', orientation='h',
                                      color='Rata-Rata Skor', color_continuous_scale=px.colors.sequential.Inferno,
                                      title="Perbandingan Rata-Rata Skor Likert")
                    fig_mean.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig_mean, use_container_width=True)
            else:
                st.warning("Tidak ada kolom numerik (Likert) yang terdeteksi untuk analisis skor.")

        # ------------------ Effectiveness Analysis (New Page) ------------------
        elif app_mode == "Analisis Kunci (Efektivitas Parkir)":
            st.title("Analisis Kunci: Efektivitas Ketersediaan Lahan Parkir")
            st.markdown("Halaman ini menyajikan sintesis temuan yang secara langsung menjawab tujuan penelitian mengenai efektivitas lahan parkir.")

            if len(numeric_cols) < 2:
                 st.warning("Data numerik tidak memadai. Pastikan Anda meng-upload data survei yang mengandung skor Likert.")
                 st.stop()

            # Calculate key metrics
            mean_scores = df[numeric_cols].mean().sort_values()
            overall_mean = mean_scores.mean()

            col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
            with col_kpi1:
                st.metric(label="Rata-Rata Skor Efektivitas Keseluruhan", value=f"{overall_mean:.2f}",
                          delta=f"{(overall_mean - 3.0)*100/3:.2f}% dari skala maks (jika skala 1-5)") # Assuming 3 is neutral
            with col_kpi2:
                top_complaint = mean_scores.index[0]
                st.metric(label="Poin Paling Rentan/Butuh Perbaikan", value=f"{top_complaint}", delta=f"Skor: {mean_scores.iloc[0]:.2f}", delta_color="inverse")
            with col_kpi3:
                top_satisfaction = mean_scores.index[-1]
                st.metric(label="Poin Paling Memuaskan", value=f"{top_satisfaction}", delta=f"Skor: {mean_scores.iloc[-1]:.2f}")

            st.markdown("---")

            st.subheader("Perbandingan Antar Variabel (Poin Kekuatan vs Kelemahan)")

            col_eff1, col_eff2 = st.columns(2)
            with col_eff1:
                st.markdown("#### 3 Poin Kritis (Skor Terendah)")
                critical_df = mean_scores.head(3).to_frame(name='Skor Rata-Rata')
                fig_critical = px.bar(critical_df, x='Skor Rata-Rata', y=critical_df.index, orientation='h',
                                      color='Skor Rata-Rata', color_continuous_scale=px.colors.sequential.Reds,
                                      title="Aspek dengan Efektivitas Terendah")
                st.plotly_chart(fig_critical, use_container_width=True)

            with col_eff2:
                st.markdown("#### Perbandingan Skor Likert Berdasarkan Demografi")
                # Find a suitable categorical column (e.g., 'Fakultas') for breakdown
                cat_cols = [c for c in df.columns if df[c].dtype == 'object' or df[c].dtype.name == 'category']
                
                if cat_cols:
                    breakdown_col = st.selectbox("Pilih Kategori Pembanding:", options=cat_cols, index=min(len(cat_cols)-1, 0))
                    score_col = st.selectbox("Pilih Variabel Likert:", options=numeric_cols, index=0)

                    grouped_mean = df.groupby(breakdown_col)[score_col].mean().sort_values(ascending=False).reset_index()
                    fig_breakdown = px.bar(grouped_mean, x=breakdown_col, y=score_col,
                                           title=f"Skor {score_col} Berdasarkan {breakdown_col}",
                                           color=score_col, color_continuous_scale=px.colors.sequential.Bluyl)
                    st.plotly_chart(fig_breakdown, use_container_width=True)
                else:
                    st.info("Tidak ada kolom kategorikal yang terdeteksi untuk perbandingan demografi.")

            st.markdown("---")
            st.subheader("Kesimpulan Analisis Efektivitas Ketersediaan")
            st.info("Berdasarkan data yang tersedia, efektivitas ketersediaan lahan parkir di UPN 'Veteran' Jawa Timur dapat disimpulkan melalui perbandingan skor rata-rata. Poin-poin dengan skor terendah (misalnya, 'Ketersediaan saat jam sibuk' atau 'Kemudahan mencari tempat') menunjukkan prioritas utama untuk perbaikan, sedangkan skor tertinggi mencerminkan area yang sudah berjalan efektif.")

        # ------------------ Correlation ------------------
        elif app_mode == "Korelasi":
            st.title("Analisis Korelasi")
            st.markdown("Menampilkan korelasi antar variabel numerik (Likert). Gunakan analisis ini untuk melihat hubungan linier sederhana antar skor. Nilai mendekati **+1** berarti korelasi positif kuat, **-1** negatif kuat.")

            if len(numeric_cols) < 2:
                st.warning("Tidak cukup variabel numerik untuk analisis korelasi.")
            else:
                cols_for_corr = st.multiselect("Pilih variabel numerik untuk korelasi", numeric_cols, default=numeric_cols[:8])
                if len(cols_for_corr) >= 2:
                    corr = df[cols_for_corr].corr()
                    fig = px.imshow(corr, text_auto=".2f", title='Matriks Korelasi (Pearson)',
                                    color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
                    st.plotly_chart(fig, use_container_width=True)

                    st.subheader("Tabel korelasi")
                    st.dataframe(corr.style.background_gradient(cmap='RdBu_r', vmin=-1, vmax=1), use_container_width=True)

                    if st.checkbox("Tampilkan scatter matrix (pairwise)"):
                        fig2 = px.scatter_matrix(df[cols_for_corr].dropna(), dimensions=cols_for_corr)
                        st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("Pilih minimal 2 variabel untuk melihat korelasi.")

        # ------------------ Regression ------------------
        elif app_mode == "Regresi Linear Berganda":
            st.title("Regresi Linear Berganda")
            st.markdown("Pilih satu variabel dependen (Y, misal: 'Kepuasan Overall') dan beberapa variabel independen (X) numerik. Ini berguna untuk memprediksi variabel Y dari kombinasi variabel X.")

            if len(numeric_cols) < 2:
                st.warning("Tidak cukup variabel numerik untuk regresi.")
            else:
                dep = st.selectbox("Pilih variabel dependen (Y)", options=numeric_cols)
                indep = st.multiselect("Pilih variabel independen (X) — minimal 1", options=[c for c in numeric_cols if c!=dep])
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

                    st.subheader("Hasil model (sklearn) — Uji Prediksi")
                    col_res1, col_res2 = st.columns(2)
                    with col_res1:
                        st.write(f"R^2 (koefisien determinasi) di Test Set: **{r2_score(y_test, y_pred):.4f}**")
                        st.write(f"MSE (Mean Squared Error) di Test Set: **{mean_squared_error(y_test, y_pred):.4f}**")
                    with col_res2:
                        st.write(f"Intercept: {lr.intercept_:.4f}")
                        coef = pd.Series(lr.coef_, index=indep).to_frame('Koefisien Regresi')
                        st.dataframe(coef, use_container_width=True)

                    # statsmodels OLS for summary table
                    if st.checkbox("Tampilkan Ringkasan Statistik Penuh (statsmodels OLS)"):
                        X_const = sm.add_constant(sub[indep])
                        model = sm.OLS(sub[dep], X_const, missing='drop').fit()
                        st.text(model.summary())

                    # Plot actual vs predicted
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=y_test, mode='markers', name='Actual', marker={'opacity': 0.7}))
                    fig.add_trace(go.Scatter(y=y_pred, mode='markers', name='Predicted', marker={'opacity': 0.7}))
                    fig.update_layout(title='Actual vs Predicted (Test Set)', xaxis_title='Index Observasi', yaxis_title=dep)
                    st.plotly_chart(fig, use_container_width=True)

                else:
                    st.info("Pilih minimal 1 variabel independen untuk memodelkan.")

        # ------------------ Text analysis (essay) ------------------
        elif app_mode == "Teks (essay)":
            st.title("Analisis Jawaban Essay/Open-Ended")
            st.markdown("Analisis frekuensi kata membantu merangkum keluhan, kendala, atau saran yang paling sering diungkapkan responden.")

            text_cols = [c for c in df.columns if df[c].dtype == 'object']
            sel_text = st.multiselect("Pilih kolom essay/text", text_cols, default=[c for c in text_cols if 'kendala' in c.lower() or 'solusi' in c.lower()][:1])

            if sel_text:
                for c in sel_text:
                    st.header(f"Ringkasan Kata Kunci — {c}")
                    tw = top_words(df[c].astype(str), n=30)
                    
                    # Use Plotly Bar chart for a better visual representation of frequency
                    fig = px.bar(tw.sort_values(by='count', ascending=True), x='count', y='word', orientation='h',
                                 title=f'Top 30 Kata Kunci di Kolom: {c}',
                                 color='count', color_continuous_scale=px.colors.sequential.Viridis)
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)

                    st.subheader("Contoh Jawaban Acak (5 baris)")
                    st.dataframe(df[c].dropna().sample(min(5, df[c].dropna().shape[0])).to_frame(name=c), use_container_width=True)
            else:
                st.info("Pilih minimal satu kolom teks untuk dianalisis.")

        # ------------------ Download & Petunjuk ------------------
        elif app_mode == "Download & Petunjuk":
            st.title("Petunjuk Deploy & Download")
            st.markdown(
                "1. Pastikan file `app.py` dan `requirements.txt` (jika ada) ada di repository GitHub kamu.\n"
                "2. Jika ingin data disertakan di repo, tambahkan `Responden.xlsx` ke repo agar Streamlit Cloud dapat membukanya.\n"
                "3. Di Streamlit Cloud (share.streamlit.io) buat New app → konek ke repo → pilih `app.py`.\n"
                "4. Jika terjadi error, buka Logs di Streamlit Cloud untuk melihat pesan kesalahan."
            )

            st.subheader("Download sample data (jika mau)")
            st.write("Gunakan tombol di bawah ini untuk mendownload ringkasan data sebagai CSV (hasil preprocess).")
            # Ensure numeric_cols is up-to-date and not empty before trying to use it
            if len(numeric_cols) > 0:
                st.download_button("Download ringkasan numeric", df[numeric_cols].describe().to_csv().encode('utf-8'), file_name='ringkasan_numeric.csv')
            else:
                 st.warning("Tidak ada data numerik untuk diunduh.")

# EOF
