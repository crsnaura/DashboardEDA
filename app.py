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
    # Standardize column names: strip whitespace
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

    # Identify categorical columns (excluding identifiers and timestamps)
    # Categorical are non-numeric types, or numeric types with very few unique values (though we primarily focus on object types here)
    # The actual dropping is now done BEFORE calling preprocess_df, but we keep this filter for robust categorical list generation.
    identifier_keywords = ['nama', 'npm', 'timestamp'] 
    categorical_cols = [c for c in df.columns if c not in numeric_cols and (df[c].dtype == 'object' or df[c].nunique() < 10)]
    # Filter out columns that are likely identifiers or timestamps
    categorical_cols = [c for c in categorical_cols if not any(kw in c.lower() for kw in identifier_keywords)]

    return df, numeric_cols, categorical_cols


@st.cache_data
def top_words(text_series, n=30):
    indonesian_stopwords = set(['yang', 'dan', 'di', 'ke', 'dari', 'tidak', 'dengan', 'saya', 'untuk', 'pada', 'adalah', 'ini', 'itu', 'sangat', 'agar', 'bisa', 'akan', 'juga', 'dalam', 'mereka'])
    vec = CountVectorizer(stop_words=list(indonesian_stopwords), min_df=2)
    cleaned = text_series.fillna("").astype(str).str.lower()
    X = vec.fit_transform(cleaned)
    s = np.asarray(X.sum(axis=0)).ravel()
    terms = np.array(vec.get_feature_names_out())
    top_idx = np.argsort(s)[::-1][:n]
    return pd.DataFrame({"word": terms[top_idx], "count": s[top_idx]})


# ---------- Layout & Data Loading ----------
st.sidebar.title("Kontrol Dashboard & Filter")

# --- Bagian Data Loading ---
default_filename = "Responden.xlsx"
df = None
load_status = st.sidebar.empty()

try:
    load_status.info(f"Memuat data dari {default_filename}...")
    df = load_data_from_path(default_filename)
    load_status.success("Data berhasil dimuat!")
except Exception as e:
    load_status.error(f"Gagal memuat data utama: {default_filename}. Pastikan file ini ada. Error: {e}")
    st.stop()

# -------------------------------------------------------------
# --- PERBAIKAN MODIFIKASI: Hapus Kolom Identitas/Timestamp ---
# -------------------------------------------------------------
df_raw = df.copy()

# 1. Standardize column names for robust searching
original_columns = df_raw.columns.tolist()
standardized_columns = [c.strip().lower().replace(' ', '_') for c in original_columns]
df_raw.columns = original_columns # Keep original names for display, but use list for mapping

# 2. Define keywords to drop
keywords_to_drop = ['timestamp', 'nama', 'npm']
columns_to_drop = []

# 3. Identify and collect columns to drop based on original names and standardized match
for original_col_name, std_col_name in zip(original_columns, standardized_columns):
    if any(kw in std_col_name for kw in keywords_to_drop):
        columns_to_drop.append(original_col_name)
            
# 4. Drop the identified columns
df_raw = df_raw.drop(columns=columns_to_drop, errors='ignore')
if columns_to_drop:
    st.sidebar.warning(f"Kolom ID/Timestamp Dihapus: {', '.join(columns_to_drop)}")
else:
    st.sidebar.info("Tidak ada kolom ID/Timestamp yang terdeteksi untuk dihapus.")
# -------------------------------------------------------------
# --- AKHIR PERBAIKAN MODIFIKASI ---
# -------------------------------------------------------------

# Preprocess
df, numeric_cols, categorical_cols = preprocess_df(df_raw)


# ---------- Global Filtering in Sidebar (UNCHANGED) ----------

st.sidebar.markdown("---")
st.sidebar.subheader("Filter Data Global")

# List of columns to be used as filters
filter_cols_mapping = {
    "Fakultas": "Fakultas",
    "Program Studi": "Program Studi"
    # Other filter columns can be added here if needed
}

# --- Apply Filters ---
df_filtered = df.copy()
initial_rows = len(df_filtered)

for label, col_name in filter_cols_mapping.items():
    if col_name in df_filtered.columns:
        options = df_filtered[col_name].dropna().unique().tolist()
        options.sort()
        
        selected_values = st.sidebar.multiselect(
            f"Pilih {label}:",
            options=options,
            default=options
        )
        
        if selected_values:
            df_filtered = df_filtered[df_filtered[col_name].isin(selected_values)]
            
df = df_filtered
final_rows = len(df)

st.sidebar.info(f"Data tersaring: {final_rows} dari {initial_rows} baris.")

if final_rows == 0:
    st.error("Semua data terfilter habis. Sesuaikan pilihan filter Anda.")
    st.stop()
    
# Re-run preprocess on the filtered data to update numeric/categorical lists based on subset
df, numeric_cols, categorical_cols = preprocess_df(df) 


# ------------------ TOP HEADER/JUDUL DASHBOARD (Kembali ke Lahan Parkir) ------------------

UPN_LOGO_URL = "https://upnjatim.ac.id/wp-content/uploads/2025/05/cropped-logo-1.png"
PARKING_ICON_URL = "https://png.pngtree.com/png-clipart/20230414/original/pngtree-car-parking-sign-design-template-png-image_9055938.png" # Contoh ikon mobil/parkir

st.markdown("""
    <style>
    /* Mengatur tata letak Judul Utama agar rapi dan terpusat */
    .dashboard-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 20px 0;
        margin-bottom: 20px;
        border-bottom: 2px solid #333333;
        
    }
    .header-title {
        text-align: center;
        flex-grow: 1;
    }
    .header-title h1 {
        font-size: 2.5em;
        font-weight: 700;
        margin: 0;
        color: #1E90FF;
    }
    .header-title h3 {
        font-size: 1.2em;
        font-weight: 400;
        color: #AAAAAA;
        margin-top: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

col_logo_left, col_title, col_logo_right = st.columns([1, 4, 1])

with col_logo_left:
    st.image(UPN_LOGO_URL, use_container_width=True)

with col_title:
    st.markdown(f"""
    <div class="header-title">
        <h1>Dashboard Analisis Lahan Parkir</h1>
        <h3>Keefektifan dan Ketersediaan di UPN "Veteran" Jawa Timur</h3>
    </div>
    """, unsafe_allow_html=True)

with col_logo_right:
    st.image(PARKING_ICON_URL, use_container_width=True)

st.markdown("---")

# ------------------ Navigation Tabs (Menjaga nama Analisis Kunci) ------------------

# !!! PERBAIKAN PENTING: Tambahkan emoji ke dalam list APP_MODES
# agar namanya persis sama dengan yang ada di conditional check (if/elif)
APP_MODES = [
    "Overview💌", 
    "🗂️Deskriptif", 
    "Analisis Kunci (Efektivitas Parkir)📩", 
    "📊Korelasi", 
    "📈Regresi Linear Berganda", 
    "📉Teks (essay)", 
    "📩Download & Petunjuk📩"
]

# Mengganti sidebar selectbox dengan tabs
tabs = st.tabs(APP_MODES)

# Map tab index to app_mode
for i, tab in enumerate(tabs):
    with tab:
        app_mode = APP_MODES[i]

        # ------------------ Overview (REVISI FINAL: Teks Naratif + Semua Data Preview) ------------------
        if app_mode == "Overview💌":
            st.title("🔥Selamat Datang di Dashboard Analisis Parkir UPN🙌")
            
            # --- 1. Teks Naratif Awal (Membuat Menarik) ---
            st.markdown("""
            <div style="background-color: #F0F8FF; padding: 25px; border-radius: 12px; border-left: 6px solid #1E90FF; margin-bottom: 25px; box-shadow: 2px 2px 8px rgba(0,0,0,0.1);">
                <p style="font-size: 1.15em; font-weight: 500; margin-bottom: 15px; color: #1E90FF;">
                    <b>Eksplorasi Data Analisis (EDA) Parkir UPN "Veteran" Jawa Timur</b>
                </p>
                <p style="font-size: 1.0em; color: #333;">
                    Dashboard ini didedikasikan untuk melakukan analisis mendalam terhadap hasil survei ketersediaan dan efektivitas lahan parkir. 
                    Tujuannya adalah untuk memberikan <b>wawasan berbasis data</b> yang konkret bagi pengambilan keputusan strategis terkait 
                    manajemen fasilitas kampus. Bagian ini menyajikan gambaran cepat mengenai struktur data dan ringkasan awal.
                </p>
                <p style="font-size: 0.9em; color: #666; margin-top: 10px;">
                    Gunkaan filter di sidebar dan jelajahi tab "Analisis Kunci" untuk temuan utama.
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Menampilkan Statistik Kunci (Total Responden)
            st.info(f"🤷‍♀️Total Responden yang Sedang Dianalisis Saat Ini: **{df.shape[0]:,} orang**.🦋")
            st.markdown("---")
            
            # 2. Preview Data
            st.subheader("🗂️Preview Data Keseluruhan (5 Baris Pertama)")
            st.markdown("Contoh baris data untuk memverifikasi format dan isinya (kolom ID seperti NPM, Nama, dan Timestamp sudah dihapus).")
            st.dataframe(df.head(), use_container_width=True)
            
            # 3. Ringkasan Tipe Data (Lebih Jelas)
            st.subheader("💌Ringkasan Tipe Data & Klasifikasi Analisis")
            
            col_tip1, col_tip2 = st.columns(2)
            
            with col_tip1:
                st.markdown("#### Preview Kolom Numerik (Skor Likert / Kuantitatif)🔢")
                if numeric_cols:
                    st.write(f"Total **{len(numeric_cols)}** Variabel:")
                    # MENAMPILKAN DATA NUMERIK SECARA LANGSUNG
                    st.dataframe(df[numeric_cols].head(5), use_container_width=True)
                else:
                    st.warning("Tidak ada kolom numerik (Likert) yang terdeteksi.")
                    
            with col_tip2:
                st.markdown("#### Preview Kolom Kategorikal (Demografi / Kualitatif)🅰️")
                if categorical_cols:
                    st.write(f"Total **{len(categorical_cols)}** Variabel:")
                    # MENAMPILKAN DATA KATEGORIKAL SECARA LANGSUNG
                    st.dataframe(df[categorical_cols].head(5), use_container_width=True)
                else:
                    st.warning("Tidak ada kolom kategorikal yang terdeteksi.")
            
            st.markdown("---")

            # 4. Analisis Data Numerik (Skor Likert)
            if len(numeric_cols) > 0:
                st.subheader("🔢Ringkasan Statistik Data Numerik (Skor Likert)")
                col_num_1, col_num_2 = st.columns([2, 3])
                
                with col_num_1:
                    st.markdown("#### Statistik Deskriptif (Skala 1-5)")
                    stats = df[numeric_cols].describe().T[['count', 'mean', 'std', 'min', 'max']].sort_values(by='mean', ascending=False)
                    # Menambahkan gradient warna untuk mean
                    st.dataframe(stats.style.background_gradient(cmap='RdYlGn', subset=['mean']), use_container_width=True)
                    
                with col_num_2:
                    st.markdown("#### Perbandingan Rata-Rata Skor Efektivitas")
                    mean_df = df[numeric_cols].mean().sort_values(ascending=True).to_frame(name='Rata-Rata Skor')
                    mean_df = mean_df.reset_index().rename(columns={'index': 'Variabel'})
                    fig_mean = px.bar(mean_df, x='Rata-Rata Skor', y='Variabel', orientation='h',
                                      color='Rata-Rata Skor', color_continuous_scale=px.colors.sequential.Inferno,
                                      title="Rata-Rata Skor Likert")
                    fig_mean.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig_mean, use_container_width=True)
            else:
                st.subheader("🔢Ringkasan Statistik Data Numerik (Skor Likert)")
                st.info("Tidak ada kolom numerik (Likert) yang terdeteksi untuk analisis skor.")

            st.markdown("---")
            
            # 5. Analisis Data Kategori Utama (Demografi)
            st.subheader("🫡Distribusi Data Kategori Utama (Demografi)")
            
            # Contoh Visualisasi (Fakultas)
            if 'Fakultas' in df.columns:
                counts = df['Fakultas'].value_counts().reset_index()
                counts.columns = ['Fakultas', 'count']
                fig_cat = px.pie(counts, names='Fakultas', values='count', 
                                 title=f"Proporsi Responden Berdasarkan Fakultas",
                                 color_discrete_sequence=px.colors.qualitative.Pastel,
                                 hole=.3)
                fig_cat.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_cat, use_container_width=True)
            elif categorical_cols:
                # Fallback to the first available categorical column
                first_cat = categorical_cols[0]
                counts = df[first_cat].value_counts().reset_index()
                counts.columns = [first_cat, 'count']
                fig_cat = px.pie(counts, names=first_cat, values='count', 
                                 title=f"Proporsi Responden Berdasarkan {first_cat}",
                                 color_discrete_sequence=px.colors.qualitative.Pastel,
                                 hole=.3)
                fig_cat.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_cat, use_container_width=True)
            else:
                st.info("Kolom 'Fakultas' atau kolom kategorikal utama lainnya tidak ditemukan dalam data.")


        # ------------------ Descriptive ------------------
        elif app_mode == "🗂️Deskriptif":
            st.title("🦋Analisis Deskriptif & Visualisasi Variatif🦋")
            st.markdown("Halaman ini menampilkan statistik dasar dan distribusi data kategorikal serta numerik.")

            # Section 1: Demografi
            st.header("🫨Distribusi Demografi & Kategori")
            with st.expander("Pilih kolom demografis/kategorikal untuk breakdown"):
                
                # Exclude obvious filter columns from the selection list for visual clarity
                sel_cat_options = [c for c in categorical_cols if c not in filter_cols_mapping.values()]
                
                sel_cat = st.multiselect('Pilih kategori (mis. Program Studi, Jenis Kendaraan)', sel_cat_options, default=sel_cat_options[:2])

            col_dem1, col_dem2 = st.columns(2)
            for i, c in enumerate(sel_cat):
                if i % 2 == 0:
                    col = col_dem1
                else:
                    col = col_dem2

                with col:
                    counts = df[c].value_counts().reset_index()
                    counts.columns = [c, 'count']
                    if len(counts) <= 8:
                        fig = px.pie(counts, names=c, values='count', title=f"Proporsi: {c}")
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                    else:
                        fig = px.bar(counts.head(10), x=c, y='count', title=f"Top 10 Distribusi: {c}")

                    st.plotly_chart(fig, use_container_width=True)


            # Section 2: Analisis Skor Likert
            st.header("🤯Statistik & Perbandingan Skor Likert (Numerik)")
            if len(numeric_cols)>0:
                col_stat1, col_stat2 = st.columns([1, 2])
                with col_stat1:
                    st.subheader("Ringkasan Skor")
                    stats = df[numeric_cols].describe().T[['count', 'mean', 'std', 'min', 'max']].sort_values(by='mean', ascending=False)
                    st.dataframe(stats.style.background_gradient(cmap='RdYlGn', subset=['mean']), use_container_width=True)

                with col_stat2:
                    st.subheader("Visualisasi Rata-Rata Skor")
                    mean_df = df[numeric_cols].mean().sort_values(ascending=True).to_frame(name='Rata-Rata Skor')
                    mean_df = mean_df.reset_index().rename(columns={'index': 'Variabel'})
                    fig_mean = px.bar(mean_df, x='Rata-Rata Skor', y='Variabel', orientation='h',
                                      color='Rata-Rata Skor', color_continuous_scale=px.colors.sequential.Inferno,
                                      title="Perbandingan Rata-Rata Skor Likert")
                    fig_mean.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig_mean, use_container_width=True)
            else:
                st.warning("Tidak ada kolom numerik (Likert) yang terdeteksi untuk analisis skor.")

        # ------------------ Effectiveness Analysis (Analisis Kunci) ------------------
        elif app_mode == "Analisis Kunci (Efektivitas Parkir)📩":
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
                          delta=f"{(overall_mean - 3.0)*100/3:.2f}% dari skala maks (jika skala 1-5)", delta_color="normal")
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
                
                if categorical_cols:
                    breakdown_col = st.selectbox("Pilih Kategori Pembanding:", options=categorical_cols, index=0)
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
        elif app_mode == "📊Korelasi":
            st.title("🫨Analisis Korelasi🫨")
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

                else:
                    st.info("Pilih minimal 2 variabel untuk melihat korelasi.")

        # ------------------ Regression ------------------
        elif app_mode == "📈Regresi Linear Berganda":
            st.title("🤗Regresi Linear Berganda🤗")
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
        elif app_mode == "📉Teks (essay)":
            st.title("🦋Analisis Jawaban Essay/Open-Ended🦋")
            st.markdown("Analisis frekuensi kata membantu merangkum keluhan, kendala, atau saran yang paling sering diungkapkan responden.")

            # Filter kolom teks yang mungkin adalah jawaban essay
            essay_cols = [c for c in df.columns if df[c].dtype == 'object' and not any(kw in c.lower() for kw in ['nama', 'npm', 'timestamp', 'fakultas', 'studi'])]
            
            sel_text = st.multiselect("Pilih Kolom Teks (Essay/Jawaban Terbuka)", essay_cols, default=essay_cols[:1])

            if sel_text:
                for c in sel_text:
                    st.header(f"Ringkasan Kata Kunci — {c}")
                    tw = top_words(df[c].astype(str), n=30)
                    
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
        elif app_mode == "📩Download & Petunjuk📩":
            st.title("🦋Petunjuk Deploy & Download🦋")
            st.markdown(
                "1. Pastikan file `app.py` dan `requirements.txt` (jika ada) ada di repository GitHub kamu.\n"
                "2. Jika ingin data disertakan di repo, tambahkan `Responden.xlsx` ke repo agar Streamlit Cloud dapat membukanya.\n"
                "3. Di Streamlit Cloud (share.streamlit.io) buat New app → konek ke repo → pilih `app.py`.\n"
                "4. Jika terjadi error, buka Logs di Streamlit Cloud untuk melihat pesan kesalahan."
            )

            st.subheader("Download Data Analisis")
            st.write("Gunakan tombol di bawah ini untuk mendownload data yang sudah terfilter sebagai CSV.")
            
            @st.cache_data
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')
            
            csv = convert_df_to_csv(df)
            
            st.download_button("Download Data Hasil Filter (.csv)", csv, file_name='data_filtered_parkir.csv', mime='text/csv')
