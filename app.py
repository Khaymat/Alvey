import streamlit as st
import pandas as pd
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import os

# Impor fungsi preprocessing dari file preprocessing.py
from preprocessing import preprocess_text

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Fungsi Caching untuk Model dan Data ---

@st.cache_resource
def load_sbert_model(model_name='all-MiniLM-L6-v2'):
    """Memuat model Sentence-BERT dan menyimpannya di cache."""
    logging.info(f"Memuat model SBERT: {model_name}...")
    model = SentenceTransformer(model_name)
    logging.info("Model SBERT berhasil dimuat.")
    return model

@st.cache_data
def load_data(db_name='arxiv_papers.db'):
    """Memuat data dari database dan menyimpannya di cache."""
    logging.info(f"Memuat data dari database: {db_name}...")
    conn = sqlite3.connect(db_name)
    df = pd.read_sql_query("SELECT * FROM papers", conn)
    conn.close()
    logging.info(f"Berhasil memuat {len(df)} paper.")
    return df

@st.cache_data
def generate_embeddings(_df, _model, embeddings_file='sbert_embeddings.npy'):
    """Membuat atau memuat SBERT embeddings dari file."""
    if os.path.exists(embeddings_file):
        logging.info(f"Memuat embeddings dari file: {embeddings_file}...")
        embeddings = np.load(embeddings_file)
        return embeddings
    
    logging.info("Membuat SBERT embeddings baru...")
    # Kita tidak perlu membuat df_copy karena _df dari @st.cache_data adalah salinan
    _df['full_text'] = _df['title'] + '. ' + _df['summary']
    embeddings = _model.encode(_df['full_text'].tolist(), show_progress_bar=True)
    np.save(embeddings_file, embeddings)
    logging.info(f"Embeddings berhasil dibuat dan disimpan ke {embeddings_file}.")
    return embeddings

@st.cache_resource
def build_tfidf_model(_df):
    """Membangun model TF-IDF (matriks dan cosine similarity) dan menyimpannya di cache."""
    logging.info("Membangun model TF-IDF...")
    df_copy = _df.copy()
    df_copy['full_text'] = df_copy['title'] + ' ' + df_copy['summary']
    df_copy['processed_text'] = df_copy['full_text'].apply(preprocess_text)
    
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(df_copy['processed_text'])
    cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    logging.info("Model TF-IDF berhasil dibangun.")
    return cosine_sim_matrix

# --- Fungsi Rekomendasi ---

def get_recommendations_tfidf(arxiv_id, df, cosine_sim_matrix, top_n=5):
    """Memberikan rekomendasi berdasarkan TF-IDF & Cosine Similarity."""
    try:
        # Dapatkan index dari paper input
        idx = df.index[df['arxiv_id'] == arxiv_id].tolist()[0]
    except IndexError:
        return None

    # Dapatkan skor kesamaan untuk paper tersebut
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    
    # Urutkan paper berdasarkan skor kesamaan
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Ambil skor dari top_n paper (abaikan paper input itu sendiri)
    sim_scores = sim_scores[1:top_n+1]
    
    # Dapatkan index paper yang direkomendasikan
    paper_indices = [i[0] for i in sim_scores]
    
    return df.iloc[paper_indices]

def get_recommendations_sbert(arxiv_id, df, embeddings, top_n=5):
    """Memberikan rekomendasi berdasarkan Sentence-BERT."""
    try:
        idx = df.index[df['arxiv_id'] == arxiv_id].tolist()[0]
    except IndexError:
        return None
    
    query_embedding = embeddings[idx]
    
    # Hitung cosine similarity antara query embedding dengan semua embedding lain
    hits = util.semantic_search(query_embedding, embeddings, top_k=top_n + 1)
    
    # Ambil hasil teratas (abaikan hasil pertama karena itu adalah paper input)
    recommendation_indices = [hit['corpus_id'] for hit in hits[0][1:]]
    
    return df.iloc[recommendation_indices]

# --- UI Aplikasi Streamlit ---

st.set_page_config(page_title="Rekomendasi Paper arXiv", layout="wide")

st.title("ALVEY")
st.markdown("Autonomous Learning Variable Extraction Yield")

# Muat model dan data
sbert_model = load_sbert_model()
df_papers = load_data()
sbert_embeddings = generate_embeddings(df_papers, sbert_model)
tfidf_cosine_sim = build_tfidf_model(df_papers)

# Sidebar untuk pilihan model
st.sidebar.title("Pengaturan Model")
model_choice = st.sidebar.selectbox(
    "Pilih Model Rekomendasi:",
    ("Sentence-BERT (Semantik)", "TF-IDF (Kata Kunci)")
)

# Input dari pengguna
default_id = df_papers['arxiv_id'].iloc[0] if not df_papers.empty else "1706.03762"
arxiv_id_input = st.text_input(f"Masukkan ID arXiv (contoh: {default_id}):", default_id)

if st.button("Dapatkan Rekomendasi"):
    if arxiv_id_input and not df_papers.empty:
        try:
            # Tampilkan detail paper input
            input_paper = df_papers[df_papers['arxiv_id'] == arxiv_id_input].iloc[0]
            
            with st.container(border=True):
                st.subheader("Paper Input:")
                st.markdown(f"**Judul:** {input_paper['title']}")
                st.markdown(f"**Penulis:** {input_paper['authors']}")
                with st.expander("Lihat Abstrak"):
                    st.write(input_paper['summary'])
            
            st.subheader(f"Rekomendasi Berdasarkan Model {model_choice}:")
            
            recommendations = None
            if model_choice == "Sentence-BERT (Semantik)":
                recommendations = get_recommendations_sbert(arxiv_id_input, df_papers, sbert_embeddings)
            else: # TF-IDF
                recommendations = get_recommendations_tfidf(arxiv_id_input, df_papers, tfidf_cosine_sim)

            if recommendations is not None and not recommendations.empty:
                for index, row in recommendations.iterrows():
                    with st.container(border=True):
                        st.markdown(f"**Judul:** {row['title']}")
                        st.markdown(f"**Penulis:** {row['authors']}")
                        st.markdown(f"**ID arXiv:** `{row['arxiv_id']}` | **Link PDF:** [Unduh]({row['pdf_url']})")
                        with st.expander("Lihat Abstrak"):
                            st.write(row['summary'])
            else:
                st.error(f"Tidak dapat menemukan paper dengan ID '{arxiv_id_input}' atau tidak ada rekomendasi yang ditemukan.")
                
        except IndexError:
            st.error(f"Paper dengan ID arXiv '{arxiv_id_input}' tidak ditemukan dalam database lokal kami.")
    else:
        st.warning("Silakan masukkan ID arXiv atau pastikan database tidak kosong.")