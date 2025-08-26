# ALVEY: Autonomous Learning Variable Extraction Yield

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ALVEY adalah sistem rekomendasi paper akademis *end-to-end* yang dirancang untuk membantu para peneliti dan mahasiswa menavigasi banjir informasi di repositori arXiv. Alih-alih hanya mengandalkan kata kunci, ALVEY menggunakan pemahaman semantik untuk menemukan paper yang relevan secara konseptual.

![Screenshot Aplikasi ALVEY](![Uploading image.png…]()
)
*(Ganti gambar di atas dengan screenshot aplikasi Anda yang sedang berjalan)*

## 🚀 Fitur Utama

* **Pengambilan Data Otomatis**: Mengambil data paper terbaru langsung dari arXiv API.
* **Penyimpanan Efisien**: Menggunakan SQLite untuk menyimpan dan mengelola data paper secara lokal.
* **Dua Model Rekomendasi**:
    1.  **TF-IDF & Cosine Similarity**: Pendekatan klasik berbasis kata kunci yang cepat dan efisien.
    2.  **Sentence-BERT**: Pendekatan modern berbasis Transformer untuk pemahaman semantik dan pencarian makna.
* **Antarmuka Interaktif**: Dibangun dengan Streamlit untuk kemudahan penggunaan dan perbandingan model secara langsung.

## 🛠️ Teknologi yang Digunakan

* **Backend & Logika**: Python
* **Antarmuka Web**: Streamlit
* **Pengambilan Data**: ArXiv API Wrapper
* **Manipulasi Data**: Pandas
* **Database**: SQLite
* **NLP & Machine Learning**: Scikit-learn, Sentence-Transformers (Hugging Face), NLTK

## ⚙️ Panduan Instalasi & Menjalankan

Berikut adalah langkah-langkah untuk menjalankan proyek ini di komputer lokal Anda.

### 1. Prasyarat

* Pastikan Anda sudah menginstal **Python 3.9** atau versi yang lebih baru.

### 2. Instalasi

1.  **Clone Repositori**
    ```sh
    git clone [https://github.com/](https://github.com/)[USERNAME_ANDA]/[NAMA_REPOSITORI_ANDA].git
    cd [NAMA_REPOSITORI_ANDA]
    ```

2.  **Buat dan Aktifkan Lingkungan Virtual (Virtual Environment)**
    ```sh
    # Buat venv
    python -m venv venv

    # Aktifkan venv (Windows)
    .\venv\Scripts\activate
    # Aktifkan venv (macOS / Linux)
    source venv/bin/activate
    ```

3.  **Instal Semua Pustaka yang Dibutuhkan**
    ```sh
    pip install -r requirements.txt
    ```

4.  **Unduh Data NLTK (Penting!)**
    Jalankan skrip ini untuk memastikan semua data NLTK terunduh dengan benar di dalam folder proyek.
    ```sh
    python download_nltk.py
    ```

### 3. Menjalankan Proyek

1.  **Langkah Pertama: Isi Database**
    Jalankan skrip ini untuk mengambil data dari arXiv dan menyimpannya ke database lokal. Proses ini memerlukan koneksi internet.
    ```sh
    python data_ingestion.py
    ```
    Tunggu hingga proses selesai. Sebuah file bernama `arxiv_papers.db` akan dibuat.

2.  **Langkah Kedua: Jalankan Aplikasi Streamlit**
    Setelah database terisi, jalankan aplikasi web.
    ```sh
    streamlit run app.py
    ```
    Aplikasi akan terbuka secara otomatis di browser Anda. Saat pertama kali dijalankan, aplikasi akan butuh waktu untuk mengunduh model AI dan membuat file *embeddings*.

## ✒️ Penulis

* **Rafi Khairan** - [LinkedIn](https://linkedin.com/in/rafikhairan) | [Portfolio](https://khay.my.id)

## 📄 Lisensi

Proyek ini dilisensikan di bawah Lisensi MIT.
