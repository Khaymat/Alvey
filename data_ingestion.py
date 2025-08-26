import arxiv
import sqlite3
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_database(db_name='arxiv_papers.db'):
    """Membuat database dan tabel jika belum ada."""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            arxiv_id TEXT UNIQUE,
            title TEXT,
            summary TEXT,
            authors TEXT,
            published_date DATE,
            primary_category TEXT,
            pdf_url TEXT
        )
    ''')
    conn.commit()
    logging.info(f"Database '{db_name}' siap digunakan.")
    return conn

def fetch_and_store_papers(conn, query, max_results=100):
    """Mengambil data dari arXiv dan menyimpannya ke database."""
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    cursor = conn.cursor()
    papers_added = 0
    
    logging.info(f"Memulai pengambilan data untuk kueri: '{query}'")
    try:
        results = client.results(search)
        for result in results:
            # Mengambil ID unik dari URL entri
            arxiv_id = result.entry_id.split('/')[-1]
            authors = ', '.join(author.name for author in result.authors)
            published_date = result.published.strftime('%Y-%m-%d')
            
            paper_data = (
                arxiv_id,
                result.title,
                result.summary,
                authors,
                published_date,
                result.primary_category,
                result.pdf_url
            )
            
            try:
                # INSERT OR IGNORE untuk mencegah duplikasi berdasarkan arxiv_id UNIQUE
                cursor.execute('''
                    INSERT OR IGNORE INTO papers (arxiv_id, title, summary, authors, published_date, primary_category, pdf_url)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', paper_data)
                
                if cursor.rowcount > 0:
                    papers_added += 1
            except sqlite3.Error as e:
                logging.error(f"Gagal memasukkan paper {arxiv_id}: {e}")
                
    except Exception as e:
        logging.error(f"Terjadi kesalahan saat mengambil data dari arXiv: {e}")
    finally:
        conn.commit()
        logging.info(f"Selesai. {papers_added} paper baru ditambahkan ke database.")


if __name__ == '__main__':
    # Download resource NLTK yang dibutuhkan untuk preprocessing
    import nltk
    print("Mengunduh resource NLTK (punkt, stopwords, wordnet)...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    print("Unduhan NLTK selesai.")

    DB_NAME = 'arxiv_papers.db'
    # Kueri untuk mendapatkan paper dari kategori AI, Machine Learning, dan NLP
    SEARCH_QUERY = "cat:cs.AI OR cat:cs.LG OR cat:cs.CL"
    MAX_RESULTS = 500  # Anda bisa menaikkan angka ini untuk mendapatkan lebih banyak data

    connection = setup_database(DB_NAME)
    if connection:
        fetch_and_store_papers(connection, SEARCH_QUERY, MAX_RESULTS)
        connection.close()
        logging.info("Koneksi database ditutup.")