import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    """
    Membersihkan dan memproses teks input:
    1. Lowercasing
    2. Hapus Punctuation
    3. Hapus Angka
    4. Tokenisasi
    5. Hapus Stop Words
    6. Lemmatisasi
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercasing
    text = text.lower()
    
    # 2. Hapus Punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 3. Hapus Angka
    text = re.sub(r'\d+', '', text)
    
    # 4. Tokenisasi
    tokens = word_tokenize(text)
    
    # 5. Hapus Stop Words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # 6. Lemmatisasi
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    
    return " ".join(lemmatized_tokens)