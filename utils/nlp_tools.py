# utils/nlp_tools.py

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag # Untuk Part-of-Speech Tagging

# --- Unduh NLTK Resources ---
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")  # Open Multilingual Wordnet, sering dibutuhkan oleh WordNet
nltk.download("averaged_perceptron_tagger")
nltk.download('averaged_perceptron_tagger_eng') 

# --- Inisialisasi Alat NLP ---
ps = PorterStemmer() # Stemmer (Porter)
lemmatizer = WordNetLemmatizer() # Lemmatizer (WordNet)
stop_words = set(stopwords.words("english")) # Stopwords Bahasa Indonesia




# --- Fungsi Bantuan untuk Lemmatizer ---
def get_wordnet_pos(treebank_tag):
    """
    Konversi NLTK POS tag ke format yang diterima WordNetLemmatizer.
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        # Default ke Noun jika tidak cocok
        return wordnet.NOUN

# --- Fungsi NLP Utama ---


def tokenize_text(text):
    """Tokenisasi teks menjadi list of words."""
    # Menambahkan str() untuk memastikan input adalah string
    return word_tokenize(str(text))

def remove_stopwords(tokens):
    """Hapus stopwords dari list of tokens."""
    # Memastikan perbandingan case-insensitive
    return [w for w in tokens if w.lower() not in stop_words]

def apply_stemming(tokens):
    """Terapkan Porter Stemming pada list of tokens."""
    return [ps.stem(w) for w in tokens]

def apply_lemmatization(tokens):
    # ... (fungsi ini tetap sama) ...
    pos_tags = pos_tag(tokens) # Baris ini yang menyebabkan error sebelumnya
    lemmatized_tokens = [
        lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag))
        for word, tag in pos_tags
    ]
    return lemmatized_tokens
