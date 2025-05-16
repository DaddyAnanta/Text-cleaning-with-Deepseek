# utils/nlp_tools.py

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag # Untuk Part-of-Speech Tagging


import Sastrawi

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


stop_words = StopWordRemoverFactory().get_stop_words()
new_array = ArrayDictionary(stop_words)
stop_words_remover_new = StopWordRemover(new_array)


# --- Unduh NLTK Resources ---
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")  
nltk.download("averaged_perceptron_tagger")
nltk.download('averaged_perceptron_tagger_eng') 

# --- Inisialisasi Alat NLP ---
ps = PorterStemmer() # Stemmer (Porter)
lemmatizer = WordNetLemmatizer() # Lemmatizer (WordNet)
stop_words = set(stopwords.words("indonesian")) # Stopwords Bahasa Indonesia


def tokenize_text(text):
    """Tokenisasi teks menjadi list of words."""
    # Menambahkan str() untuk memastikan input adalah string
    return word_tokenize(str(text))

def remove_stopwords_indo(tokens):
    """Hapus stopwords dari list of tokens."""
    # Memastikan perbandingan case-insensitive
    return [w for w in tokens if w.lower() not in stop_words]

def sastrawi(str_text):
    str_text = stop_words_remover_new.remove(str_text)
    return str_text



def stemming_indo(text_cleaning):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    do = []
    for w in text_cleaning:
        dt = stemmer.stem(w)
        do.append(dt)
    d_clean = []
    d_clean = " ".join(do)
    return d_clean
