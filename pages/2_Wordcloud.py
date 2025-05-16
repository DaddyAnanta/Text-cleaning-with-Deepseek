import streamlit as st
import pandas as pd
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import io
import base64 # Untuk encoding gambar
import requests # Untuk API call
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords
from collections import Counter
import re
import time

# Asumsikan utils.nlp_indo ada dan berisi fungsi-fungsi ini
# Jika tidak ada, Anda perlu mendefinisikannya atau menginstalnya
try:
    from utils.nlp_indo import remove_stopwords_indo, sastrawi, stemming_indo, tokenize_text
    NLP_UTILS_AVAILABLE = True
except ImportError:
    NLP_UTILS_AVAILABLE = False
    st.warning("Modul 'utils.nlp_indo' tidak ditemukan. Fungsi preprocessing Bahasa Indonesia tidak akan tersedia.")
    # Definisikan fungsi placeholder jika modul tidak ada
    def remove_stopwords_indo(text): return text
    def sastrawi(text): return text
    def stemming_indo(text): return text
    def tokenize_text(text):
        if isinstance(text, str):
            return nltk.word_tokenize(text) # Gunakan NLTK dasar jika util tidak ada
        return []


# --- Konfigurasi Halaman & Download NLTK ---
st.set_page_config(page_title="Analisis Teks & Insight AI", layout="wide")
st.title("📊 Analisis Teks CSV dengan Filter Sentimen, Visualisasi & Insight AI")
st.markdown("Unggah CSV, pilih kolom teks & sentimen, filter data, lihat visualisasi N-Gram, lalu dapatkan insight AI.")

# --- Inisialisasi Session State ---
# (Session state Anda sudah benar)
if 'image_data' not in st.session_state:
    st.session_state.image_data = {}
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'last_uploaded_filename' not in st.session_state:
    st.session_state.last_uploaded_filename = None
if 'active_filters' not in st.session_state:
    st.session_state.active_filters = {'sentiments': [], 'text_col': None, 'sentiment_col': None} # Untuk info AI


# Download resource NLTK
@st.cache_resource
def download_nltk_resources():
    resources = {'corpora/stopwords': 'stopwords', 'tokenizers/punkt': 'punkt'}
    all_downloaded = True
    for path, pkg_id in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            st.info(f"Mengunduh resource NLTK: {pkg_id}...")
            try:
                nltk.download(pkg_id, quiet=True)
                time.sleep(1) # Beri jeda sedikit
                nltk.data.find(path) # Verify download
                st.success(f"Resource '{pkg_id}' siap.")
            except Exception as e:
                 st.error(f"Gagal mengunduh resource '{pkg_id}': {e}. Fitur terkait mungkin tidak berfungsi.")
                 all_downloaded = False
    return all_downloaded

NLTK_READY = download_nltk_resources()

# --- Start Helper Functions ---
def preprocess_text(text, stop_words_set):
    """Membersihkan dan mentokenisasi teks. (Lowercase, remove digits, punctuation, stopwords)"""
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r'\d+', '', text) # Hapus angka
    text = re.sub(r'[^\w\s]', '', text) # Hapus punctuation
    text = text.strip() # Hapus spasi di awal/akhir
    if not text:
        return []
    try:
        tokens = nltk.word_tokenize(text)
        filtered_tokens = [
            word for word in tokens
            if word.isalnum() and word not in stop_words_set and len(word) > 1
        ]
        return filtered_tokens
    except Exception as e:
        # st.warning(f"Error tokenizing text: '{text[:50]}...' ({e}). Skipping.")
        return []


def generate_ngrams_freq(tokens, n):
    """Menghasilkan N-gram dan menghitung frekuensinya."""
    if len(tokens) < n:
        return Counter()
    n_grams_list = list(ngrams(tokens, n))
    formatted_ngrams = [" ".join(gram) for gram in n_grams_list]
    return Counter(formatted_ngrams)

def create_wordcloud_fig(word_counts, title):
    """Membuat figure Word Cloud (TIDAK MENAMPILKAN)."""
    if not word_counts:
        return None # Kembalikan None jika tidak ada data

    # Pilih colormap berdasarkan N-gram type
    colormap = 'viridis'
    if "Bigram" in title:
        colormap = 'plasma'
    elif "Trigram" in title:
        colormap = 'magma'

    fig, ax = plt.subplots(figsize=(8, 5)) # Ukuran disesuaikan untuk kolom

    try:
        wordcloud = WordCloud(
            width=800, height=500,
            random_state=42,
            background_color='white',
            colormap=colormap,
            collocations=False, # Penting untuk n-gram > 1
            # prefer_horizontal=0.9, # Opsional: lebih banyak kata horizontal
            # max_words=100 # Opsional: batasi jumlah kata
        ).generate_from_frequencies(dict(word_counts)) # Konversi Counter ke dict
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        # Jangan tambahkan plt.title di sini agar tidak terduplikasi saat fig_to_base64
        return fig # Kembalikan objek figure
    except ValueError as ve:
         st.warning(f"Tidak dapat membuat WordCloud {title}: Mungkin semua frekuensi nol atau format data tidak valid. {ve}")
         plt.close(fig) # Tutup jika error
         return None
    except Exception as e:
        st.warning(f"Error membuat WordCloud {title}: {e}")
        plt.close(fig) # Tutup jika error
        return None

def create_barplot_fig(word_counts, title, n_top=15):
    """Membuat figure Bar Plot Horizontal (TIDAK MENAMPILKAN)."""
    if not word_counts:
        return None

    # Pilih warna berdasarkan N-gram type
    color = 'skyblue' # Default untuk Unigram/lainnya
    ylabel_text = "Kata"
    if "Bigram" in title:
        color = 'lightcoral'
        ylabel_text = "Bigram"
    elif "Trigram" in title:
        color = 'lightgreen'
        ylabel_text = "Trigram"


    top_items = word_counts.most_common(n_top)
    if not top_items:
        return None # Tidak ada item untuk diplot

    labels = [item[0] for item in top_items]
    counts = [item[1] for item in top_items]

    # Balik urutan agar yang frekuensi tertinggi di atas
    labels.reverse()
    counts.reverse()

    fig, ax = plt.subplots(figsize=(8, max(5, len(labels) * 0.35))) # Ukuran dinamis
    bars = ax.barh(labels, counts, color=color)
    ax.set_xlabel('Frekuensi')
    ax.set_ylabel(ylabel_text)
    # ax.set_title(f'Top {n_top} {title}') # Judul bisa ditambahkan di sini atau saat display

    # Tambahkan label angka di ujung bar
    max_count = max(counts) if counts else 1
    for bar in bars:
        width = bar.get_width()
        # Penyesuaian posisi teks agar tidak terlalu menempel
        ax.text(width + (max_count * 0.01), bar.get_y() + bar.get_height()/2.,
                f'{width}', va='center', ha='left', fontsize=9)

    # Pastikan label tidak terpotong
    plt.tight_layout()
    return fig # Kembalikan objek figure

def fig_to_base64(fig):
    """Mengubah figure Matplotlib menjadi string Base64."""
    if fig is None:
        return None
    img_buf = io.BytesIO()
    try:
        # Gunakan bbox_inches='tight' untuk mengurangi whitespace
        fig.savefig(img_buf, format='png', bbox_inches='tight', dpi=150) # DPI bisa disesuaikan
        plt.close(fig) # Tutup figure setelah disimpan untuk membebaskan memori
        img_buf.seek(0)
        base64_string = base64.b64encode(img_buf.read()).decode('utf-8')
        return f"data:image/png;base64,{base64_string}"
    except Exception as e:
        st.warning(f"Gagal mengkonversi figure ke base64: {e}")
        plt.close(fig) # Pastikan figure ditutup meski error
        return None

def analyze_image_with_openrouter(api_key, model_id, prompt, image_data_list):
    """Mengirim prompt dan gambar ke API OpenRouter."""
    # (Fungsi ini sepertinya sudah benar, tidak perlu diubah)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "urn:application:streamlit-ngram-insight", # Ganti nama app Anda
        "X-Title": "Streamlit N-Gram Insight",
    }

    content_list = [{"type": "text", "text": prompt}]
    for img_data_url in image_data_list:
        if img_data_url and isinstance(img_data_url, str) and img_data_url.startswith('data:image'):
             content_list.append({
                 "type": "image_url",
                 "image_url": {"url": img_data_url}
             })
        # else: st.warning(f"Skipping invalid image data: {type(img_data_url)}") # Debug

    if len(content_list) <= 1: # Hanya ada prompt teks
        st.warning("Tidak ada gambar visualisasi yang valid untuk dikirim ke AI.")
        return None

    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": content_list}],
        "max_tokens": 3500, # Naikkan jika perlu analisis lebih panjang
        "temperature": 0.5
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=300 # Timeout lebih lama
        )
        response.raise_for_status() # Akan raise error untuk status 4xx/5xx
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0 and "message" in result["choices"][0] and "content" in result["choices"][0]["message"]:
             # Ganti \n dengan spasi ganda + newline agar markdown render baris baru
             return result["choices"][0]["message"]["content"].replace('\n', '  \n')
        else:
             st.warning("Struktur respons API tidak sesuai harapan.")
             st.json(result) # Tampilkan respons mentah untuk debug
             return "Gagal memproses respons dari model. Struktur tidak dikenali."
    except requests.exceptions.Timeout:
        st.error("Permintaan ke API OpenRouter timeout. Coba lagi nanti atau dengan model yang lebih cepat.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error saat menghubungi API OpenRouter: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                # Coba tampilkan detail error dari JSON response jika ada
                error_details = e.response.json()
                st.error(f"Detail Error API (Status {e.response.status_code}):")
                st.json(error_details)
            except requests.exceptions.JSONDecodeError:
                # Jika response bukan JSON
                st.error(f"Detail Error API (Status {e.response.status_code}, Non-JSON): {e.response.text}")
        return None
    except Exception as e: # Tangkap error lainnya
        st.error(f"Terjadi error tak terduga saat analisis AI: {e}")
        import traceback
        st.error(traceback.format_exc()) # Tampilkan traceback untuk debug
        return None

# --- End Helper Functions ---

# --- Input API Key & Model (Sidebar) ---
with st.sidebar:
    st.header("Pengaturan AI")
    api_key = st.text_input("Masukkan API Key OpenRouter", type="password", key="api_key_input")
    model_id = st.selectbox("Pilih Model Multi-Modal", [
        "google/gemini-flash-1.5", # Lebih cepat & murah
        "google/gemini-pro-vision",
        "anthropic/claude-3-haiku-20240307", # Cepat & bagus
        "anthropic/claude-3-sonnet-20240229",
        "anthropic/claude-3-opus-20240229", # Paling canggih, tapi mahal
    ], key="model_select", help="Pilih model AI yang mendukung input gambar.", index=0)

# --- INPUT FILE ---
st.subheader("1. Unggah Data")
uploaded_file = st.file_uploader("Pilih file CSV Anda", type="csv", key="file_uploader")

# --- PEMROSESAN DATA & VISUALISASI ---
if uploaded_file is not None:
    try:
        # Baca data original, simpan ke session state agar tidak perlu dibaca ulang terus
        # kecuali jika file baru diupload.
        if st.session_state.last_uploaded_filename != uploaded_file.name:
            st.session_state.data_ori = pd.read_csv(uploaded_file)
            st.session_state.image_data = {}
            st.session_state.analysis_done = False
            st.session_state.last_uploaded_filename = uploaded_file.name
            st.session_state.active_filters = {'sentiments': [], 'text_col': None, 'sentiment_col': None}
            st.success(f"File '{uploaded_file.name}' berhasil diunggah dan dibaca.")
            # Bersihkan state lama jika ada file baru
            if 'data_processed' in st.session_state:
                 del st.session_state.data_processed
            if 'all_tokens_filtered' in st.session_state:
                 del st.session_state.all_tokens_filtered


        # Gunakan data dari session state
        data_ori = st.session_state.data_ori

        with st.expander("Pratinjau Data Original (5 Baris Pertama)"):
            st.dataframe(data_ori.head())

        # --- Pengaturan Kolom & Filter ---
        st.subheader("2. Pengaturan Analisis")
        col_setting1, col_setting2, col_setting3 = st.columns(3)

        with col_setting1:
            # Pemilihan kolom teks
            text_column = st.selectbox(
                "Pilih kolom teks:",
                options=[None] + list(data_ori.columns),
                key="text_column_select",
                index=0, # Default ke None
                help="Kolom yang berisi teks utama untuk dianalisis."
            )

        with col_setting2:
             # Pilih bahasa untuk preprocessing (stopwords, stemming)
             language_option = st.radio(
                 "Bahasa Teks:",
                 ("Bahasa Indonesia", "Lainnya (English Stopwords)"),
                 key="language_radio",
                 horizontal=True,
                 help="Pilih bahasa untuk menentukan stopwords dan stemming (jika Bahasa Indonesia)."
             )

        with col_setting3:
            # Pemilihan kolom sentimen (opsional)
            sentiment_column = st.selectbox(
                "Kolom sentimen (Opsional):",
                options=[None] + [col for col in data_ori.columns if col != text_column], # Jangan tawarkan kolom teks lagi
                key="sentiment_column_select",
                index=0, # Default ke None
                help="Kolom berisi label sentimen (misal: 'Positif', 'Negatif')."
            )


        # --- Filter Sentimen (muncul jika kolom sentimen dipilih) ---
        selected_sentiments_display = [] # Untuk tampilan di info box
        selected_sentiments_lower = []   # Untuk filtering
        use_sentiment_filter = False

        if sentiment_column:
            st.markdown("**Filter berdasarkan Sentimen:**")
            # Ambil nilai unik dari kolom sentimen (convert ke string & lowercase)
            try:
                available_sentiments = data_ori[sentiment_column].dropna().astype(str).str.lower().unique()
                # Tampilkan checkbox untuk setiap sentimen unik yang ditemukan
                filter_cols = st.columns(min(len(available_sentiments), 5)) # Batasi jumlah kolom filter
                selected_sentiments_temp = []
                for i, sentiment in enumerate(available_sentiments):
                    with filter_cols[i % len(filter_cols)]:
                         # Label checkbox pakai versi asli (jika bisa) atau title case
                         # Cari case asli pertama yang cocok
                         original_case_sentiment = sentiment.capitalize() # Default
                         try:
                            mask = data_ori[sentiment_column].dropna().astype(str).str.lower() == sentiment
                            original_case_sentiment = data_ori.loc[mask, sentiment_column].iloc[0]
                         except:
                            pass # Gunakan capitalize jika gagal

                         if st.checkbox(f"{original_case_sentiment}", key=f"cb_sent_{sentiment}", value=False): # Default tidak terpilih
                             selected_sentiments_temp.append(sentiment) # Kumpulkan yang dipilih (lowercase)
                             selected_sentiments_display.append(original_case_sentiment) # Kumpulkan nama asli

                if selected_sentiments_temp:
                     selected_sentiments_lower = selected_sentiments_temp
                     use_sentiment_filter = True
                else:
                     # Jika tidak ada yang dipilih, anggap analisis semua sentimen
                     use_sentiment_filter = False
                     selected_sentiments_display = ["Semua"]


            except Exception as e:
                 st.warning(f"Gagal memproses nilai unik di kolom sentimen '{sentiment_column}': {e}. Filter tidak aktif.")
                 use_sentiment_filter = False
                 selected_sentiments_display = ["Semua"]
        else:
            # Jika tidak ada kolom sentimen dipilih
            selected_sentiments_display = ["Semua"]


        # --- Pengaturan N-Gram ---
        st.markdown("**Pilih Analisis N-Gram:**")
        col_opts1, col_opts2, col_opts3 = st.columns(3)
        with col_opts1:
            run_unigram = st.checkbox("Unigram", value=True, key="cb_unigram")
        with col_opts2:
            run_bigram = st.checkbox("Bigram", value=False, key="cb_bigram")
        with col_opts3:
            run_trigram = st.checkbox("Trigram", value=False, key="cb_trigram")

        analysis_requested = run_unigram or run_bigram or run_trigram

        # --- Tombol Jalankan Analisis ---
        st.markdown("---")
        analysis_button_pressed = st.button("🚀 Jalankan Analisis & Visualisasi", key="run_analysis_button", type="primary", disabled=(not text_column or not analysis_requested))

        if not text_column:
            st.info("Pilih kolom teks untuk memulai analisis.")
        elif not analysis_requested:
            st.warning("Pilih setidaknya satu jenis analisis N-Gram (Unigram, Bigram, atau Trigram).")

        # --- BLOK UTAMA ANALISIS (HANYA JIKA TOMBOL DITEKAN) ---
        if analysis_button_pressed:
            if not NLTK_READY:
                 st.error("Resource NLTK gagal diunduh. Analisis tidak dapat dilanjutkan.")
            else:
                st.session_state.image_data = {} # Kosongkan gambar lama setiap analisis baru
                st.session_state.analysis_done = False
                data_filtered = data_ori.copy() # Mulai dengan data asli

                # --- Terapkan Filter Sentimen ---
                filter_info = "Menganalisis semua data"
                if use_sentiment_filter and selected_sentiments_lower:
                    try:
                        # Filter case-insensitive menggunakan nilai lowercase yang sudah dikumpulkan
                        mask = data_filtered[sentiment_column].astype(str).str.lower().isin(selected_sentiments_lower)
                        data_filtered = data_filtered[mask]
                        filter_info = f"Menganalisis data dengan sentimen: **{', '.join(selected_sentiments_display)}**"
                        st.session_state.active_filters['sentiments'] = selected_sentiments_display # Simpan nama asli untuk AI
                        st.session_state.active_filters['sentiment_col'] = sentiment_column
                    except KeyError:
                        st.error(f"Kolom sentimen '{sentiment_column}' tidak ditemukan saat filtering. Analisis menggunakan semua data.")
                        use_sentiment_filter = False
                        filter_info = "Menganalisis semua data (kolom sentimen error)"
                        st.session_state.active_filters['sentiments'] = []
                        st.session_state.active_filters['sentiment_col'] = None
                    except Exception as e:
                        st.error(f"Error saat menerapkan filter sentimen: {e}. Analisis menggunakan semua data.")
                        use_sentiment_filter = False
                        filter_info = "Menganalisis semua data (filter error)"
                        st.session_state.active_filters['sentiments'] = []
                        st.session_state.active_filters['sentiment_col'] = None
                else:
                    # Info jika tidak ada filter aktif (baik karena tidak dipilih atau tidak ada kolom)
                    if sentiment_column:
                         filter_info = f"Menganalisis data dengan **semua sentimen** yang ada di kolom '{sentiment_column}'"
                    else:
                         filter_info = "Menganalisis semua data (tidak ada filter sentimen dipilih)"
                    st.session_state.active_filters['sentiments'] = [] # Kosong berarti semua
                    st.session_state.active_filters['sentiment_col'] = sentiment_column # Tetap simpan nama kolomnya

                # Simpan nama kolom teks yang digunakan
                st.session_state.active_filters['text_col'] = text_column
                st.info(f"{filter_info} dari kolom teks **'{text_column}'**.")

                if data_filtered.empty:
                    st.warning("Tidak ada data yang cocok dengan filter yang dipilih. Tidak ada visualisasi yang dapat dibuat.")
                else:
                    with st.spinner("Memproses teks (Stemming, Tokenisasi, Stopwords)..."):
                        # Ambil kolom teks yang sudah difilter
                        texts_to_process = data_filtered[text_column].dropna().astype(str)

                        if texts_to_process.empty:
                            st.warning(f"Kolom '{text_column}' tidak berisi data teks valid setelah filtering/dropna.")
                        else:
                             # --- PREPROCESSING TEKS (STEMMING, TOKENIZE, STOPWORDS) ---
                             # Tahap ini sekarang HANYA berjalan di dalam blok tombol

                             # 1. Stemming (hanya jika Bahasa Indonesia dan utils tersedia)
                             if language_option == "Bahasa Indonesia" and NLP_UTILS_AVAILABLE:
                                 st.write("⏳ Menerapkan stemming Bahasa Indonesia...")
                                 stemming_start_time = time.time()
                                 try:
                                     # Terapkan stemming_indo jika ada dan berbeda dari sastrawi

                                     # Terapkan Sastrawi
                                     if 'sastrawi' in globals() and callable(sastrawi):
                                        texts_to_process = texts_to_process.apply(lambda x: sastrawi(x)) # Gunakan langsung


                                     if 'stemming_indo' in globals() and callable(stemming_indo):
                                         texts_to_process = texts_to_process.apply(lambda x:x.split())
                                         texts_to_process = texts_to_process.apply(stemming_indo)

                                         
                                     st.success(f"Stemming Bahasa Indonesia selesai ({time.time() - stemming_start_time:.2f} detik).")
 
#                                     # --- TAMBAHKAN DEBUG DI SINI ---
#                                     st.subheader("Debug: Contoh Teks Setelah Stemming (10 Baris Pertama)")
#                                     if not texts_to_process.empty:
#                                         st.dataframe(texts_to_process.head(10).reset_index(drop=True)) # Tampilkan 10 contoh hasil stemming
#                                         # Juga cek apakah ada yang kosong
#                                         empty_after_stemming = texts_to_process.astype(str).str.strip().eq("").sum()
#                                         st.write(f"Jumlah baris yang menjadi string kosong setelah stemming: {empty_after_stemming} dari {len(texts_to_process)}")
#                                     else:
#                                         st.warning("texts_to_process kosong bahkan sebelum stemming loop?") # Seharusnya tidak terjadi jika data awal ada
 
                                 except Exception as e:
                                     st.error(f"Gagal menerapkan stemming Bahasa Indonesia: {e}. Melanjutkan tanpa stemming.")
                                     # Jika error, texts_to_process masih berisi teks sebelum stemming
                                     st.subheader("Debug: Contoh Teks Karena Stemming Gagal (10 Baris Pertama)")
                                     if not texts_to_process.empty:
                                          st.dataframe(texts_to_process.head(10).reset_index(drop=True))
                             # --- AKHIR BLOK DEBUG ---
 
                             elif language_option == "Bahasa Indonesia" and not NLP_UTILS_AVAILABLE:
                                  st.warning("Utils NLP Bahasa Indonesia tidak tersedia, stemming dilewati.")
 
                             # 2. Load Stopwords berdasarkan bahasa
                             stop_words_set = set() # Default kosong







                             try:
                                 if language_option == "Bahasa Indonesia":
                                     stop_words_set = set(stopwords.words('indonesian'))
                                     # Tambahkan stopwords custom jika perlu
                                     # custom_stopwords = ['yg', 'dg', 'rt', 'dgn', 'nya', 'klo', 'tdk', 'gak', 'ga', 'aja', 'sih']
                                     # stop_words_set.update(custom_stopwords)
                                     # st.write(f"Stopwords Indonesia dimuat: {len(stop_words_set)} kata.")
                                 else: # Asumsikan 'Lainnya' -> English
                                     stop_words_set = set(stopwords.words('english'))
                                     # st.write(f"Stopwords Inggris dimuat: {len(stop_words_set)} kata.")
                             except LookupError as le:
                                 st.error(f"Resource Stopwords NLTK tidak ditemukan ({le}). Pastikan sudah terunduh. Analisis dilanjutkan tanpa stopwords.")
                             except Exception as e:
                                 st.warning(f"Gagal memuat stopwords: {e}. Analisis dilanjutkan tanpa stopwords.")



                             # 3. Tokenisasi dan Pembersihan Akhir (menggunakan preprocess_text)
                             st.write("⏳ Melakukan tokenisasi dan pembersihan teks...")
                             tokenization_start_time = time.time()
                             all_tokens_filtered = []
                             processed_count = 0
                             for text in texts_to_process:
                                 tokens = preprocess_text(text, stop_words_set)
                                 all_tokens_filtered.extend(tokens)
                                 processed_count += 1
                                 # Mungkin tambahkan progress bar jika sangat lambat
                                 # if processed_count % 100 == 0:
                                 #     st.progress(processed_count / len(texts_to_process))

                             st.success(f"Tokenisasi selesai ({time.time() - tokenization_start_time:.2f} detik).")

                             # Simpan hasil token ke state (opsional, jika perlu diakses lagi tanpa rerun)
                             # st.session_state.all_tokens_filtered = all_tokens_filtered

                             if not all_tokens_filtered:
                                st.warning("Tidak ada token valid yang ditemukan setelah preprocessing.")
                             else:
                                st.success(f"Pemrosesan teks selesai. Ditemukan **{len(all_tokens_filtered)}** token valid dari {len(texts_to_process)} baris data yang diproses.")
                                st.markdown("---")
                                st.subheader("3. Hasil Visualisasi")

                                # --- VISUALISASI N-GRAM ---
                                viz_cols = st.columns(max(run_unigram, run_bigram, run_trigram)) # Jumlah kolom sesuai N-gram yg aktif
                                current_col = 0

                                # --- UNIGRAM ---
                                if run_unigram:
                                    with st.spinner("Membuat visualisasi Unigram..."):
                                        unigram_counts = Counter(all_tokens_filtered)
                                        if unigram_counts:
                                            st.markdown("#### Unigram")
                                            fig_wc_uni = create_wordcloud_fig(unigram_counts, "Unigram")
                                            if fig_wc_uni:
                                                # Simpan base64 SEBELUM ditampilkan/ditutup
                                                st.session_state.image_data['unigram_wc'] = fig_to_base64(fig_wc_uni)
                                                # Tampilkan gambar dari base64 agar konsisten
                                                if st.session_state.image_data['unigram_wc']:
                                                    st.image(st.session_state.image_data['unigram_wc'], caption="Word Cloud Unigram", use_column_width=True)
                                                else:
                                                     st.warning("Gagal membuat atau menyimpan Word Cloud Unigram.")

                                            else:
                                                st.info("Tidak cukup data untuk Word Cloud Unigram.")
                                        else:
                                            st.info("Tidak ada data Unigram untuk divisualisasikan.")
                                    st.markdown("---")


                                # --- BIGRAM ---
                                if run_bigram:
                                     with st.spinner("Membuat visualisasi Bigram..."):
                                        bigram_counts = generate_ngrams_freq(all_tokens_filtered, 2)
                                        if bigram_counts:
                                            st.markdown("#### Bigram")
                                            col_bi1, col_bi2 = st.columns(2)
                                            with col_bi1:
                                                st.markdown("**Word Cloud**")
                                                fig_wc_bi = create_wordcloud_fig(bigram_counts, "Bigram")
                                                if fig_wc_bi:
                                                    st.session_state.image_data['bigram_wc'] = fig_to_base64(fig_wc_bi)
                                                    if st.session_state.image_data['bigram_wc']:
                                                         st.image(st.session_state.image_data['bigram_wc'], caption="Word Cloud Bigram", use_column_width=True)
                                                    else: st.warning("Gagal membuat/menyimpan Word Cloud Bigram.")
                                                else: st.info("Tidak dapat membuat Word Cloud Bigram.")
                                            with col_bi2:
                                                st.markdown("**Frekuensi Teratas**")
                                                fig_bar_bi = create_barplot_fig(bigram_counts, "Bigram")
                                                if fig_bar_bi:
                                                    st.session_state.image_data['bigram_bar'] = fig_to_base64(fig_bar_bi)
                                                    if st.session_state.image_data['bigram_bar']:
                                                         st.image(st.session_state.image_data['bigram_bar'], caption="Frekuensi Bigram Teratas", use_column_width=True)
                                                    else: st.warning("Gagal membuat/menyimpan Bar Plot Bigram.")
                                                else: st.info("Tidak dapat membuat Bar Plot Bigram.")
                                        else:
                                            st.info("Tidak cukup data untuk analisis Bigram (minimal 2 token per teks diperlukan).")
                                     st.markdown("---")


                                # --- TRIGRAM ---
                                if run_trigram:
                                    with st.spinner("Membuat visualisasi Trigram..."):
                                        trigram_counts = generate_ngrams_freq(all_tokens_filtered, 3)
                                        if trigram_counts:
                                            st.markdown("#### Trigram")
                                            col_tri1, col_tri2 = st.columns(2)
                                            with col_tri1:
                                                st.markdown("**Word Cloud**")
                                                fig_wc_tri = create_wordcloud_fig(trigram_counts, "Trigram")
                                                if fig_wc_tri:
                                                    st.session_state.image_data['trigram_wc'] = fig_to_base64(fig_wc_tri)
                                                    if st.session_state.image_data['trigram_wc']:
                                                         st.image(st.session_state.image_data['trigram_wc'], caption="Word Cloud Trigram", use_column_width=True)
                                                    else: st.warning("Gagal membuat/menyimpan Word Cloud Trigram.")
                                                else: st.info("Tidak dapat membuat Word Cloud Trigram.")
                                            with col_tri2:
                                                st.markdown("**Frekuensi Teratas**")
                                                fig_bar_tri = create_barplot_fig(trigram_counts, "Trigram")
                                                if fig_bar_tri:
                                                    st.session_state.image_data['trigram_bar'] = fig_to_base64(fig_bar_tri)
                                                    if st.session_state.image_data['trigram_bar']:
                                                         st.image(st.session_state.image_data['trigram_bar'], caption="Frekuensi Trigram Teratas", use_column_width=True)
                                                    else: st.warning("Gagal membuat/menyimpan Bar Plot Trigram.")
                                                else: st.info("Tidak dapat membuat Bar Plot Trigram.")
                                        else:
                                            st.info("Tidak cukup data untuk analisis Trigram (minimal 3 token per teks diperlukan).")
                                    st.markdown("---")

                                # Tandai analisis selesai jika setidaknya satu gambar berhasil dibuat & disimpan
                                if any(st.session_state.image_data.values()):
                                    st.session_state.analysis_done = True
                                    st.success("Analisis dan visualisasi selesai!")
                                else:
                                    st.warning("Analisis selesai, tetapi tidak ada visualisasi yang berhasil dibuat atau disimpan.")
                                    st.session_state.analysis_done = False # Pastikan state benar

        # --- Bagian Insight AI (Muncul jika analisis selesai) ---
        # Taruh di luar blok tombol analisis agar tetap terlihat setelah analisis selesai
        if st.session_state.get('analysis_done', False):
            st.markdown("---")
            st.subheader("4. Dapatkan Insight dari AI")
            if not api_key:
                st.warning("⚠️ Masukkan API Key OpenRouter Anda di sidebar untuk mengaktifkan fitur ini.")
            elif not model_id:
                 st.warning("⚠️ Pilih model AI Multi-Modal di sidebar.")
            else:
                if st.button("✨ Generate Insight dari Visualisasi", key="generate_insight_button"):
                    images_to_send = []
                    prompt_image_description = "Berikut adalah visualisasi hasil analisis N-gram dari data teks:\n"
                    image_index = 1
                    # Urutkan berdasarkan N-gram dan tipe plot untuk konsistensi prompt
                    plot_order = ['unigram_wc', 'bigram_wc', 'bigram_bar', 'trigram_wc', 'trigram_bar']
                    plot_types_desc = {
                        'unigram_wc': "Gambar {}: Word Cloud Unigram (kata tunggal paling umum).",
                        'bigram_wc': "Gambar {}: Word Cloud Bigram (pasangan kata paling umum).",
                        'bigram_bar': "Gambar {}: Bar Plot Frekuensi Bigram Teratas.",
                        'trigram_wc': "Gambar {}: Word Cloud Trigram (rangkaian tiga kata paling umum).",
                        'trigram_bar': "Gambar {}: Bar Plot Frekuensi Trigram Teratas."
                    }

                    for key in plot_order:
                        # Hanya kirim gambar yang ada dan valid
                        img_data = st.session_state.image_data.get(key)
                        if img_data and isinstance(img_data, str) and img_data.startswith('data:image'):
                            images_to_send.append(img_data)
                            prompt_image_description += f"- {plot_types_desc[key].format(image_index)}\n"
                            image_index += 1

                    if not images_to_send:
                        st.error("Tidak ada gambar visualisasi valid yang tersimpan untuk dianalisis oleh AI.")
                    else:
                        # --- Tambahkan Konteks Filter ke Prompt ---
                        filter_context = ""
                        # Ambil dari state yang disimpan saat analisis dijalankan
                        active_sents = st.session_state.active_filters.get('sentiments', [])
                        sent_col = st.session_state.active_filters.get('sentiment_col')
                        txt_col = st.session_state.active_filters.get('text_col', 'yang dipilih') # Fallback

                        if active_sents and sent_col:
                            # Filter aktif
                            filter_context = (f"**Konteks Analisis:** Visualisasi ini dibuat HANYA dari data teks (kolom '{txt_col}') "
                                              f"dengan sentimen '{', '.join(active_sents)}' "
                                              f"(berdasarkan kolom '{sent_col}').")
                        elif sent_col:
                             # Kolom sentimen ada, tapi tidak ada filter aktif (analisis semua sentimen)
                             filter_context = (f"**Konteks Analisis:** Visualisasi ini dibuat dari SEMUA data teks (kolom '{txt_col}'). "
                                               f"Kolom sentimen ('{sent_col}') ada, tetapi TIDAK ADA filter sentimen yang diterapkan untuk visualisasi ini.")
                        else:
                            # Tidak ada kolom sentimen sama sekali
                            filter_context = (f"**Konteks Analisis:** Visualisasi ini dibuat dari SEMUA data teks (kolom '{txt_col}'). "
                                              "Tidak ada informasi atau filter sentimen yang digunakan.")

                        # Susun prompt akhir yang lebih detail
                        final_prompt = (
                            f"{filter_context}\n\n"
                            f"{prompt_image_description}\n"
                            "**Instruksi untuk AI (Analis Data Ahli):**\n"
                            "Anda adalah seorang analis data yang sangat terampil dalam menginterpretasi visualisasi data teks (word clouds, bar plots n-gram) dan memahami konteksnya. Berdasarkan gambar-gambar visualisasi yang diberikan dan konteks analisis di atas:\n\n"
                            "1.  **Analisis Komprehensif:** Tinjau setiap gambar dengan cermat. Identifikasi kata atau frasa (unigram, bigram, trigram) yang paling dominan (paling besar di word cloud, paling tinggi di bar plot).\n"
                            "2.  **Identifikasi Tema & Pola:** Apa tema atau topik utama yang muncul dari kata/frasa dominan ini? Apakah ada pola menarik (misalnya, bigram yang sering muncul bersamaan, kata kunci yang sama di beberapa n-gram)?\n"
                            "3.  **Interpretasi Kontekstual:** Jelaskan makna dari temuan Anda. **Sangat penting:** Kaitkan interpretasi Anda dengan **konteks analisis** yang disebutkan di awal (apakah data difilter berdasarkan sentimen tertentu atau mencakup semua data?). Apa cerita yang bisa disimpulkan dari visualisasi ini, mengingat konteks tersebut?\n"
                            "4.  **Insight & Kesimpulan:** Berikan kesimpulan utama atau insight paling penting yang dapat ditarik. Apa implikasi dari pola yang Anda temukan? Adakah rekomendasi atau area investigasi lebih lanjut yang bisa disarankan (berdasarkan visualisasi saja)?\n\n"
                            "**Format Jawaban:** Sajikan analisis Anda dalam format poin-poin yang jelas, terstruktur, dan mudah dibaca. Gunakan bahasa yang profesional namun mudah dipahami. Fokus HANYA pada informasi yang terlihat di gambar dan konteks yang diberikan."
                        )

                        # Debug: Tampilkan prompt final (opsional)
                        # with st.expander("Debug: Prompt Final ke AI"):
                        #    st.text_area("Prompt:", final_prompt, height=300)
                        #    st.write(f"Jumlah gambar dikirim: {len(images_to_send)}")


                        with st.spinner(f"🤖 Menghubungi model {model_id} untuk mendapatkan insight... Ini mungkin memerlukan waktu..."):
                            insight = analyze_image_with_openrouter(api_key, model_id, final_prompt, images_to_send)

                        st.markdown("---") # Pemisah sebelum hasil
                        if insight:
                            st.success("✅ Insight dari Model AI:")
                            st.markdown(insight) # Tampilkan hasil markdown dari AI
                        else:
                            # Pesan error sudah ditampilkan di dalam fungsi analyze_image_with_openrouter
                            st.error("Gagal mendapatkan insight dari model AI. Silakan cek pesan error di atas atau coba lagi.")

    except pd.errors.EmptyDataError:
        st.error("File CSV yang diunggah kosong atau tidak valid.")
    except KeyError as e:
        st.error(f"Error: Kolom '{e}' tidak ditemukan dalam file CSV. Pastikan nama kolom sudah benar dan sesuai dengan yang dipilih.")
    except Exception as e:
        st.error(f"Terjadi error saat memproses file atau data: {e}")
        st.error("Tips: Pastikan file CSV valid, memiliki header (baris pertama nama kolom), encoding UTF-8, dan kolom yang dipilih memang ada.")
        import traceback
        st.exception(traceback.format_exc()) # Tampilkan traceback lengkap untuk debug

elif not uploaded_file and not st.session_state.get('last_uploaded_filename'):
    # Tampilkan pesan ini hanya jika belum pernah ada file yang diupload
    st.info("☝️ Mulai dengan mengunggah file CSV di atas.")

# --- Footer Opsional ---
st.markdown("---")
st.caption("Aplikasi Analisis Teks & Visualisasi N-Gram dengan Insight AI | Dibuat dengan Streamlit")
