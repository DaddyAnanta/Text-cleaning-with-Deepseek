import streamlit as st

st.set_page_config(page_title="Prediksi Sentimen", layout="wide")

st.title("Prediksi Sentimen Teks")

st.markdown("""
Halaman ini menjelaskan proses untuk melakukan prediksi sentimen pada data teks yang telah melalui tahap analisis lanjutan.
Proses ini menggunakan model _transformer-based_ (RoBERTa) yang cukup berat secara komputasi dan **sangat disarankan untuk dijalankan di lingkungan yang memiliki akses ke GPU**, seperti Google Colaboratory (Colab), untuk performa yang optimal dan waktu pemrosesan yang lebih cepat.

**Alasan Menjalankan di Google Colab:**
- **Akses GPU Gratis:** Google Colab menyediakan akses ke GPU (seperti Tesla T4) secara gratis yang dapat mempercepat training dan inferensi model _deep learning_ secara signifikan.
- **Lingkungan Pra-instal:** Banyak _library_ populer untuk _data science_ dan _machine learning_ sudah terinstal atau mudah diinstal di Colab.
- **Kemudahan Berbagi:** Notebook Colab mudah dibagikan dan dijalankan oleh orang lain.

Di bawah ini adalah kode Python yang dapat Anda gunakan di Google Colab untuk melakukan analisis sentimen pada data Anda.
""")

st.header("Langkah-langkah Menjalankan Analisis Sentimen di Google Colab:")

st.markdown("""
1.  **Siapkan Data Anda:**
    * Pastikan Anda memiliki file `hasil_analisis_lengkap.csv` (atau file dengan nama yang sesuai yang berisi kolom `text_final` dari hasil analisis lanjutan di aplikasi ini).
    * File ini adalah output dari tahap "Analisis Teks Lanjutan" pada halaman utama aplikasi.

2.  **Buka Google Colab:**
    * Kunjungi [Google Colaboratory](https://colab.research.google.com/).
    * Buat Notebook baru atau buka Notebook yang sudah ada.

3.  **Atur Runtime ke GPU:**
    * Di menu Colab, pilih `Runtime` -> `Change runtime type`.
    * Pada bagian `Hardware accelerator`, pilih `GPU` dan klik `Save`.

4.  **Unggah File Data Anda:**
    * Di panel sebelah kiri Colab, klik ikon folder untuk membuka _Files_.
    * Klik tombol `Upload to session storage` (ikon file dengan panah ke atas) dan pilih file `hasil_analisis_lengkap.csv` Anda.
    * *Catatan: File yang diunggah ke _session storage_ akan hilang ketika sesi Colab berakhir. Untuk penyimpanan permanen, Anda bisa menghubungkan Google Drive Anda.*

5.  **Instalasi Library yang Diperlukan:**
    * Jalankan sel kode berikut di Colab untuk menginstal `transformers` dan `torch` (jika belum ada atau versi tertentu diperlukan). Pandas biasanya sudah tersedia.
    ```python
    !pip install pandas transformers torch
    ```

6.  **Jalankan Kode Analisis Sentimen:**
    * Salin dan tempel kode Python di bawah ini ke sel baru di Notebook Colab Anda.
    * Sesuaikan nama file pada baris `data = pd.read_csv("/content/nama_file_anda.csv")` agar sesuai dengan nama file yang Anda unggah (biasanya akan menjadi `/content/hasil_analisis_lengkap.csv` jika Anda mengikuti nama dari aplikasi ini).
    * Kode ini akan membaca data, melakukan analisis sentimen pada kolom `text_final`, menambahkan kolom sentimen baru, dan menyimpan hasilnya ke file `Wordcloud.csv`.
""")

st.subheader("Kode Python untuk Analisis Sentimen di Google Colab:")
kode_colab = """
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# Pastikan nama file CSV sudah sesuai dengan yang Anda unggah ke Colab
# Biasanya, jika Anda unggah langsung, path-nya adalah "/content/nama_file.csv"
try:
    # Ganti "hasil_analisis_lengkap.csv" dengan nama file Anda jika berbeda
    data = pd.read_csv("/content/hasil_analisis_lengkap.csv") 
except FileNotFoundError:
    print("File tidak ditemukan. Pastikan nama file dan path sudah benar.")
    print("Contoh path jika diunggah langsung ke Colab: '/content/nama_file_anda.csv'")
    # Anda bisa menghentikan eksekusi di sini atau menangani error lebih lanjut
    raise

# Opsional: Jika Anda hanya ingin memproses sebagian data (misalnya 20 baris terakhir)
# data = data.tail(20)

# Periksa apakah kolom 'text_final' ada dalam data
if "text_final" not in data.columns:
    print("Error: Kolom 'text_final' tidak ditemukan dalam file CSV.")
    print("Pastikan file CSV Anda adalah output dari tahap analisis lanjutan dan memiliki kolom 'text_final'.")
    # Anda bisa menghentikan eksekusi di sini
    raise KeyError("Kolom 'text_final' tidak ditemukan.")

# Hapus baris dengan nilai NaN di kolom 'text_final' jika ada, karena pipeline akan error
data.dropna(subset=['text_final'], inplace=True)
# Pastikan tipe data adalah string
data['text_final'] = data['text_final'].astype(str)


# Load model dan tokenizer dari CardiffNLP
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest" # Menggunakan model terbaru
# Jika ingin versi spesifik seperti di kode awal Anda:
# model_name = "cardiffnlp/twitter-roberta-base-sentiment" 

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Cek ketersediaan GPU dan pindahkan model ke GPU jika tersedia
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"Menggunakan device: {device}")

# Buat pipeline sentiment dengan device yang sesuai
# batch_size bisa disesuaikan tergantung memori GPU
sentiment_pipeline = pipeline(
    "sentiment-analysis", 
    model=model, 
    tokenizer=tokenizer, 
    device=0 if device=="cuda" else -1, # device=0 untuk GPU pertama, -1 untuk CPU
    return_all_scores=False,
    batch_size=8 # Default batch_size untuk pipeline adalah 1, menaikkannya bisa mempercepat
)

# Lakukan analisis sentimen pada list teks dari kolom "text_final"
# Pastikan tidak ada nilai non-string atau NaN
text_list = data["text_final"].tolist()

# Lakukan prediksi
print(f"Memulai analisis sentimen untuk {len(text_list)} teks...")
results = sentiment_pipeline(text_list)
print("Analisis sentimen selesai.")

# Pemetaan label ke nama sentimen yang lebih deskriptif
# Model CardiffNLP biasanya menghasilkan label seperti 'LABEL_0', 'LABEL_1', 'LABEL_2'
# atau 'negative', 'neutral', 'positive' tergantung versi/konfigurasi.
# Jika outputnya sudah 'negative', 'neutral', 'positive', map ini tidak terlalu diperlukan,
# tapi aman untuk memilikinya jika outputnya adalah LABEL_X.

# Cek dulu format label dari hasil pertama
first_label = results[0]['label'] if results else None
print(f"Contoh label dari model: {first_label}")

if first_label and first_label.startswith("LABEL_"):
    label_map = {
        "LABEL_0": "Negatif", # Atau sesuaikan dengan output model (misal, 'negative')
        "LABEL_1": "Netral",  # Atau sesuaikan (misal, 'neutral')
        "LABEL_2": "Positif"  # Atau sesuaikan (misal, 'positive')
    }
    data["sentiment_label"] = [r['label'] for r in results]
    data["sentiment"] = data["sentiment_label"].map(label_map)
    # Jika ada label yang tidak ada di map, akan menjadi NaN, bisa diisi default
    data["sentiment"].fillna("Tidak Diketahui", inplace=True)
else:
    # Jika label sudah deskriptif (misal 'positive', 'negative', 'neutral')
    data["sentiment"] = [r['label'].capitalize() for r in results]

# Tampilkan beberapa baris hasil data dengan sentimen
print("\\nContoh data dengan sentimen:")
print(data[["text_final", "sentiment"]].head())

# Simpan hasil ke file CSV baru
output_filename = "hasil_prediksi_sentimen.csv"
data.to_csv(output_filename, index=False)
print(f"\\nData dengan prediksi sentimen telah disimpan ke {output_filename}")

# Anda bisa menambahkan kode untuk mengunduh file ini dari Colab
# from google.colab import files
# files.download(output_filename)
"""
st.code(kode_colab, language="python")

st.markdown("""
7.  **Unduh Hasil:**
    * Setelah kode selesai dijalankan, file `hasil_prediksi_sentimen.csv` akan muncul di panel _Files_ di Colab.
    * Anda dapat mengunduhnya dengan mengklik kanan pada nama file dan memilih `Download`.
    * Jika Anda uncomment baris `files.download(output_filename)` di akhir kode, file akan otomatis terunduh ke komputer Anda.
""")

st.markdown("---")
st.info("Pastikan untuk memeriksa output dari model sentimen (misalnya, 'LABEL_0', 'LABEL_1', 'LABEL_2' atau 'negative', 'neutral', 'positive') dan sesuaikan `label_map` dalam kode jika diperlukan agar sesuai dengan output model yang Anda gunakan.")
