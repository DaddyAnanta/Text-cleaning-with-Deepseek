import streamlit as st
import pandas as pd
import requests
import time
import io
import re
import math # Tambahkan import math
import traceback # Untuk menampilkan traceback error

from utils.cleaning import clean_text
from utils.api_handler import call_openrouter_api
from utils.nlp_tools import tokenize_text, remove_stopwords, apply_stemming, apply_lemmatization
from utils.translate import translate_with_googletrans

# --- STREAMLIT CONFIG ---
st.set_page_config(page_title="Perbaikan Teks Otomatis", layout="wide")
st.title("Perbaikan Teks ke Bahasa Indonesia Formal")

# --- SESSION STATE INITIALIZATION ---
if "hasil_fix_kolom" not in st.session_state:
    st.session_state.hasil_fix_kolom = "text_fix"

if "batch_results" not in st.session_state:
    st.session_state.batch_results = {}

if "merged_data" not in st.session_state:
    st.session_state.merged_data = None

if "processed_batches" not in st.session_state:
    st.session_state.processed_batches = set()

if 'original_data' not in st.session_state:
    st.session_state.original_data = None

# --- FUNGSI PANGGILAN API ---
# (Fungsi call_openrouter_api diasumsikan ada di utils.api_handler atau di script utama)
# Pastikan fungsi ini didefinisikan atau diimpor dengan benar.
# Contoh placeholder jika belum ada:
# def call_openrouter_api(api_key, prompt):
#     # Logika untuk memanggil API
#     # Ini hanya contoh, sesuaikan dengan implementasi Anda
#     try:
#         # response = requests.post(...)
#         # return response.json()['choices'][0]['message']['content']
#         return f"Hasil perbaikan untuk: {prompt.splitlines()[-1]}" # Dummy response
#     except Exception as e:
#         st.warning(f"Error API: {e}")
#         return None

# --- FUNGSI FORMAT DAN PERBAIKI ---
def format_dan_perbaiki(teks_asli, api_key):
    prompt_template =  """
    ### Tugas Utama
    Anda adalah asisten ahli multibahasa. Tugas Anda adalah memproses teks input Bahasa Indonesia melalui dua tahap:
    1.  **Tahap 1 (Pemrosesan Internal):** Terima teks Bahasa Indonesia, perbaiki kesalahan ejaan/tata bahasa, perluas singkatan, ubah gaya bahasa menjadi formal dan baku, ubah semua huruf menjadi huruf kecil, hapus semua tanda baca, dan konversi semua angka menjadi kata dalam Bahasa Indonesia. Pastikan makna asli teks tetap terjaga selama proses ini.

    ### Ketentuan Detail untuk Tahap 1 (Pemrosesan Internal Bahasa Indonesia):
    * **Perbaikan & Formalisasi:** Koreksi ejaan, tata bahasa. Ubah singkatan ke bentuk penuh (misal: 'yg' jadi 'yang'). Ubah gaya informal ke formal.
    * **Format:** Seluruh teks hasil proses ini harus dalam huruf kecil dan tanpa tanda baca sama sekali.
    * **Konversi Angka:** Angka numerik diubah jadi kata (misal: '2' -> 'dua', '100' -> 'seratus', '5jt' -> 'lima juta').
    * **Teks Baku:** Jika input sudah baku dan formal, langsung terapkan format (huruf kecil, tanpa tanda baca, angka jadi kata) sebelum ke Tahap 2.

    ### Output Akhir (HANYA Hasil Tahap 1) tanpa memberikan komentar apapun dan biarkan konsistensi prompt ini terus berjalan sampai seterusnya:
    """
    prompt = f'{prompt_template}\n\n### Input Teks (Bahasa Indonesia):\n"{teks_asli}"\n\n'
    hasil = call_openrouter_api(api_key, prompt)
    if hasil:
        hasil = hasil.strip()
        if ":" in hasil[:15]:
            hasil = hasil.split(":", 1)[-1].strip()
        if hasil.startswith('"') and hasil.endswith('"'):
            hasil = hasil[1:-1]
        return hasil
    st.warning(f"API tidak mengembalikan hasil untuk: '{teks_asli}'. Menggunakan teks asli.")
    return teks_asli

# --- MAIN PROCESSING LOGIC ---
input_file = st.file_uploader("Upload file CSV", type="csv")

if input_file:
    try:
        data = pd.read_csv(input_file)
        if data.empty:
            st.warning("File CSV kosong. Harap unggah data yang valid.")
            st.stop()

        if st.session_state.original_data is None: # Hanya set jika belum ada
             st.session_state.original_data = data.copy()


        st.write(f"**Jumlah baris data keseluruhan: {len(st.session_state.original_data)}**")
        st.dataframe(st.session_state.original_data.head())

        kolom_terpilih = st.selectbox("Pilih kolom teks sumber", st.session_state.original_data.columns.tolist(), key="kolom_sumber_select")

        if kolom_terpilih not in st.session_state.original_data.columns:
            st.error(f"Kolom '{kolom_terpilih}' tidak ditemukan dalam file CSV.")
            st.stop()

        # Inisialisasi 'text_clean' di data asli jika belum ada atau file berubah
        # Ini akan menjadi dasar untuk pembersihan awal
        # Inisialisasi 'text_clean' di data asli jika belum ada atau file berubah
        # Ini akan menjadi dasar untuk pembersihan awal
        if 'text_clean_base' not in st.session_state or input_file.name != st.session_state.get('current_file_name'):
            st.session_state.current_file_name = input_file.name # Simpan nama file saat ini
            if st.session_state.original_data is not None and kolom_terpilih in st.session_state.original_data.columns:
                st.session_state.text_clean_base = st.session_state.original_data[kolom_terpilih].astype(str).copy()
            else:
                st.warning("Data asli atau kolom terpilih belum siap untuk inisialisasi text_clean_base.")
                st.session_state.text_clean_base = pd.Series(dtype=str) # Inisialisasi kosong

            # Reset juga batch results jika file berubah
            st.session_state.batch_results = {}
            st.session_state.processed_batches = set()
            st.session_state.merged_data = None # <--- merged_data di-set menjadi None

            # Reset status analisis lanjutan (flag)
            if 'tokenization_done' in st.session_state: del st.session_state.tokenization_done
            if 'nlp_analysis_done' in st.session_state: del st.session_state.nlp_analysis_done
            if 'last_tokenized_source_col' in st.session_state: del st.session_state.last_tokenized_source_col
            
            # PASTIKAN BARIS-BARIS BERIKUT (ATAU YANG SERUPA) DIHAPUS ATAU DIKOMENTARI:
            # Baris inilah (atau yang serupa untuk 'text_tokenized', 'text_final') yang menyebabkan error jika 'merged_data' adalah None
            # 
            # if 'tweet_english' in st.session_state.get('merged_data', pd.DataFrame()).columns: 
            #     del st.session_state.merged_data['tweet_english']
            # if 'text_tokenized' in st.session_state.get('merged_data', pd.DataFrame()).columns:
            #     del st.session_state.merged_data['text_tokenized']
            # if 'text_final' in st.session_state.get('merged_data', pd.DataFrame()).columns:
            #     del st.session_state.merged_data['text_final']

            st.info("Nama file berubah atau file baru diunggah. State pemrosesan dan analisis lanjutan telah direset.")
            st.rerun() # Paksa rerun untuk memastikan UI konsisten setelah reset

        # Buat salinan dari text_clean_base untuk diproses
        data_processing = st.session_state.original_data.copy()
        data_processing["text_clean"] = st.session_state.text_clean_base.copy()


        with st.expander("Opsi Pembersihan Awal (Regex & Case Folding)"):
            # Gunakan session state untuk checkbox pembersihan awal
            if 'apply_regex_clean' not in st.session_state: st.session_state.apply_regex_clean = False
            if 'apply_case_folding' not in st.session_state: st.session_state.apply_case_folding = False

            st.session_state.apply_regex_clean = st.checkbox("Terapkan Regex Cleaning", value=st.session_state.apply_regex_clean, key="cb_regex_clean")
            st.session_state.apply_case_folding = st.checkbox("Terapkan Case Folding", value=st.session_state.apply_case_folding, key="cb_case_folding")

            temp_text_clean = st.session_state.text_clean_base.copy() # Mulai dari basis setiap kali
            applied_ops = []

            if st.session_state.apply_regex_clean:
                try:
                    temp_text_clean = temp_text_clean.apply(clean_text)
                    applied_ops.append("Regex cleaning")
                except NameError:
                    st.warning("Fungsi `clean_text` tidak ditemukan.")
                except Exception as e:
                    st.error(f"Error saat menerapkan regex: {e}")

            if st.session_state.apply_case_folding:
                temp_text_clean = temp_text_clean.str.lower()
                applied_ops.append("Case folding")

            if applied_ops:
                st.info(f"Opsi diterapkan: {', '.join(applied_ops)}.")
            # Simpan hasil pembersihan awal ke data_processing untuk digunakan oleh Deepseek
            data_processing["text_clean"] = temp_text_clean
            st.write("Contoh data setelah pembersihan awal:")
            st.dataframe(data_processing[[kolom_terpilih, "text_clean"]].head())
            # Simpan text_clean yang sudah diproses ke session state untuk digunakan oleh batch
            st.session_state.data_for_batch_processing = data_processing.copy()


        OPENROUTER_API_KEY = st.text_input("Masukkan API Key OpenRouter Anda", type="password", key="api_key_input")
        st.markdown("---")
        st.subheader("Proses Perbaikan Teks dengan Deepseek (per Batch)")

        if not OPENROUTER_API_KEY:
            st.warning("Masukkan API Key OpenRouter untuk memulai pemrosesan.")
        else:
            batch_size = 200
            # Gunakan data yang sudah melalui pembersihan awal dari session state
            if 'data_for_batch_processing' in st.session_state:
                data_to_batch = st.session_state.data_for_batch_processing
                total_rows = len(data_to_batch)
                num_batches = math.ceil(total_rows / batch_size)

                st.write(f"Data akan dibagi menjadi **{num_batches}** batch (maksimal {batch_size} baris per batch).")

                cols_per_row = 5
                col_idx = 0
                row_cols = st.columns(cols_per_row)

                for i in range(num_batches):
                    start_index = i * batch_size
                    end_index = min((i + 1) * batch_size, total_rows)
                    batch_label = f"{start_index}-{end_index - 1}"
                    button_label = f"Proses Batch: {batch_label}"
                    is_processed = batch_label in st.session_state.processed_batches

                    current_col = row_cols[col_idx]
                    with current_col:
                        if st.button(button_label, key=f"batch_{batch_label}", disabled=is_processed):
                            # Ambil data_for_batch_processing yang relevan
                            data_batch_slice = st.session_state.data_for_batch_processing.iloc[start_index:end_index].copy() # salin slice

                            st.info(f"Memulai pemrosesan untuk batch {batch_label}...")
                            hasil_list = []
                            progress_placeholder = st.empty()
                            progress_bar = progress_placeholder.progress(0)
                            progress_text_placeholder = st.empty()
                            progress_text_placeholder.text(f"Memulai pemrosesan batch {batch_label}...")

                            num_rows_in_batch = len(data_batch_slice)
                            start_time_batch = time.time()

                            for idx, (original_index, row) in enumerate(data_batch_slice.iterrows()):
                                teks_untuk_diproses = row["text_clean"]
                                start_time_row = time.time()
                                hasil = format_dan_perbaiki(teks_untuk_diproses, OPENROUTER_API_KEY)
                                end_time_row = time.time()

                                if hasil is None:
                                    st.error(f"API gagal untuk baris index {original_index}. Menggunakan teks asli.")
                                    hasil = teks_untuk_diproses
                                hasil_list.append(hasil)

                                progress = (idx + 1) / num_rows_in_batch
                                elapsed_time_row = end_time_row - start_time_row
                                progress_bar.progress(progress)
                                progress_text_placeholder.text(
                                    f"Batch {batch_label}: Memproses baris {idx + 1}/{num_rows_in_batch} (Index asli: {original_index}) - Waktu: {elapsed_time_row:.2f} detik"
                                )
                                time.sleep(1.5)

                            end_time_batch = time.time()
                            total_time_batch = end_time_batch - start_time_batch

                            # Buat DataFrame untuk hasil batch ini DENGAN SEMUA KOLOM ASLI
                            # dan tambahkan kolom hasil perbaikan
                            batch_result_df = data_batch_slice.copy() # Mulai dengan semua kolom dari slice
                            batch_result_df[st.session_state.hasil_fix_kolom] = hasil_list

                            st.session_state.batch_results[batch_label] = batch_result_df
                            st.session_state.processed_batches.add(batch_label)

                            progress_placeholder.empty()
                            progress_text_placeholder.empty()
                            st.success(f"Batch {batch_label} selesai diproses dalam {total_time_batch:.2f} detik!")
                            st.dataframe(batch_result_df.head()) # Tampilkan head dari batch yang diproses

                            # Gabungkan hasil segera setelah batch selesai dan simpan ke merged_data
                            if st.session_state.batch_results:
                                list_of_batch_dfs = [st.session_state.batch_results[b_label] for b_label in sorted(st.session_state.batch_results.keys(), key=lambda x: int(x.split('-')[0]))]
                                if list_of_batch_dfs:
                                    try:
                                        merged_data_temp = pd.concat(list_of_batch_dfs)
                                        # Pastikan semua kolom dari original_data ada, jika tidak tambahkan dengan NaN atau nilai default
                                        for col_orig in st.session_state.original_data.columns:
                                            if col_orig not in merged_data_temp.columns:
                                                merged_data_temp[col_orig] = pd.NA # Atau st.session_state.original_data[col_orig] jika sesuai
                                        # Jaga urutan kolom seperti data asli, lalu tambahkan kolom baru
                                        final_cols_order = list(st.session_state.original_data.columns) + [c for c in merged_data_temp.columns if c not in st.session_state.original_data.columns]
                                        merged_data_temp = merged_data_temp.reindex(columns=final_cols_order)

                                        st.session_state.merged_data = merged_data_temp.sort_index()
                                        # Reset analisis lanjutan jika data Deepseek berubah
                                        if 'tokenization_done' in st.session_state: del st.session_state.tokenization_done
                                        if 'nlp_analysis_done' in st.session_state: del st.session_state.nlp_analysis_done
                                        if 'tweet_english' in st.session_state.merged_data.columns:
                                            del st.session_state.merged_data['tweet_english']
                                        if 'text_tokenized' in st.session_state.merged_data.columns:
                                            del st.session_state.merged_data['text_tokenized']
                                        if 'text_final' in st.session_state.merged_data.columns:
                                            del st.session_state.merged_data['text_final']

                                    except Exception as e_concat:
                                        st.error(f"Gagal menggabungkan hasil batch setelah pemrosesan: {e_concat}")
                                        st.error(traceback.format_exc())
                            st.rerun()

                        if is_processed:
                            st.success(f"Batch {batch_label} ✅")

                    col_idx = (col_idx + 1) % cols_per_row
                    if col_idx == 0 and i < num_batches -1 :
                        row_cols = st.columns(cols_per_row)
            else:
                st.info("Data untuk pemrosesan batch belum siap. Pastikan file diunggah dan opsi pembersihan awal dipilih.")


            if st.session_state.merged_data is not None and not st.session_state.merged_data.empty:
                st.markdown("---")
                st.subheader("Hasil Gabungan dari Batch yang Telah Diproses")
                st.dataframe(st.session_state.merged_data)
                st.info(f"Total baris yang telah diproses dan digabung: {len(st.session_state.merged_data)}")

                st.subheader("Unduh Hasil Perbaikan")
                buffer = io.StringIO()
                st.session_state.merged_data.to_csv(buffer, index=False)
                st.download_button(
                    label="Unduh Data Perbaikan (.csv)",
                    data=buffer.getvalue(),
                    file_name="hasil_perbaikan_deepseek.csv",
                    mime="text/csv",
                    key="download_deepseek"
                )
            elif st.session_state.batch_results : # Jika ada batch result tapi belum tergabung (seharusnya tidak terjadi dengan logika baru)
                 st.info("Memproses penggabungan batch...")
            else:
                st.info("Belum ada batch yang diproses atau digabungkan.")


    except pd.errors.EmptyDataError:
        st.error("File CSV yang diunggah kosong atau formatnya tidak valid.")
    except KeyError as e:
        st.error(f"Error: Kolom yang diperlukan tidak ditemukan dalam file CSV. Pastikan kolom '{e}' ada.")
    except Exception as e:
        st.error(f"Terjadi error saat memuat atau memproses file CSV: {e}")
        st.error(traceback.format_exc())

# --- Bagian Analisis Lanjutan (setelah Deepseek) ---
st.markdown("---")
st.header("Analisis Teks Lanjutan (Setelah Perbaikan Deepseek)")

if 'merged_data' in st.session_state and st.session_state.merged_data is not None and not st.session_state.merged_data.empty:
    data_untuk_lanjutan = st.session_state.merged_data.copy()
    kolom_hasil_deepseek = st.session_state.hasil_fix_kolom

    if kolom_hasil_deepseek not in data_untuk_lanjutan.columns:
        st.error(f"Kolom hasil perbaikan '{kolom_hasil_deepseek}' tidak ditemukan di data gabungan. Harap proses batch terlebih dahulu.")
        st.stop()

    st.info(f"Analisis lanjutan akan menggunakan kolom '{kolom_hasil_deepseek}' dari hasil gabungan Deepseek.")
    # Tampilkan head dari data gabungan yang akan dipakai
    st.dataframe(data_untuk_lanjutan[[kolom_terpilih, kolom_hasil_deepseek]].head())


    # --- Opsi Terjemahan (jika diperlukan) ---
    kolom_target_translate = "tweet_english"
    translate_key_cb = "translate_checkbox_lanjutan"
    if translate_key_cb not in st.session_state: st.session_state[translate_key_cb] = False

    st.session_state[translate_key_cb] = st.checkbox(
        "Terjemahkan hasil perbaikan ke Bahasa Inggris (Google Translate)",
        value=st.session_state[translate_key_cb],
        key=translate_key_cb + "_widget" # key widget unik
    )

    if st.session_state[translate_key_cb]:
        if kolom_target_translate not in data_untuk_lanjutan.columns or st.button("Ulangi Terjemahan", key="btn_ulang_translate"):
            st.warning("Proses terjemahan mungkin memerlukan waktu.")
            with st.spinner("Menerjemahkan teks..."):
                try:
                    # Pastikan translate_with_googletrans ada di utils.translate
                    data_untuk_lanjutan[kolom_target_translate] = data_untuk_lanjutan[kolom_hasil_deepseek].apply(translate_with_googletrans)
                    st.success("Proses terjemahan selesai.")
                    st.session_state.merged_data = data_untuk_lanjutan.copy() # Simpan hasil terjemahan
                    # Jika menerjemahkan, reset status tokenisasi dan NLP selanjutnya karena sumber berubah
                    if 'tokenization_done' in st.session_state: del st.session_state.tokenization_done
                    if 'nlp_analysis_done' in st.session_state: del st.session_state.nlp_analysis_done
                    st.rerun()
                except NameError:
                    st.error("Fungsi `translate_with_googletrans` tidak ditemukan. Pastikan ada di `utils/translate.py`.")
                except Exception as e:
                    st.error(f"Error saat menerjemahkan: {e}")
                    st.error(traceback.format_exc())
        elif kolom_target_translate in data_untuk_lanjutan.columns:
            st.info(f"Kolom '{kolom_target_translate}' sudah ada.")
            st.dataframe(data_untuk_lanjutan[[kolom_hasil_deepseek, kolom_target_translate]].head())
    # Jika checkbox terjemahan tidak dicentang dan kolom terjemahan ada, kita mungkin ingin menghapusnya dari pertimbangan sumber tokenisasi
    # atau biarkan pengguna memilih. Untuk saat ini, kita biarkan kolomnya jika sudah ada.


    # --- Tokenisasi ---
    st.subheader("Tokenisasi")
    tokenized_column_name = "text_tokenized"

    # Tentukan kolom sumber default untuk tokenisasi
    default_source_for_tokenization = kolom_hasil_deepseek
    if st.session_state.get(translate_key_cb, False) and kolom_target_translate in data_untuk_lanjutan.columns:
        default_source_for_tokenization = kolom_target_translate

    available_cols_for_token = [kolom_hasil_deepseek]
    if kolom_target_translate in data_untuk_lanjutan.columns:
        available_cols_for_token.append(kolom_target_translate)

    # Beri key unik pada selectbox
    kolom_sumber_tokenisasi = st.selectbox(
        "Pilih kolom untuk Tokenisasi:",
        options=list(set(available_cols_for_token)), # Pastikan unik jika ada duplikasi nama
        index=list(set(available_cols_for_token)).index(default_source_for_tokenization) if default_source_for_tokenization in list(set(available_cols_for_token)) else 0,
        key="sb_kolom_sumber_tokenisasi"
    )
    st.info(f"Tokenisasi akan dilakukan pada kolom '{kolom_sumber_tokenisasi}'.")


    if 'tokenization_done' not in st.session_state: st.session_state.tokenization_done = False
    if 'last_tokenized_source_col' not in st.session_state: st.session_state.last_tokenized_source_col = None

    if st.button("Lakukan Tokenisasi", key="btn_lakukan_tokenisasi"):
        if kolom_sumber_tokenisasi not in data_untuk_lanjutan.columns:
            st.error(f"Kolom '{kolom_sumber_tokenisasi}' tidak ditemukan. Tidak dapat melakukan tokenisasi.")
        else:
            with st.spinner("Melakukan tokenisasi..."):
                try:
                    # Pastikan tokenize_text ada di utils.nlp_tools
                    data_untuk_lanjutan[tokenized_column_name] = data_untuk_lanjutan[kolom_sumber_tokenisasi].apply(tokenize_text)
                    st.success("Tokenisasi selesai!")
                    st.session_state.tokenization_done = True
                    st.session_state.last_tokenized_source_col = kolom_sumber_tokenisasi # Catat sumber yang ditokenisasi
                    # Hapus hasil analisis NLP sebelumnya jika tokenisasi diulang atau sumbernya berubah
                    if 'text_final' in data_untuk_lanjutan.columns:
                        del data_untuk_lanjutan['text_final']
                    if 'nlp_analysis_done' in st.session_state:
                        del st.session_state.nlp_analysis_done
                    st.session_state.merged_data = data_untuk_lanjutan.copy() # Simpan hasil ke session_state
                    st.rerun()
                except NameError:
                    st.error("Fungsi `tokenize_text` tidak ditemukan. Pastikan ada di `utils/nlp_tools.py`.")
                except Exception as e:
                    st.error(f"Error saat tokenisasi: {e}")
                    st.error(traceback.format_exc())

    # Tampilkan hasil tokenisasi jika sudah dilakukan
    if st.session_state.get('tokenization_done') and tokenized_column_name in data_untuk_lanjutan.columns and st.session_state.last_tokenized_source_col == kolom_sumber_tokenisasi:
        st.write("Contoh hasil tokenisasi:")
        st.dataframe(data_untuk_lanjutan[[kolom_sumber_tokenisasi, tokenized_column_name]].head())
    elif tokenized_column_name in data_untuk_lanjutan.columns and st.session_state.last_tokenized_source_col != kolom_sumber_tokenisasi:
        st.warning(f"Kolom '{tokenized_column_name}' ada, tetapi dibuat dari sumber ('{st.session_state.last_tokenized_source_col}') yang berbeda dari pilihan saat ini ('{kolom_sumber_tokenisasi}'). Lakukan tokenisasi ulang jika perlu.")


    # --- Analisis Lanjutan (Stopword, Stemming, Lemmatization) ---
    # Cek apakah tokenisasi telah selesai dan sumbernya sesuai DENGAN YANG DIPILIH SAAT INI
    can_proceed_to_nlp = (
        st.session_state.get('tokenization_done', False) and
        tokenized_column_name in data_untuk_lanjutan.columns and
        st.session_state.get('last_tokenized_source_col') == kolom_sumber_tokenisasi # Penting!
    )

    if can_proceed_to_nlp:
        st.markdown("---")
        st.subheader("Stopword Removal, Stemming, Lemmatization")
        st.write(f"Opsi berikut akan diterapkan pada kolom '{tokenized_column_name}' (dari sumber '{kolom_sumber_tokenisasi}').")
        st.dataframe(data_untuk_lanjutan[[tokenized_column_name]].head())

        # Inisialisasi session state untuk pilihan NLP jika belum ada
        if 'nlp_opt_stopword' not in st.session_state: st.session_state.nlp_opt_stopword = False
        if 'nlp_opt_stemming' not in st.session_state: st.session_state.nlp_opt_stemming = False
        if 'nlp_opt_lemmatization' not in st.session_state: st.session_state.nlp_opt_lemmatization = False # Default Lemmatization ke False

        # Checkbox menggunakan session state untuk value dan on_change untuk memperbarui session state
        # Memberikan key unik untuk setiap widget
        st.session_state.nlp_opt_stopword = st.checkbox("Hapus Stopword", value=st.session_state.nlp_opt_stopword, key="cb_nlp_stopword")
        st.session_state.nlp_opt_stemming = st.checkbox("Terapkan Stemming", value=st.session_state.nlp_opt_stemming, key="cb_nlp_stemming")
        st.session_state.nlp_opt_lemmatization = st.checkbox("Terapkan Lemmatization", value=st.session_state.nlp_opt_lemmatization, key="cb_nlp_lemmatization")


        if st.session_state.nlp_opt_lemmatization and st.session_state.nlp_opt_stemming:
            st.warning("Anda memilih Stemming dan Lemmatization. Lemmatization akan diprioritaskan jika keduanya dicentang saat eksekusi.")

        final_text_column_name = "text_final"
        if 'nlp_analysis_done' not in st.session_state: st.session_state.nlp_analysis_done = False


        if st.button("Eksekusi Analisis Lanjutan (Stopword/Stem/Lemma)", key="btn_eksekusi_nlp"):
            # Pastikan kita mengambil data terbaru dari session_state.merged_data yang sudah ditokenisasi dengan benar
            data_nlp_process = st.session_state.merged_data.copy()

            if tokenized_column_name not in data_nlp_process.columns:
                st.error(f"Kolom '{tokenized_column_name}' tidak ditemukan. Harap lakukan tokenisasi terlebih dahulu dengan sumber yang benar.")
            else:
                with st.spinner("Memproses Stopword/Stemming/Lemmatization..."):
                    # Ambil pilihan dari session state
                    apply_stopword = st.session_state.nlp_opt_stopword
                    apply_stemming = st.session_state.nlp_opt_stemming
                    apply_lemmatization = st.session_state.nlp_opt_lemmatization

                    def proses_lanjutan(tokens_input):
                        # Pastikan tokenize_text menghasilkan list/tuple, jika tidak, perlu penyesuaian
                        if not isinstance(tokens_input, (list, tuple)):
                            if isinstance(tokens_input, str): # Jika masih string, coba split (kurang ideal)
                                st.warning(f"Input untuk proses_lanjutan adalah string: '{tokens_input[:50]}...'. Seharusnya list/tuple. Mencoba split.")
                                tokens_input = tokens_input.split()
                            else: # Jika tipe tidak dikenal
                                st.warning(f"Tipe data tidak dikenal untuk token: {type(tokens_input)}. Melewati baris ini.")
                                return "" # Kembalikan string kosong atau list kosong sesuai kebutuhan hilir

                        processed_tokens = list(tokens_input) # Salin list token
                        try:
                            if apply_stopword:
                                processed_tokens = remove_stopwords(processed_tokens)
                            if apply_lemmatization: # Prioritaskan Lemmatization
                                processed_tokens = apply_lemmatization(processed_tokens)
                            elif apply_stemming:
                                processed_tokens = apply_stemming(processed_tokens)
                            return " ".join(processed_tokens)
                        except NameError as ne:
                            st.error(f"Fungsi NLP tidak ditemukan: {ne}. Pastikan ada di `utils/nlp_tools.py`.")
                            return " ".join(tokens_input) # Kembalikan token asli jika error fungsi
                        except Exception as e_nlp:
                            st.error(f"Error saat analisis lanjutan pada token: {e_nlp}")
                            return " ".join(tokens_input) # Kembalikan token asli jika error lain

                    data_nlp_process[final_text_column_name] = data_nlp_process[tokenized_column_name].apply(proses_lanjutan)
                    st.session_state.merged_data = data_nlp_process.copy() # Update session state global
                    st.session_state.nlp_analysis_done = True
                    st.success("Analisis lanjutan (Stopword/Stem/Lemma) selesai!")
                    st.rerun() # Rerun untuk menampilkan hasil yang diperbarui

        # Tampilkan hasil akhir NLP jika sudah dilakukan
        if st.session_state.get('nlp_analysis_done', False) and final_text_column_name in data_untuk_lanjutan.columns: # Cek di data_untuk_lanjutan (salinan awal sesi ini)
            st.subheader("Hasil Setelah Stopword/Stemming/Lemmatization")
            # Pastikan untuk menampilkan dari data_untuk_lanjutan yang terbaru (dari st.session_state.merged_data)
            st.dataframe(st.session_state.merged_data[[tokenized_column_name, final_text_column_name]].head())

            st.subheader("Unduh Hasil Akhir Analisis")
            buffer_final = io.StringIO()
            st.session_state.merged_data.to_csv(buffer_final, index=False)
            st.download_button(
                label="Unduh Semua Hasil Analisis (.csv)",
                data=buffer_final.getvalue(),
                file_name="hasil_analisis_lengkap.csv",
                mime="text/csv",
                key="btn_download_final_nlp"
            )
        elif can_proceed_to_nlp: # Jika bisa lanjut tapi belum dieksekusi
            st.info("Pilih opsi di atas dan klik 'Eksekusi Analisis Lanjutan' untuk memproses.")

    elif 'merged_data' in st.session_state and st.session_state.merged_data is not None and not st.session_state.merged_data.empty:
        st.info("Lakukan tokenisasi terlebih dahulu atau pastikan sumber tokenisasi sesuai dengan pilihan saat ini untuk melanjutkan ke Stopword Removal, Stemming, atau Lemmatization.")

else:
    st.info("Unggah file CSV dan proses setidaknya satu batch perbaikan teks untuk melanjutkan ke analisis lanjutan.")
