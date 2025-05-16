from googletrans import Translator
import string 

def translate_with_googletrans(tweet):
    # Validasi input: pastikan itu string dan tidak hanya spasi kosong
    if not isinstance(tweet, str) or not tweet.strip():
        print("Input tidak valid: Harap berikan string yang tidak kosong.")
        return None

    translator = Translator()
    try:
        # Lakukan translasi dari Indonesia (id) ke Inggris (en)
        translation_result = translator.translate(tweet, src='id', dest='en')
        if not translation_result or not translation_result.text:
             # Kasus di mana translasi mungkin mengembalikan hasil kosong/None
             print(f"Hasil translasi kosong untuk: '{str(tweet)[:50]}...'")
             return None
        translated_text = translation_result.text

        # 1. Ubah hasil terjemahan menjadi huruf kecil semua
        lower_text = translated_text.lower()
        translator_table = str.maketrans('', '', string.punctuation)
        cleaned_text = lower_text.translate(translator_table)

        return cleaned_text

    except Exception as e:
        print(f"Error saat menerjemahkan: '{str(tweet)[:50]}...' - Error: {e}") # Atau tetap print
        return None