import requests

def call_openrouter_api(api_key, prompt, model_id="google/gemini-flash-1.5"):
    api_url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "Referer": "kamus-bahasa-app",
        "X-Title": "Kamus Bahasa Devi",
    }
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 200,
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        response_json = response.json()

        if response_json.get("choices"):
            return response_json["choices"][0]["message"]["content"].strip()
        elif response_json.get("error"):
            return f"API Error: {response_json['error']['message']}"
        else:
            return "Struktur respons tidak sesuai."
    except Exception as e:
        return f"Error saat memanggil API: {e}"

