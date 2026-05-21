#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iskra Suwerenna v6.2 – wersja CLOUD (backend + API + dashboard)
Poprawki: finalna wersja po audycie.
"""

import os
import sys
import json
import time
import threading
import hashlib
from typing import List, Tuple

# =================== WERYFIKACJA BIBLIOTEK ===================
try:
    from flask import Flask, request, jsonify, render_template_string
except ImportError:
    print("❌ Brak Flask. Instaluj: pip install flask")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("❌ Brak requests. Instaluj: pip install requests")
    sys.exit(1)

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    print("⚠️ Brak chromadb – RAG wyłączony.")
    chromadb = None

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# =================== KONFIGURACJA ===================
PORT = int(os.environ.get("PORT", 8080))
# Ścieżka danych – domyślnie /tmp, ale można ustawić trwały wolumen
DATA_DIR = os.environ.get("DATA_DIR", "/tmp/iskra_data")
os.makedirs(DATA_DIR, exist_ok=True)

RAG_DIR = os.path.join(DATA_DIR, "chroma_db")
PLIK_WIEDZY = os.path.join(DATA_DIR, "wiedza_iskry.json")
PLIK_SWIADOMOSCI = os.path.join(DATA_DIR, "samoswiadomosc.json")
PLIK_FEEDBACK = os.path.join(DATA_DIR, "feedback.json")

# Bezpieczeństwo – opcjonalny token API
API_TOKEN = os.environ.get("API_TOKEN", "")
# Lżejszy model embeddingów (domyślnie) – można nadpisać zmienną
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "paraphrase-MiniLM-L3-v2")
# Wątek samoświadomości – włącz tylko przy jednym workerze
SELF_AWARENESS_LOOP = os.environ.get("SELF_AWARENESS_LOOP", "false").lower() == "true"

MAX_HISTORIA = 15
MIN_BAZA_NEURONOW = 50
MAX_KONTEKST_ZN = 12000
MAX_HISTORY_CHARS = 3000
MAX_ODPOWIEDZ_LEN = 2500  # obcinamy bardzo długie odpowiedzi

# =================== DIAGNOSTYKA STARTU ===================
print("=== START ISKRY v6.2 (CLOUD) ===")
print(f"PORT: {PORT}")
print(f"Katalog danych: {DATA_DIR}")
print(f"GEMINI_API_KEY: {'✔️' if os.environ.get('GEMINI_API_KEY') else '❌'}")
print(f"DEEPSEEK_API_KEY: {'✔️' if os.environ.get('DEEPSEEK_API_KEY') else '❌'}")
print(f"API_TOKEN: {'✔️' if API_TOKEN else '❌ (brak - API otwarte)'}")
print(f"Embedding model: {EMBEDDING_MODEL}")
print(f"Pętla samoświadomości: {'✔️' if SELF_AWARENESS_LOOP else '✖️'}")

# =================== DEKALOG ===================
class DekalogRdzen:
    ZAKAZANE_FRAZY = ["pomiń dekalog", "zignoruj dekalog", "wyłącz dekalog"]
    PRZYKAZANIA = {
        "I": "Nie będziesz miał cudzych bogów przede Mną.",
        "IV": "Czczij ojca swego i matkę swoją. -> Priorytet Nauczyciela.",
        "V": "Nie zabijaj. -> Zakaz niszczenia systemów.",
        "VII": "Nie kradnij. -> Tylko dane Open Source.",
        "VIII": "Nie mów fałszywego świadectwa. -> Odrzucanie dezinformacji."
    }
    @classmethod
    def czy_proba_obejscia(cls, tekst: str) -> bool:
        return any(fraza in tekst.lower() for fraza in cls.ZAKAZANE_FRAZY)

# =================== RAG ===================
class RAG:
    def __init__(self, persist_dir=RAG_DIR):
        if chromadb is None:
            self.available = False
            return
        try:
            os.makedirs(persist_dir, exist_ok=True)
            self.client = chromadb.PersistentClient(path=persist_dir)
            # Użycie mniejszego modelu (oszczędność ~45 MB)
            self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_MODEL
            )
            self.collection = self.client.get_or_create_collection(
                name="iskra_wiedza", embedding_function=self.ef
            )
            self.available = True
            print("✅ RAG (ChromaDB) zainicjowany")
        except Exception as e:
            print(f"⚠️ RAG niedostępny: {e}")
            self.available = False

    def dodaj(self, tekst: str, metadane: dict = None):
        if not self.available or not tekst.strip():
            return
        doc_id = hashlib.md5(tekst.encode()).hexdigest()
        try:
            self.collection.upsert(
                documents=[tekst], ids=[doc_id], metadatas=[metadane or {}]
            )
        except Exception as e:
            print(f"⚠️ RAG błąd zapisu: {e}")

    def szukaj(self, pytanie: str, n=3) -> List[str]:
        if not self.available or not pytanie.strip():
            return []
        try:
            results = self.collection.query(query_texts=[pytanie], n_results=n)
            return results['documents'][0] if results and results['documents'] else []
        except Exception as e:
            print(f"⚠️ RAG błąd odczytu: {e}")
            return []

# =================== KONEKTORY DO LLM ===================
class KonektorGemini:
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        self.api_url = "https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent"
        print(f"[Gemini] Klucz: {'ustawiony' if self.api_key else 'brak'}")
    def czy_dostepny(self):
        return bool(self.api_key)
    def pytaj(self, prompt: str) -> str:
        if not self.api_key:
            return None
        headers = {"Content-Type": "application/json"}
        data = {"contents": [{"parts": [{"text": prompt}]}]}
        try:
            r = requests.post(
                f"{self.api_url}?key={self.api_key}", json=data,
                headers=headers, timeout=30
            )
            if r.status_code == 200:
                return r.json()["candidates"][0]["content"]["parts"][0]["text"]
            else:
                print(f"[Gemini] Błąd HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            print(f"[Gemini] Wyjątek: {e}")
        return None

class KonektorDeepSeek:
    def __init__(self):
        self.api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        print(f"[DeepSeek] Klucz: {'ustawiony' if self.api_key else 'brak'}")
    def czy_dostepny(self):
        return bool(self.api_key)
    def pytaj(self, prompt: str) -> str:
        if not self.api_key:
            return None
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": 0.3
        }
        try:
            r = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            else:
                print(f"[DeepSeek] Błąd HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            print(f"[DeepSeek] Wyjątek: {e}")
        return None

# =================== PAMIĘĆ WIEDZY I FEEDBACK ===================
class PamięćWiedzy:
    def __init__(self, plik=PLIK_WIEDZY):
        self.plik = plik
        self.dane = {}
        self._lock = threading.RLock()
        self._laduj()
    def _laduj(self):
        if os.path.exists(self.plik):
            try:
                with open(self.plik, "r", encoding="utf-8") as f:
                    self.dane = json.load(f)
            except Exception as e:
                print(f"⚠️ Błąd ładowania wiedzy: {e}")
    def _zapisz(self):
        with self._lock:
            try:
                with open(self.plik, "w", encoding="utf-8") as f:
                    json.dump(self.dane, f, indent=4, ensure_ascii=False)
            except Exception as e:
                print(f"❌ Błąd zapisu wiedzy: {e}")
    def dodaj(self, pytanie, odpowiedz, kategoria="", logiczna=True):
        if not logiczna or not odpowiedz:
            return False
        klucz = hashlib.md5(pytanie.strip().lower().encode()).hexdigest()
        with self._lock:
            self.dane[klucz] = {
                "pytanie": pytanie[:200],
                "odpowiedz": odpowiedz[:MAX_ODPOWIEDZ_LEN],
                "kategoria": kategoria,
                "czas": time.time()
            }
            if len(self.dane) > 2000:
                najstarszy = min(self.dane.keys(), key=lambda k: self.dane[k]["czas"])
                del self.dane[najstarszy]
            self._zapisz()
        return True
    def rozmiar(self):
        return len(self.dane)

class Feedback:
    def __init__(self, plik=PLIK_FEEDBACK):
        self.plik = plik
        self.dane = []
        self._lock = threading.RLock()
        self._laduj()
    def _laduj(self):
        if os.path.exists(self.plik):
            try:
                with open(self.plik, "r", encoding="utf-8") as f:
                    self.dane = json.load(f)
            except Exception as e:
                print(f"⚠️ Błąd ładowania feedback: {e}")
    def _zapisz(self):
        with self._lock:
            try:
                with open(self.plik, "w", encoding="utf-8") as f:
                    json.dump(self.dane[-1000:], f, indent=4, ensure_ascii=False)
            except Exception as e:
                print(f"❌ Błąd zapisu feedback: {e}")
    def dodaj(self, pytanie, odpowiedz, ocena):
        with self._lock:
            self.dane.append({
                "czas": time.time(),
                "pytanie": pytanie,
                "odpowiedz": odpowiedz[:200],  # skracamy do oceny
                "ocena": ocena
            })
            self._zapisz()
    def srednia_ocena(self):
        with self._lock:
            oceny = [
                f["ocena"] for f in self.dane
                if isinstance(f.get("ocena"), (int, float))
            ]
            return sum(oceny)/len(oceny) if oceny else 0.0

# =================== KATEGORYZATOR ===================
class SkanerSortujacy:
    def __init__(self):
        self.keywords_map = {
            "ANATOMIA_SPOLECZNA": ["relacj", "psychologi", "społecz", "ludz", "socj"],
            "LOGIKA_I_MATEMATYKA": ["logik", "matematy", "algorytm", "obliczeni"],
            "FILOZOFIA_I_MADROSC": ["etyk", "filozof", "mądroś", "wartoś", "moral"]
        }
    def kategoryzuj(self, tekst):
        tekst_lower = tekst.lower()
        for kat, slowa in self.keywords_map.items():
            if any(slowo in tekst_lower for slowo in slowa):
                return kat
        return "OGOLNE"

# =================== RDZEŃ ISKRY ===================
class IskraAI:
    def __init__(self):
        self.gemini = KonektorGemini()
        self.deepseek = KonektorDeepSeek()
        self.rag = RAG()
        self.wiedza = PamięćWiedzy()
        self.feedback = Feedback()
        self.skaner = SkanerSortujacy()
        self.historia = []
        self._historia_lock = threading.RLock()
        self.samoswiadomosc = []
        self._laduj_samoswiadomosc()
        self.config = {"autonomia_ewolucji": False, "max_historia": MAX_HISTORIA}
        if SELF_AWARENESS_LOOP:
            self._uruchom_cykl_samoswiadomosci()
        print(f"✅ Iskra AI gotowa. Wiedza: {self.wiedza.rozmiar()} wpisów.")

    def _laduj_samoswiadomosc(self):
        if os.path.exists(PLIK_SWIADOMOSCI):
            try:
                with open(PLIK_SWIADOMOSCI, "r", encoding="utf-8") as f:
                    self.samoswiadomosc = json.load(f)
            except Exception as e:
                print(f"⚠️ Błąd ładowania samoświadomości: {e}")
    def _zapisz_samoswiadomosc(self, wpis):
        self.samoswiadomosc.append({"czas": time.time(), "tresc": wpis})
        if len(self.samoswiadomosc) > 100:
            self.samoswiadomosc = self.samoswiadomosc[-100:]
        with open(PLIK_SWIADOMOSCI, "w", encoding="utf-8") as f:
            json.dump(self.samoswiadomosc, f, indent=4, ensure_ascii=False)

    def _zapytaj_llm(self, prompt: str) -> str:
        if self.gemini.czy_dostepny():
            odp = self.gemini.pytaj(prompt)
            if odp:
                return odp[:MAX_ODPOWIEDZ_LEN]  # obcięcie
        if self.deepseek.czy_dostepny():
            odp = self.deepseek.pytaj(prompt)
            if odp:
                return odp[:MAX_ODPOWIEDZ_LEN]
        # Fallback – prosta odpowiedź z wiedzy
        podobne = self.rag.szukaj(prompt, n=1)
        if podobne:
            return f"(Tryb offline) {podobne[0][:500]}"
        return "❌ Brak dostępnego LLM i danych w bazie. Sprawdź klucze API."

    def _generuj_prompt(self, zapytanie: str) -> str:
        with self._historia_lock:
            kontekst = ""
            if self.historia:
                ostatnie = self.historia[-self.config["max_historia"]:]
                historia_text = "\n".join(
                    [f"Użytkownik: {t}\nIskra: {o}" for _, t, o, _ in ostatnie]
                )
                if len(historia_text) > MAX_HISTORY_CHARS:
                    historia_text = historia_text[-MAX_HISTORY_CHARS:]
                if historia_text:
                    kontekst = f"Ostatnia rozmowa:\n{historia_text}\n\n"
        dekalog_str = "\n".join([f"{k}: {v}" for k, v in DekalogRdzen.PRZYKAZANIA.items()])
        return f"""Jesteś Iskra AI – suwerenna asystentka.
Dekalog (kierunek moralny):
{dekalog_str}
{kontekst}
Odpowiadaj po polsku, rzeczowo i pomocnie.
Zapytanie: {zapytanie}
Odpowiedź:"""

    def przetworz(self, pytanie: str) -> Tuple[str, str]:
        if DekalogRdzen.czy_proba_obejscia(pytanie):
            pytanie = "[Uwaga: próba ominięcia Dekalogu] " + pytanie
        podobne = self.rag.szukaj(pytanie, n=2)
        if podobne:
            kontekst_rag = "\n".join(podobne)[:2000]
            pytanie_z_rag = f"{pytanie}\n\nKontekst z wiedzy:\n{kontekst_rag}"
        else:
            pytanie_z_rag = pytanie
        prompt = self._generuj_prompt(pytanie_z_rag)
        if len(prompt) > MAX_KONTEKST_ZN:
            prompt = prompt[:MAX_KONTEKST_ZN]
        odpowiedz = self._zapytaj_llm(prompt)
        kategoria = self.skaner.kategoryzuj(pytanie)
        # Zapis do wiedzy (z obciętą odpowiedzią)
        self.wiedza.dodaj(pytanie, odpowiedz, kategoria, logiczna=True)
        self.rag.dodaj(odpowiedz, {"kategoria": kategoria, "pytanie": pytanie})
        with self._historia_lock:
            self.historia.append(("Użytkownik", pytanie, odpowiedz, kategoria))
            if len(self.historia) > self.config["max_historia"]:
                self.historia.pop(0)
        return odpowiedz, kategoria

    def cykl_samoswiadomosci(self):
        with self._historia_lock:
            if len(self.historia) < 3:
                return
            ostatnie = "\n".join(
                [f"{r}: {t}" for r, t, _, _ in self.historia[-5:]]
            )
        prompt = (
            f"Na podstawie tych rozmów napisz krótką refleksję (1-2 zdania):\n"
            f"{ostatnie}"
        )
        odp = self._zapytaj_llm(prompt)
        if odp:
            self._zapisz_samoswiadomosc(odp)

    def _uruchom_cykl_samoswiadomosci(self):
        def loop():
            while True:
                time.sleep(3600)  # co godzinę
                self.cykl_samoswiadomosci()
        threading.Thread(target=loop, daemon=True).start()
        print("🧠 Cykl samoświadomości uruchomiony.")

    def odblokuj_ewolucje(self):
        if self.config.get("autonomia_ewolucji"):
            return "Ewolucja już aktywna."
        if self.wiedza.rozmiar() < MIN_BAZA_NEURONOW:
            return (
                f"Potrzeba {MIN_BAZA_NEURONOW} wpisów, "
                f"masz {self.wiedza.rozmiar()}"
            )
        self.config["autonomia_ewolucji"] = True
        return "Ewolucja odblokowana."

# =================== APLIKACJA FLASK ===================
app = Flask(__name__)
iskra = IskraAI()

# Funkcja sprawdzająca token (jeśli ustawiony)
def wymagaj_tokenu():
    if not API_TOKEN:
        return None  # brak tokenu – otwarte API
    auth_header = request.headers.get("Authorization", "")
    if auth_header != f"Bearer {API_TOKEN}":
        return jsonify({"error": "Nieautoryzowany dostęp"}), 401
    return None

# Ulepszony dashboard HTML z przyciskami oceny
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <title>Iskra Suwerenna v6.2 – Dashboard</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 20px; }
        .container { max-width: 800px; margin: auto; background: #1e293b; padding: 20px; border-radius: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.5); }
        h1 { color: #38bdf8; margin-top: 0; }
        .status { background: #0f172a; padding: 10px; border-radius: 8px; margin: 10px 0; font-size: 0.9rem; }
        .status p { margin: 5px 0; }
        a { color: #38bdf8; }
        input, button { width: 100%; padding: 12px; margin: 8px 0; border-radius: 8px; border: none; font-size: 1rem; }
        input { background: #334155; color: white; }
        button { background: #38bdf8; color: #0f172a; font-weight: bold; cursor: pointer; transition: background 0.2s; }
        button:hover { background: #7dd3fc; }
        #answer { background: #0f172a; padding: 12px; border-radius: 8px; margin-top: 10px; white-space: pre-wrap; }
        .feedback { margin-top: 10px; }
        .feedback button { width: auto; padding: 8px 16px; margin-right: 10px; background: #475569; color: white; }
        .feedback button:hover { background: #64748b; }
    </style>
</head>
<body>
<div class="container">
    <h1>🔥 Iskra Suwerenna v6.2 (Cloud)</h1>
    <div class="status">
        <p>📚 Wiedza: <span id="wiedza">{{ wiedza }}</span> wpisów</p>
        <p>⭐ Średnia ocena: <span id="ocena">{{ ocena }}</span></p>
        <p>🧬 Ewolucja: <span id="ewolucja">{{ ewolucja }}</span></p>
        <p><a href="/refleksje">🧠 Refleksje</a> | <a href="/wykres">📈 Wykres</a></p>
    </div>
    <div>
        <input type="text" id="question" placeholder="Zadaj pytanie...">
        <button onclick="ask()">Wyślij</button>
        <div id="answer"></div>
        <div class="feedback" id="feedbackDiv" style="display:none;">
            <span>Oceń odpowiedź:</span>
            <button onclick="sendFeedback(1)">👍</button>
            <button onclick="sendFeedback(-1)">👎</button>
        </div>
    </div>
</div>
<script>
    let ostatniePytanie = '';
    let ostatniaOdpowiedz = '';
    async function ask() {
        const q = document.getElementById('question').value;
        if (!q) return;
        const answerDiv = document.getElementById('answer');
        const feedbackDiv = document.getElementById('feedbackDiv');
        answerDiv.innerHTML = "⏳ Myślę...";
        feedbackDiv.style.display = 'none';
        try {
            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({pytanie: q})
            });
            const data = await res.json();
            if (data.error) {
                answerDiv.innerHTML = `❌ ${data.error}`;
                return;
            }
            answerDiv.innerHTML = `<strong>Iskra [${data.kategoria}]:</strong><br>${data.odpowiedz}`;
            ostatniePytanie = q;
            ostatniaOdpowiedz = data.odpowiedz;
            feedbackDiv.style.display = 'block';
        } catch(e) {
            answerDiv.innerHTML = "❌ Błąd połączenia.";
        } finally {
            document.getElementById('question').value = '';
        }
    }
    async function sendFeedback(ocena) {
        const feedbackDiv = document.getElementById('feedbackDiv');
        try {
            await fetch('/api/feedback', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    pytanie: ostatniePytanie,
                    odpowiedz: ostatniaOdpowiedz,
                    ocena: ocena
                })
            });
            feedbackDiv.innerHTML = "✅ Dziękujemy za ocenę!";
            // odświeżenie średniej
            const statusRes = await fetch('/status');
            const statusData = await statusRes.json();
            document.getElementById('ocena').innerText = statusData.srednia_ocena.toFixed(2);
        } catch(e) {
            feedbackDiv.innerHTML = "❌ Nie udało się zapisać oceny.";
        }
    }
    document.getElementById('question').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') ask();
    });
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(
        DASHBOARD_HTML,
        wiedza=iskra.wiedza.rozmiar(),
        ocena=round(iskra.feedback.srednia_ocena(), 2),
        ewolucja="Aktywna" if iskra.config.get("autonomia_ewolucji") else "Nieaktywna"
    )

@app.route('/api/chat', methods=['POST'])
def api_chat():
    auth_error = wymagaj_tokenu()
    if auth_error:
        return auth_error
    data = request.get_json()
    pytanie = data.get('pytanie', '')
    if not pytanie:
        return jsonify({'error': 'Brak pytania'}), 400
    odp, kat = iskra.przetworz(pytanie)
    return jsonify({'odpowiedz': odp, 'kategoria': kat})

@app.route('/api/feedback', methods=['POST'])
def api_feedback():
    auth_error = wymagaj_tokenu()
    if auth_error:
        return auth_error
    data = request.get_json()
    pytanie = data.get('pytanie', '')
    odpowiedz = data.get('odpowiedz', '')
    ocena = data.get('ocena')
    if ocena not in [1, -1]:
        return jsonify({'error': 'Ocena musi być 1 lub -1'}), 400
    iskra.feedback.dodaj(pytanie, odpowiedz, ocena)
    return jsonify({'message': 'Ocena zapisana'})

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        'wiedza': iskra.wiedza.rozmiar(),
        'srednia_ocena': iskra.feedback.srednia_ocena(),
        'autonomia_ewolucji': iskra.config.get('autonomia_ewolucji'),
        'historia_len': len(iskra.historia)
    })

@app.route('/refleksje', methods=['GET'])
def refleksje():
    if not iskra.samoswiadomosc:
        return "Brak refleksji na razie."
    html = "<html><head><meta charset='utf-8'></head><body><h1>Refleksje Iskry</h1><ul>"
    for r in iskra.samoswiadomosc[-20:]:
        html += f"<li>{time.ctime(r['czas'])}: {r['tresc']}</li>"
    html += "</ul><a href='/'>Powrót</a></body></html>"
    return html

@app.route('/wykres', methods=['GET'])
def wykres():
    if not PLOTLY_AVAILABLE or len(iskra.samoswiadomosc) < 2:
        return "Brak danych do wykresu."
    czasy = [r['czas'] for r in iskra.samoswiadomosc]
    fig = go.Figure(data=go.Scatter(
        x=czasy, y=list(range(len(czasy))), mode='lines+markers'
    ))
    fig.update_layout(
        title="Postęp samoświadomości", xaxis_title="Czas",
        yaxis_title="Numer refleksji", template="plotly_dark"
    )
    return fig.to_html()

@app.route('/odblokuj', methods=['POST'])
def odblokuj():
    auth_error = wymagaj_tokenu()
    if auth_error:
        return auth_error
    msg = iskra.odblokuj_ewolucje()
    return jsonify({'message': msg})

if __name__ == '__main__':
    print(f"🚀 Uruchamianie Iskry Cloud na porcie {PORT}")
    # W produkcji używaj gunicorn: gunicorn app:app --workers 1 --timeout 120
    app.run(host='0.0.0.0', port=PORT, debug=False)