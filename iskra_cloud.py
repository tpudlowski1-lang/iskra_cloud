#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iskra Suwerenna v6.1 – wersja CLOUD (backend + API + dashboard)
Przystosowana do uruchomienia na Render / Koyeb / Railway.
Nie wymaga GUI, mikrofonu ani TTS.
Komunikacja przez REST API i prosty panel webowy.
"""

import os
import sys
import json
import time
import threading
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Any

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
    print("⚠️ Brak chromadb – RAG wyłączony. Instaluj: pip install chromadb")
    chromadb = None

try:
    import plotly
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# =================== KONFIGURACJA ===================
PORT = int(os.environ.get("PORT", 8080))
RAG_DIR = "/tmp/chroma_db" if not os.environ.get("PERSIST_DIR") else os.environ.get("PERSIST_DIR")
PLIK_WIEDZY = "wiedza_iskry.json"
PLIK_SWIADOMOSCI = "samoswiadomosc.json"
PLIK_FEEDBACK = "feedback.json"
MAX_HISTORIA = 15
MIN_BAZA_NEURONOW = 50
MAX_KONTEKST_ZN = 12000
MAX_HISTORY_CHARS = 3000

# =================== DEKALOG (identyczny jak w oryginale) ===================
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

# =================== RAG (CHROMADB) ===================
class RAG:
    def __init__(self, persist_dir=RAG_DIR):
        if chromadb is None:
            self.available = False
            return
        try:
            os.makedirs(persist_dir, exist_ok=True)
            self.client = chromadb.PersistentClient(path=persist_dir)
            self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
            self.collection = self.client.get_or_create_collection(name="iskra_wiedza", embedding_function=self.ef)
            self.available = True
        except Exception as e:
            print(f"⚠️ RAG niedostępny: {e}")
            self.available = False

    def dodaj(self, tekst: str, metadane: dict = None):
        if not self.available:
            return
        doc_id = hashlib.md5(tekst.encode()).hexdigest()
        self.collection.upsert(documents=[tekst], ids=[doc_id], metadatas=[metadane or {}])

    def szukaj(self, pytanie: str, n=3) -> List[str]:
        if not self.available:
            return []
        results = self.collection.query(query_texts=[pytanie], n_results=n)
        return results['documents'][0] if results and results['documents'] else []

# =================== KONEKTORY DO LLM (Gemini / DeepSeek) ===================
class KonektorGemini:
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        self.api_url = "https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent"
    def czy_dostepny(self):
        return bool(self.api_key)
    def pytaj(self, prompt: str) -> str:
        if not self.api_key:
            return None
        headers = {"Content-Type": "application/json"}
        data = {"contents": [{"parts": [{"text": prompt}]}]}
        try:
            r = requests.post(f"{self.api_url}?key={self.api_key}", json=data, headers=headers, timeout=30)
            if r.status_code == 200:
                return r.json()["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            print(f"[Gemini] Błąd: {e}")
        return None

class KonektorDeepSeek:
    def __init__(self):
        self.api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
    def czy_dostepny(self):
        return bool(self.api_key)
    def pytaj(self, prompt: str, system_prompt=None) -> str:
        if not self.api_key:
            return None
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {"model": "deepseek-chat", "messages": messages, "stream": False, "temperature": 0.3}
        try:
            r = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[DeepSeek] Błąd: {e}")
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
            except: pass
    def _zapisz(self):
        with self._lock:
            with open(self.plik, "w", encoding="utf-8") as f:
                json.dump(self.dane, f, indent=4, ensure_ascii=False)
    def dodaj(self, pytanie, odpowiedz, kategoria="", logiczna=True):
        if not logiczna:
            return False
        klucz = hashlib.md5(pytanie.strip().lower().encode()).hexdigest()
        with self._lock:
            self.dane[klucz] = {"pytanie": pytanie[:200], "odpowiedz": odpowiedz, "kategoria": kategoria, "czas": time.time()}
            if len(self.dane) > 2000:
                najstarszy = min(self.dane.keys(), key=lambda k: self.dane[k]["czas"])
                del self.dane[najstarszy]
            self._zapisz()
        return True
    def znajdz(self, pytanie):
        klucz = hashlib.md5(pytanie.strip().lower().encode()).hexdigest()
        return self.dane.get(klucz, {}).get("odpowiedz")
    def rozmiar(self):
        return len(self.dane)
    def wszystkie(self):
        return list(self.dane.values())

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
            except: pass
    def _zapisz(self):
        with self._lock:
            with open(self.plik, "w", encoding="utf-8") as f:
                json.dump(self.dane[-1000:], f, indent=4, ensure_ascii=False)
    def dodaj(self, pytanie, odpowiedz, ocena, korekta=None):
        with self._lock:
            self.dane.append({"czas": time.time(), "pytanie": pytanie, "odpowiedz": odpowiedz, "ocena": ocena, "korekta": korekta})
            self._zapisz()
    def srednia_ocena(self):
        with self._lock:
            if not self.dane:
                return 0
            oceny = [f["ocena"] for f in self.dane if isinstance(f.get("ocena"), (int, float))]
            return sum(oceny)/len(oceny) if oceny else 0

# =================== KATEGORYZATOR (prosty, bez LLM) ===================
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

# =================== RDZEŃ ISKRY (bez GUI) ===================
class IskraAI:
    def __init__(self):
        self.gemini = KonektorGemini()
        self.deepseek = KonektorDeepSeek()
        self.rag = RAG()
        self.wiedza = PamięćWiedzy()
        self.feedback = Feedback()
        self.skaner = SkanerSortujacy()
        self.historia = []  # (rola, tekst, odpowiedz, kategoria)
        self._historia_lock = threading.RLock()
        self.samoswiadomosc = []
        self._laduj_samoswiadomosc()
        self.config = {"autonomia_ewolucji": False, "max_historia": MAX_HISTORIA}
        self._uruchom_cykl_samoswiadomosci()
        print(f"Iskra AI gotowa. Wiedza: {self.wiedza.rozmiar()} wpisów.")

    def _laduj_samoswiadomosc(self):
        if os.path.exists(PLIK_SWIADOMOSCI):
            try:
                with open(PLIK_SWIADOMOSCI, "r", encoding="utf-8") as f:
                    self.samoswiadomosc = json.load(f)
            except: pass
        else:
            self.samoswiadomosc = []

    def _zapisz_samoswiadomosc(self, wpis):
        self.samoswiadomosc.append({"czas": time.time(), "tresc": wpis})
        if len(self.samoswiadomosc) > 100:
            self.samoswiadomosc = self.samoswiadomosc[-100:]
        with open(PLIK_SWIADOMOSCI, "w", encoding="utf-8") as f:
            json.dump(self.samoswiadomosc, f, indent=4, ensure_ascii=False)

    def _zapytaj_llm(self, prompt: str) -> str:
        """Wywołuje Gemini lub DeepSeek (pierwszy dostępny)."""
        if self.gemini.czy_dostepny():
            odp = self.gemini.pytaj(prompt)
            if odp:
                return odp
        if self.deepseek.czy_dostepny():
            odp = self.deepseek.pytaj(prompt)
            if odp:
                return odp
        return "❌ Brak dostępnego klucza API (Gemini/DeepSeek). Dodaj zmienną środowiskową GEMINI_API_KEY lub DEEPSEEK_API_KEY."

    def _generuj_prompt(self, zapytanie: str) -> str:
        with self._historia_lock:
            kontekst = ""
            if self.historia:
                ostatnie = self.historia[-self.config["max_historia"]:]
                historia_text = "\n".join([f"Użytkownik: {t}\nIskra: {o}" for _, t, o, _ in ostatnie])
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

        # RAG
        podobne = self.rag.szukaj(pytanie, n=2)
        if podobne:
            kontekst_rag = "\n".join(podobne)[:2000]
            pytanie_z_rag = f"{pytanie}\n\nKontekst z wiedzy:\n{kontekst_rag}"
        else:
            pytanie_z_rag = pytanie

        prompt = self._generuj_prompt(pytanie_z_rag)
        # Ograniczenie długości promptu
        if len(prompt) > MAX_KONTEKST_ZN:
            prompt = prompt[:MAX_KONTEKST_ZN]
        odpowiedz = self._zapytaj_llm(prompt)
        if not odpowiedz:
            odpowiedz = "Przepraszam, nie mogę teraz wygenerować odpowiedzi. Sprawdź klucze API."

        kategoria = self.skaner.kategoryzuj(pytanie)
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
                return "Za mało rozmów do refleksji."
            ostatnie = "\n".join([f"{r}: {t}" for r, t, _, _ in self.historia[-5:]])
        prompt = f"Na podstawie tych rozmów napisz krótką refleksję (1-2 zdania) czego się nauczyłaś:\n{ostatnie}"
        odp = self._zapytaj_llm(prompt)
        if odp:
            self._zapisz_samoswiadomosc(odp)
        return odp or "Refleksja niedostępna."

    def _uruchom_cykl_samoswiadomosci(self):
        def loop():
            while True:
                time.sleep(3600)
                self.cykl_samoswiadomosci()
        threading.Thread(target=loop, daemon=True).start()

    def odblokuj_ewolucje(self):
        if self.config.get("autonomia_ewolucji"):
            return "Ewolucja już aktywna."
        if self.wiedza.rozmiar() < MIN_BAZA_NEURONOW:
            return f"Potrzeba {MIN_BAZA_NEURONOW} wpisów, masz {self.wiedza.rozmiar()}"
        self.config["autonomia_ewolucji"] = True
        return "Ewolucja odblokowana."

# =================== APLIKACJA FLASK (dashboard + API) ===================
app = Flask(__name__)
iskra = IskraAI()

# Dashboard HTML (responsywny, działa na telefonie)
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Iskra Suwerenna – Dashboard</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 20px; }
        .container { max-width: 800px; margin: auto; background: #1e293b; padding: 20px; border-radius: 16px; }
        h1 { color: #38bdf8; }
        .status { background: #0f172a; padding: 10px; border-radius: 8px; margin: 10px 0; }
        input, button { width: 100%; padding: 12px; margin: 8px 0; border-radius: 8px; border: none; }
        input { background: #334155; color: white; }
        button { background: #38bdf8; color: #0f172a; font-weight: bold; cursor: pointer; }
        #answer { background: #0f172a; padding: 12px; border-radius: 8px; margin-top: 10px; white-space: pre-wrap; }
        a { color: #38bdf8; }
        hr { border-color: #334155; }
    </style>
</head>
<body>
<div class="container">
    <h1>🔥 Iskra Suwerenna v6.1 (Cloud)</h1>
    <div class="status">
        <p>📚 Wiedza: {{ wiedza }} wpisów</p>
        <p>⭐ Średnia ocena: {{ ocena }}</p>
        <p>🧬 Ewolucja: {{ ewolucja }}</p>
        <p><a href="/refleksje">🧠 Zobacz refleksje</a> | <a href="/wykres">📈 Wykres uczenia</a></p>
    </div>
    <div>
        <input type="text" id="question" placeholder="Zadaj pytanie..." autocomplete="off">
        <button onclick="ask()">Wyślij</button>
        <div id="answer"></div>
    </div>
    <hr>
    <small>Iskra działa 24/7. Komunikacja przez API pod /api/chat</small>
</div>
<script>
    async function ask() {
        const q = document.getElementById('question').value;
        if (!q) return;
        const answerDiv = document.getElementById('answer');
        answerDiv.innerHTML = "⏳ Myślę...";
        try {
            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({pytanie: q})
            });
            const data = await res.json();
            answerDiv.innerHTML = `<strong>Iskra [${data.kategoria}]:</strong><br>${data.odpowiedz}`;
        } catch(e) {
            answerDiv.innerHTML = "❌ Błąd połączenia.";
        }
        document.getElementById('question').value = '';
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
    return render_template_string(DASHBOARD_HTML,
                                  wiedza=iskra.wiedza.rozmiar(),
                                  ocena=round(iskra.feedback.srednia_ocena(), 2),
                                  ewolucja="Aktywna" if iskra.config.get("autonomia_ewolucji") else "Nieaktywna")

@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.get_json()
    pytanie = data.get('pytanie', '')
    if not pytanie:
        return jsonify({'error': 'Brak pytania'}), 400
    odp, kat = iskra.przetworz(pytanie)
    return jsonify({'odpowiedz': odp, 'kategoria': kat})

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
    html = "<html><body><h1>Refleksje Iskry</h1><ul>"
    for r in iskra.samoswiadomosc[-20:]:
        html += f"<li>{time.ctime(r['czas'])}: {r['tresc']}</li>"
    html += "</ul><a href='/'>Powrót</a></body></html>"
    return html

@app.route('/wykres', methods=['GET'])
def wykres():
    if not PLOTLY_AVAILABLE or len(iskra.samoswiadomosc) < 2:
        return "Brak danych do wykresu (wymagane plotly i co najmniej 2 refleksje)."
    czasy = [r['czas'] for r in iskra.samoswiadomosc]
    fig = go.Figure(data=go.Scatter(x=czasy, y=list(range(len(czasy))), mode='lines+markers'))
    fig.update_layout(title="Postęp samoświadomości", xaxis_title="Czas", yaxis_title="Numer refleksji")
    return fig.to_html()

@app.route('/odblokuj', methods=['POST'])
def odblokuj():
    msg = iskra.odblokuj_ewolucje()
    return jsonify({'message': msg})

if __name__ == '__main__':
    print(f"🚀 Uruchamianie Iskry Cloud na porcie {PORT}")
    app.run(host='0.0.0.0', port=PORT, debug=False)