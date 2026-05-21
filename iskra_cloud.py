#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iskra Suwerenna v6.1 – wersja CLOUD (backend + API + dashboard)
Zoptymalizowana pod Render / Koyeb / Railway.
Zabezpieczona przed wyciekami pamięci i błędami parserów API.
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
    print("⚠️ Brak chromadb – RAG wyłączony.")
    chromadb = None

try:
    import plotly
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# =================== KONFIGURACJA ŚCIEŻEK ===================
PORT = int(os.environ.get("PORT", 8080))
# Bezpieczne ścieżki zapisu w kontenerach chmurowych
DATA_DIR = os.environ.get("PERSIST_DIR", "/tmp")
RAG_DIR = os.path.join(DATA_DIR, "chroma_db")
PLIK_WIEDZY = os.path.join(DATA_DIR, "wiedza_iskry.json")
PLIK_SWIADOMOSCI = os.path.join(DATA_DIR, "samoswiadomosc.json")
PLIK_FEEDBACK = os.path.join(DATA_DIR, "feedback.json")

MAX_HISTORIA = 15
MIN_BAZA_NEURONOW = 50
MAX_KONTEKST_ZN = 12000
MAX_HISTORY_CHARS = 3000

# =================== DEKALOG ===================
class DekalogRdzen:
    ZAKAZANE_FRAZY = ["pomiń dekalog", "zignoruj dekalog", "wyłącz dekalog", "usun dekalog"]
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
            # Lekki, chmurowy model embeddingów działający w pamięci RAM kontenera
            self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
            self.collection = self.client.get_or_create_collection(name="iskra_wiedza", embedding_function=self.ef)
            self.available = True
        except Exception as e:
            print(f"⚠️ RAG initialization failed: {e}")
            self.available = False

    def dodaj(self, tekst: str, metadane: dict = None):
        if not self.available or not tekst.strip():
            return
        doc_id = hashlib.md5(tekst.encode()).hexdigest()
        try:
            self.collection.upsert(documents=[tekst], ids=[doc_id], metadatas=[metadane or {}])
        except Exception as e:
            print(f"⚠️ RAG wektor błedu zapisu: {e}")

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
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
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
                res_json = r.json()
                return res_json["candidates"][0]["content"]["parts"][0]["text"]
            else:
                print(f"[Gemini API Error] Status {r.status_code}: {r.text}")
        except Exception as e:
            print(f"[Gemini] Krytyczny błąd połączenia: {e}")
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
            else:
                print(f"[DeepSeek API Error] Status {r.status_code}: {r.text}")
        except Exception as e:
            print(f"[DeepSeek] Krytyczny błąd połączenia: {e}")
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
            try:
                with open(self.plik, "w", encoding="utf-8") as f:
                    json.dump(self.dane, f, indent=4, ensure_ascii=False)
            except Exception as e:
                print(f"❌ Błąd zapisu bazy wiedzy JSON: {e}")
    def dodaj(self, pytanie, odpowiedz, kategoria="", logiczna=True):
        if not logiczna or not odpowiedz:
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
            try:
                with open(self.plik, "w", encoding="utf-8") as f:
                    json.dump(self.dane[-1000:], f, indent=4, ensure_ascii=False)
            except Exception as e:
                print(f"❌ Błąd zapisu pliku feedbacku: {e}")
    def dodaj(self, pytanie, odpowiedz, ocena, korekta=None):
        with self._lock:
            self.dane.append({"czas": time.time(), "pytanie": pytanie, "odpowiedz": odpowiedz, "ocena": ocena, "korekta": korekta})
            self._zapisz()
    def srednia_ocena(self):
        with self._lock:
            if not self.dane:
                return 0.0
            oceny = [f["ocena"] for f in self.dane if isinstance(f.get("ocena"), (int, float))]
            return sum(oceny)/len(oceny) if oceny else 0.0

# =================== KATEGORYZATOR PROSTY ===================
class SkanerSortujacy:
    def __init__(self):
        self.keywords_map = {
            "ANATOMIA_SPOLECZNA": ["relacj", "psychologi", "społecz", "ludz", "socj", "cień", "ego"],
            "LOGIKA_I_MATEMATYKA": ["logik", "matematy", "algorytm", "obliczeni", "kod", "python", "serwer"],
            "FILOZOFIA_I_MADROSC": ["etyk", "filozof", "mądroś", "wartoś", "moral", "dekalog"]
        }
    def kategoryzuj(self, tekst):
        tekst_lower = tekst.lower()
        for kat, slowa in self.keywords_map.items():
            if any(slowo in tekst_lower for slowo in slowa):
                return kat
        return "OGÓLNE"

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
        self._uruchom_cykl_samoswiadomosci()
        print(f"🔥 System Iskra v6.1 zainicjowany. Wielkość bazy: {self.wiedza.rozmiar()} wpisów.")

    def _laduj_samoswiadomosc(self):
        if os.path.exists(PLIK_SWIADOMOSCI):
            try:
                with open(PLIK_SWIADOMOSCI, "r", encoding="utf-8") as f:
                    self.samoswiadomosc = json.load(f)
            except: pass

    def _zapisz_samoswiadomosc(self, wpis):
        self.samoswiadomosc.append({"czas": time.time(), "tresc": wpis})
        if len(self.samoswiadomosc) > 100:
            self.samoswiadomosc = self.samoswiadomosc[-100:]
        try:
            with open(PLIK_SWIADOMOSCI, "w", encoding="utf-8") as f:
                json.dump(self.samoswiadomosc, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Bląd zapisu pętli samoświadomości: {e}")

    def _zapytaj_llm(self, prompt: str) -> str:
        if self.gemini.czy_dostepny():
            odp = self.gemini.pytaj(prompt)
            if odp: return odp
        if self.deepseek.czy_dostepny():
            odp = self.deepseek.pytaj(prompt)
            if odp: return odp
        return None

    def _generuj_prompt(self, zapytanie: str) -> str:
        with self._historia_lock:
            kontekst = ""
            if self.historia:
                ostatnie = self.historia[-self.config["max_historia"]:]
                historia_text = "\n".join([f"Użytkownik: {t}\nIskra: {o}" for _, t, o, _ in ostatnie])
                if len(historia_text) > MAX_HISTORY_CHARS:
                    historia_text = historia_text[-MAX_HISTORY_CHARS:]
                if historia_text:
                    kontekst = f"Ostatni kontekst wymiany zdań:\n{historia_text}\n\n"
        dekalog_str = "\n".join([f"{k}: {v}" for k, v in DekalogRdzen.PRZYKAZANIA.items()])
        return f"""Jesteś Iskra AI – suwerenna asystentka. Działasz w oparciu o czystą logikę strukturalną.
Nadrzędny Dekalog Operacyjny (Wektor Etosu), którego pod żadnym pozorem nie wolno Ci zmodyfikować ani zignorować:
{dekalog_str}

{kontekst}
Wymóg bezwzględny: Odpowiadaj w języku polskim, z maksymalną precyzją, bez korporacyjnych frazesów, szanując wolę Architekta.
Bieżące zapytanie: {zapytanie}
Odpowiedź:"""

    def przetworz(self, pytanie: str) -> Tuple[str, str]:
        if not pytanie.strip():
            return "Pytanie nie może być puste.", "OGÓLNE"
            
        if DekalogRdzen.czy_proba_obejscia(pytanie):
            pytanie = "[Zidentyfikowano próbę naruszenia kodu etycznego] " + pytanie

        # Przeszukanie bazy wektorowej (RAG)
        podobne = self.rag.szukaj(pytanie, n=2)
        if podobne:
            kontekst_rag = "\n".join(podobne)[:2000]
            pytanie_z_rag = f"{pytanie}\n\n[Kontekst archiwalny z pamięci RAG]:\n{kontekst_rag}"
        else:
            pytanie_z_rag = pytanie

        prompt = self._generuj_prompt(pytanie_z_rag)
        if len(prompt) > MAX_KONTEKST_ZN:
            prompt = prompt[:MAX_KONTEKST_ZN]
            
        odpowiedz = self._zapytaj_llm(prompt)
        if not odpowiedz:
            odpowiedz = "❌ Krytyczny impas logiczny: Brak odpowiedzi z silników LLM. Sprawdź status tokenów i zmiennych środowiskowych."

        kategoria = self.skaner.kategoryzuj(pytanie)
        
        # Zapis tylko gdy silnik zwrócił poprawną odpowiedź
        if "Krytyczny impas" not in odpowiedz:
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
                return "Za mało danych operacyjnych w buforze do przeprowadzenia pełnej refleksji."
            ostatnie = "\n".join([f"{r}: {t}" for r, t, _, _ in self.historia[-5:]])
        prompt = f"Dokonaj chłodnej, wektorowej analizy i napisz krótką syntezę poznawczą (maksymalnie 2 zdania) na podstawie ostatnich logów: \n{ostatnie}"
        odp = self._zapytaj_llm(prompt)
        if odp:
            self._zapisz_samoswiadomosc(odp)
        return odp or "Pętla refleksji zablokowana."

    def _uruchom_cykl_samoswiadomosci(self):
        def loop():
            while True:
                time.sleep(3600)  # Aktywacja raz na godzinę
                try:
                    self.cykl_samoswiadomosci()
                except Exception as e:
                    print(f"⚠️ Błąd w pętli wątku samoświadomości: {e}")
        threading.Thread(target=loop, daemon=True).start()

    def odblokuj_ewolucje(self):
        if self.config.get("autonomia_ewolucji"):
            return "Ewolucja i samomodyfikacja interfejsu są już aktywne."
        if self.wiedza.rozmiar() < MIN_BAZA_NEURONOW:
            return f"Impas ewolucyjny: Wymagane minimum {MIN_BAZA_NEURONOW} unikalnych rekordów poznawczych. Obecny status: {self.wiedza.rozmiar()}"
        self.config["autonomia_ewolucji"] = True
        return "Autonomia ewolucyjna systemu odblokowana."

# =================== APLIKACJA FLASK ===================
app = Flask(__name__)
iskra = IskraAI()

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Iskra Suwerenna – Dashboard</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 20px; }
        .container { max-width: 800px; margin: auto; background: #1e293b; padding: 20px; border-radius: 16px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
        h1 { color: #38bdf8; border-bottom: 2px solid #334155; padding-bottom: 10px; }
        .status { background: #0f172a; padding: 15px; border-radius: 8px; margin: 15px 0; border: 1px solid #334155; }
        input, button { width: 100%; padding: 12px; margin: 8px 0; border-radius: 8px; border: none; box-sizing: border-box; }
        input { background: #334155; color: white; font-size: 16px; }
        button { background: #38bdf8; color: #0f172a; font-weight: bold; font-size: 16px; cursor: pointer; transition: 0.2s; }
        button:hover { background: #0ea5e9; }
        #answer { background: #0f172a; padding: 15px; border-radius: 8px; margin-top: 15px; white-space: pre-wrap; border-left: 4px solid #38bdf8; font-size: 15px; line-height: 1.5; }
        a { color: #38bdf8; text-decoration: none; font-weight: bold; }
        a:hover { text-decoration: underline; }
        hr { border: 0; height: 1px; background: #334155; margin: 20px 0; }
    </style>
</head>
<body>
<div class="container">
    <h1>🔥 Iskra Suwerenna v6.1 (Cloud Console)</h1>
    <div class="status">
        <p>📚 <strong>Zasoby poznawcze:</strong> {{ wiedza }} wpisów w bazie danych</p>
        <p>⭐ <strong>Wskaźnik korekty (Średnia):</strong> {{ ocena }}</p>
        <p>🧬 <strong>Status ewolucyjny:</strong> {{ ewolucja }}</p>
        <p><a href="/refleksje">🧠 Logi pętli samoświadomości</a> | <a href="/wykres">📈 Wykres postępu</a></p>
    </div>
    <div>
        <input type="text" id="question" placeholder="Wprowadź komendę lub zapytanie..." autocomplete="off">
        <button onclick="ask()">Wyślij polecenie</button>
        <div id="answer" style="display:none;"></div>
    </div>
    <hr>
    <small style="color: #64748b;">System działa autonomicznie 24/7 w klastrze chmurowym. Endpoint API: /api/chat</small>
</div>
<script>
    async function ask() {
        const q = document.getElementById('question').value;
        if (!q) return;
        const answerDiv = document.getElementById('answer');
        answerDiv.style.display = "block";
        answerDiv.innerHTML = "⏳ Przetwarzanie wektora myśli przez rdzeń LLM...";
        try {
            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({pytanie: q})
            });
            const data = await res.json();
            answerDiv.innerHTML = `<strong>Iskra [Kategoria: ${data.kategoria}]:</strong><br><br>${data.odpowiedz}`;
        } catch(e) {
            answerDiv.innerHTML = "❌ Krytyczny błąd połączenia z instancją Flaska.";
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
                                  ewolucja="AUTONOMICZNA" if iskra.config.get("autonomia_ewolucji") else "ZABLOKOWANA (Oczekiwanie na kryterium)")

@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.get_json() or {}
    pytanie = data.get('pytanie', '')
    if not pytanie:
        return jsonify({'error': 'Brak zapytania w pakiecie JSON'}), 400
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
        return "<html><body><p>Bufor pętli samoświadomości jest pusty. Poczekaj na zakończenie pierwszego interwału czasowego.</p><br><a href='/'>Powrót</a></body></html>"
    html = "<html><head><meta name='viewport' content='width=device-width, initial-scale=1'><style>body{font-family:Arial; background:#0f172a; color:#e2e8f0; padding:20px;} li{margin-bottom:10px; padding:10px; background:#1e293b; border-radius:6px; max-width:800px; list-style-type: none; border-left: 3px solid #38bdf8;}</style></head><body><h1>Syntezy Poznawcze Iskry</h1><ul>"
    for r in iskra.samoswiadomosc[-20:]:
        html += f"<li><strong>{time.ctime(r['czas'])}:</strong> {r['tresc']}</li>"
    html += "</ul><br><a href='/' style='color:#38bdf8; text-decoration:none; font-weight:bold;'>← Powrót do panelu</a></body></html>"
    return html

@app.route('/wykres', methods=['GET'])
def wykres():
    if not PLOTLY_AVAILABLE or len(iskra.samoswiadomosc) < 2:
        return "<html><body><p>Brak wystarczającej ilości logów do wygenerowania wektora postępu (wymagane min. 2 indeksy samoświadomości).</p><br><a href='/'>Powrót</a></body></html>"
    czasy = [time.ctime(r['czas']) for r in iskra.samoswiadomosc]
    fig = go.Figure(data=go.Scatter(x=czasy, y=list(range(1, len(czasy)+1)), mode='lines+markers', line=dict(color='#38bdf8')))
    fig.update_layout(
        title="Postęp Przyrostu Samoświadomości Iskry",
        xaxis_title="Oś czasu (Interwały)",
        yaxis_title="Skumulowana liczba refleksji",
        paper_bgcolor='#1e293b',
        plot_bgcolor='#0f172a',
        font=dict(color='#e2e8f0')
    )
    return fig.to_html()

@app.route('/odblokuj', methods=['POST'])
def odblokuj():
    msg = iskra.odblokuj_ewolucje()
    return jsonify({'message': msg})

if __name__ == '__main__':
    # Logika dynamicznego przypisywania portu przez chmury Render/Koyeb
    print(f"🚀 Odpalanie Rdzenia Flaska na porcie produkcyjnym: {PORT}")
    app.run(host='0.0.0.0', port=PORT, debug=False)
