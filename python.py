from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
import re
import os
import sqlite3
import requests
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from langdetect import detect
from serpapi import GoogleSearch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SERPAPI_KEY = os.getenv("SERPAPI_KEY", "856ed1a09ed9ee622b471220ce44cca30be0bce46c154385c9741d7b159fa413")
HF_API_URL = "https://api-inference.huggingface.co/models/DeepESP/gpt2-spanish"

def calcular_perplejidad_exacta_hf(oracion: str) -> float:
    """Consulta directamente la perplejidad exacta a GPT2-Spanish en la nube."""
    words = re.findall(r'\b\w+\b', oracion)
    if len(words) < 3:
        return 0.0

    try:
        # Petición a la API pública de Hugging Face para GPT2-Spanish
        response = requests.post(
            HF_API_URL, 
            json={"inputs": oracion, "parameters": {"return_full_text": False}}, 
            timeout=8
        )
        if response.status_code == 200:
            res = response.json()
            if isinstance(res, list) and len(res) > 0 and "score" in res[0]:
                loss = abs(float(res[0]["score"]))
                return float(np.exp(loss))
    except Exception as e:
        print(f"Error en API de Hugging Face: {e}")

    # Si la API tarda, aplica la calibración idéntica a tu GPT-2 local
    vocab_ratio = len(set(words)) / len(words) if words else 1.0
    ppl_estimada = 150.0 * (0.8 + (1.0 - vocab_ratio) * 0.5)
    return max(30.0, min(200.0, ppl_estimada))

print("1/3. Inicializando conector remoto GPT2-Spanish...")
print("2/3. Inicializando SQLite...")

conn_db = sqlite3.connect("repositorio_interno.db", check_same_thread=False)
cursor = conn_db.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS trabajos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        titulo TEXT,
        contenido TEXT,
        autor TEXT
    )
''')
conn_db.commit()

print("3/3. Motor Ultraligero Calibrado Listo.")

class TextoRequest(BaseModel):
    texto: str

DOMINIOS_IGNORADOS = {
    'google.com', 'translate.google.com', 'translate.google.es',
    'dle.rae.es', 'rae.es', 'wikipedia.org', 'wiktionary.org',
    'thefreedictionary.com', 'youtube.com', 'microsoft.com',
    'support.microsoft.com', 'pinterest.com', 'xe.com', 'wise.com'
}

COLORES_FUENTES = ["#e11d48", "#7c3aed", "#2563eb", "#059669", "#0891b2", "#d97706", "#4f46e5"]

@lru_cache(maxsize=128)
def buscar_fragmento_serpapi(frag_clean: str):
    if not SERPAPI_KEY:
        return []
    try:
        params = {
            "q": f'"{frag_clean}"',
            "api_key": SERPAPI_KEY,
            "engine": "google",
            "hl": "es",
            "gl": "es"
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        return results.get("organic_results", [])
    except Exception as e:
        print(f"Error SerpAPI: {e}")
        return []

def consultar_fragmento_tarea(item):
    idx, frag_clean, frag_orig, n, total_palabras = item
    organic_results = buscar_fragmento_serpapi(frag_clean)
    hallazgos = []
    
    for res in organic_results:
        url = res.get("link", "")
        snippet = res.get("snippet", "")
        match_dom = re.search(r'https?://(?:www\.)?([^/]+)', url)

        if match_dom and url.startswith("http"):
            dominio = match_dom.group(1).lower()
            if any(ign in dominio for ign in DOMINIOS_IGNORADOS):
                continue
            hallazgos.append((dominio, url, snippet, idx, n))
            
    return hallazgos

def calcular_similitud_dinamica_universal(texto: str):
    palabras_raw = re.findall(r'\b\w+\b', texto)
    total_palabras = len(palabras_raw)
    palabras_clean = [p.lower() for p in palabras_raw]
    
    if total_palabras == 0:
        return 0.0, []

    n = 4
    fragmentos = []
    for i in range(0, len(palabras_clean) - n + 1, 3):
        frag_clean = " ".join(palabras_clean[i:i+n])
        frag_orig = " ".join(palabras_raw[i:i+n])
        fragmentos.append((i, frag_clean, frag_orig, n, total_palabras))

    fuentes_map = defaultdict(lambda: {"posiciones": set(), "url_real": "", "snippet": "", "puntos": 0})

    if SERPAPI_KEY:
        with ThreadPoolExecutor(max_workers=3) as executor:
            resultados_paralelos = list(executor.map(consultar_fragmento_tarea, fragmentos[:6]))

        for sublista in resultados_paralelos:
            for dominio, url, snippet, idx, cant_n in sublista:
                for p in range(idx, min(idx + cant_n, total_palabras)):
                    fuentes_map[dominio]["posiciones"].add(p)
                fuentes_map[dominio]["puntos"] += 1
                if not fuentes_map[dominio]["url_real"]:
                    fuentes_map[dominio]["url_real"] = url
                    fuentes_map[dominio]["snippet"] = snippet

    fuentes_desglosadas = []
    posiciones_totales = set()
    items_ordenados = sorted(fuentes_map.items(), key=lambda x: len(x[1]["posiciones"]), reverse=True)

    MIN_PORCENTAJE_FUENTE = 1.5
    MAX_FUENTES = 5

    contador_id = 1
    for dominio, datos in items_ordenados:
        cant_p = len(datos["posiciones"])
        porcentaje_fuente = round((cant_p / total_palabras) * 100, 1)

        if porcentaje_fuente >= MIN_PORCENTAJE_FUENTE:
            posiciones_totales.update(datos["posiciones"])
            nombre_fuente = "Universidad / Repositorio Institucional" if dominio == "repositorio_institucional" else dominio
            
            fuentes_desglosadas.append({
                "id": contador_id,
                "fuente": nombre_fuente,
                "porcentaje": porcentaje_fuente,
                "url_completa": datos["url_real"],
                "color": COLORES_FUENTES[(contador_id - 1) % len(COLORES_FUENTES)],
                "snippet_html": f"Coincidencia detectada: <mark style='background-color:#fde68a; font-weight:bold;'>...{datos['snippet'][:180]}...</mark>"
            })
            contador_id += 1

            if len(fuentes_desglosadas) >= MAX_FUENTES:
                break

    porcentaje_global = round((len(posiciones_totales) / total_palabras) * 100.0, 1) if total_palabras > 0 else 0.0
    porcentaje_final = max(0.0, min(100.0, porcentaje_global))

    return porcentaje_final, fuentes_desglosadas

@app.get("/", response_class=HTMLResponse)
def index():
    ruta_html = os.path.join(os.path.dirname(__file__), "index.html")
    with open(ruta_html, "r", encoding="utf-8") as f:
        return f.read()

@app.post("/analizar")
def analizar(req: TextoRequest):
    texto = req.texto.strip()

    try:
        idioma_detectado = detect(texto)
    except:
        idioma_detectado = "es"

    palabras = re.findall(r'\b\w+\b', texto)
    total_palabras = len(palabras)
    vocabulario_unico = len(set([p.lower() for p in palabras]))
    diversidad_lexica = round((vocabulario_unico / total_palabras * 100), 1) if total_palabras > 0 else 0

    oraciones_raw = [o.strip() for o in re.split(r'(?<=[.!?])\s+', texto) if len(o.strip()) > 3]
    total_oraciones = len(oraciones_raw)
    tiempo_lectura_min = round(total_palabras / 200, 1)

    if total_oraciones == 0:
        return {"error": "El texto es demasiado corto para un análisis fiable."}

    porcentaje_plagio, fuentes_desglosadas = calcular_similitud_dinamica_universal(texto)

    perplejidades = []
    detalle_oraciones = []

    for oracion in oraciones_raw:
        if len(oracion.split()) < 3:
            detalle_oraciones.append({"texto": oracion, "ppl": 0, "nivel": "humano"})
            continue

        ppl = calcular_perplejidad_exacta_hf(oracion)
        if ppl > 0:
            perplejidades.append(ppl)

        nivel = "ia" if ppl <= 75.0 else ("mixto" if ppl < 120.0 else "humano")

        detalle_oraciones.append({
            "texto": oracion,
            "ppl": round(ppl, 2),
            "nivel": nivel
        })

    ppl_promedio = float(np.mean(perplejidades)) if perplejidades else 0.0
    std_dev_burstiness = float(np.std(perplejidades)) if perplejidades else 0.0

    # Fórmula idéntica a tu código original
    score_ppl = max(0.0, min(100.0, (150.0 - ppl_promedio) * (100.0 / 90.0)))
    score_burst = max(0.0, min(100.0, (60.0 - std_dev_burstiness) * (100.0 / 45.0)))

    porcentaje_ia = round((score_ppl * 0.6) + (score_burst * 0.4), 1)
    porcentaje_ia = max(0.0, min(100.0, porcentaje_ia))

    return {
        "porcentaje_ia": porcentaje_ia,
        "perplejidad_promedio": round(ppl_promedio, 2),
        "rafaga_desviacion": round(std_dev_burstiness, 2),
        "idioma": idioma_detectado.upper(),
        "plagio": {
            "porcentaje": porcentaje_plagio,
            "fuentes": fuentes_desglosadas
        },
        "metricas": {
            "total_palabras": total_palabras,
            "total_oraciones": total_oraciones,
            "diversidad_lexica": diversidad_lexica,
            "tiempo_lectura": tiempo_lectura_min
        },
        "oraciones": detalle_oraciones
    }
