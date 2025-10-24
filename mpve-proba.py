import pandas as pd
import requests
import json
import numpy as np
from typing import List, Dict, Any, Union

# ==============================================================================
# 1. ADATHALMAZ LÉTREHOZÁSA ÉS KONFIGURÁCIÓ
# ==============================================================================

# Szimulált politikai adathalmaz magyar nyelven. 
# Egy valós projektben ezt egy nagyobb CSV-ből vagy adatbázisból kellene betölteni.
data: Dict[str, List[str]] = {
    'szöveg': [
        "A miniszterelnök mai bejelentése nagyszerű jövőképet vázolt, reményt adva a lakosságnak.",
        "Ez a legújabb adóemelés katasztrofális a kisvállalkozásokra nézve, tönkretesz minket.",
        "A parlament megszavazta a törvényjavaslatot 130 igen szavazattal. Elemzés nélkül.",
        "Ellenzéki vezető élesen kritizálta a kormány javaslatát, helytelenítve a gazdaságra gyakorolt hatását.",
        "Pozitívan értékelem a fővárosi önkormányzat új fejlesztési tervét; ez jó irányba visz.",
        "A sajtótájékoztató délután két órakor kezdődik a Várban, várjuk a részleteket.",
        "Teljesen elhibázott döntés született, ami csak ront a helyzeten és a közérzeten.",
        "Bár a törvény szükséges volt, a végrehajtás kaotikus és elszomorító volt.",
        "A képviselői felszólalás során bemutatott adatok meggyőzőek voltak, és támogatják az ügyet."
    ]
}
df = pd.DataFrame(data)

# OLLAMA BEÁLLÍTÁSOK
OLLAMA_URL: str = "http://localhost:11434/api/generate"
# CSERÉLJE KI 'llama2'-t arra a modellnévre, amit Ollama-ban futtat! 
# Például: "mistral", "llama3", "gemma:2b", stb.
OLLAMA_MODEL: str = "llama2" 

# ==============================================================================
# 2. FŐ ELEMZŐ FUNKCIÓ (OLLAMA API HÍVÁS)
# ==============================================================================

def elemezz_szentiment_ollama(szoveg: str) -> str:
    """
    Elküldi a szöveget az Ollama API-nak szentiment elemzésre a szigorúan meghatározott prompt alapján.
    """
    # A System Prompt: Ez határozza meg a modell szerepét és a kimeneti formátumot.
    system_prompt: str = (
        "Te egy tapasztalt, objektív magyar politikai elemző AI vagy. "
        "A feladatod, hogy a megadott szöveg érzelmi töltetét osztályozd a magyar politikai kontextusban. "
        "Válaszod CSAK EGY SZÓ lehet: 'POZITÍV', 'NEGATÍV' vagy 'SEMLEGES'. "
        "TILOS magyarázatot, indoklást vagy egyéb szöveget fűznöd a válaszhoz!"
    )

    prompt_data: Dict[str, Any] = {
        "model": OLLAMA_MODEL,
        "prompt": szoveg,
        "system": system_prompt,
        "stream": False,
        "options": {
            # Nagyon alacsony hőmérséklet a konzisztencia érdekében.
            "temperature": 0.01 
        }
    }

    try:
        # A kérés elküldése 30 másodperces időtúllépéssel
        response = requests.post(OLLAMA_URL, json=prompt_data, timeout=30)
        response.raise_for_status() # HTTP hibák (pl. 404, 500) kivételt dobnak
        
        result: Dict[str, Any] = response.json()
        
        # Kinyerjük, tisztítjuk és nagybetűssé alakítjuk a választ
        sentiment: str = result['response'].strip().upper()
        
        # Ellenőrzés, hogy a kimenet a várt kategóriák egyike-e
        valid_sentiments: List[str] = ['POZITÍV', 'NEGATÍV', 'SEMLEGES']
        if sentiment in valid_sentiments:
            return sentiment
        else:
            # Ha a modell letért a formátumtól, azt rögzítjük
            return f"HIBÁS_VÁLASZ: {sentiment[:30]}..."

    except requests.exceptions.Timeout:
        return "API_IDŐTÚLLÉPÉS"
    except requests.exceptions.ConnectionError:
        return "API_HIBA (Nincs kapcsolat, az Ollama fut a 11434-es porton?)"
    except requests.exceptions.RequestException as e:
        return f"API_HIBA: {e}"

# ==============================================================================
# 3. ELEMZÉS INDÍTÁSA ÉS EREDMÉNYEK ÖSSZEFOGLALÁSA
# ==============================================================================

if __name__ == "__main__":
    
    print(f"\n{'='*70}")
    print(f"--- Magyar Politikai Elemzés indul ---")
    print(f"Modell: {OLLAMA_MODEL} | API Cím: {OLLAMA_URL}")
    print(f"Eredmény: Minden szöveg elemzése az Ollama API-n keresztül történik.")
    print(f"{'='*70}\n")
    
    # Kézi előrehaladás jelzése
    results_list: List[str] = []
    
    for index, row in df.iterrows():
        szoveg: str = row['szöveg']
        print(f"[{index + 1}/{len(df)}] Elemzés alatt: '{szoveg[:70]}...'")
        result: str = elemezz_szentiment_ollama(szoveg)
        results_list.append(result)
        print(f"    -> Eredmény: {result}")
        
    df['Ollama_Szentiment'] = results_list

    # VÉGSŐ EREDMÉNYEK KIÍRÁSA
    print(f"\n{'='*70}")
    print("--- VÉGSŐ ELEMZÉSI ÖSSZEFOGLALÓ ---")
    print(f"{'='*70}")
    
    # Csak a releváns oszlopok kiírása
    print(df[['szöveg', 'Ollama_Szentiment']])
    
    print(f"\n{'-'*70}")
    
    # Statisztika
    szentiment_eloszlas: pd.Series = df['Ollama_Szentiment'].value_counts()
    print("Szentiment Eloszlás:")
    print(szentiment_eloszlas)
    print(f"{'-'*70}")
