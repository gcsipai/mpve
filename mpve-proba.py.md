# 🇭🇺 Ollama-alapú Magyar Politikai Véleményelemző (MPVE)

## 1. Cél és Technológiai Áttekintés

Ez a dokumentáció egy modern, **Large Language Model (LLM)** alapú megközelítést ír le a magyar nyelvű politikai szentiment elemzésére. A klasszikus gépi tanulás helyett az elemzés a lokálisan futtatott Ollama szerveren keresztül, szigorú prompt-utasítások alapján történik.

| Paraméter | Leírás | Ikon |
| :--- | :--- | :--- |
| **Cél** | Magyar politikai szövegek érzelmi töltetének (Pozitív, Negatív, Semleges) osztályozása. | 🎯 |
| **Fő Alap** | **Ollama** (Lokális LLM futtatási keretrendszer). | 🦙 |
| **Nyelv** | **Python** (Az API kommunikációhoz és az adatok kezeléséhez). | 🐍 |
| **Adatkezelés** | **Pandas** (Az eredmények táblázatos kezeléséhez). | 🐼 |
| **Módszer** | **Prompt-mérnökség** (Szigorú utasítások az LLM-nek). | ⚙️ |

***

## 2. Rendszerkövetelmények és Beállítások

A kód futtatásához elengedhetetlen, hogy a lokális környezet megfelelően legyen beállítva.

### A. Telepítési Lépések

1.  **Ollama Telepítése:** Telepítse az Ollama szoftvert a hivatalos weboldalról.
2.  **Modell Letöltése:** Töltse le a kívánt LLM-et (pl. Llama 2 vagy Mistral) a parancssorból:
    ```bash
    ollama pull llama2
    ```
3.  **Python Könyvtárak:** Telepítse a szükséges Python csomagokat:
    ```bash
    pip install requests pandas
    ```

### B. Konfigurációs Változók

Ellenőrizze, hogy a Python kódban a következő változók a helyes beállításokat tartalmazzák:

| Változó | Alapértelmezett Érték | Szerepe |
| :--- | :--- | :--- |
| `OLLAMA_URL` | `"http://localhost:11434/api/generate"` | Az Ollama szerver API címe. |
| `OLLAMA_MODEL` | `"llama2"` | A futtatandó LLM neve. **Cseréld le, ha más modellt használsz!** |

***

## 3. Python Kód: Az Elemző Rendszer

A kód feladata, hogy a `pandas` DataFrame szövegeit egyenként elküldje a helyi Ollama API-nak, és az eredményt egy új oszlopban gyűjtse.

```python
import pandas as pd
import requests
import json
import numpy as np
from typing import List, Dict, Any, Union

# ==============================================================================
# 1. ADATHALMAZ LÉTREHOZÁSA ÉS KONFIGURÁCIÓ
# ==============================================================================

# Szimulált politikai adathalmaz (valós projekthez nagyobb, külső adatforrás kell).
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
# !! ELLENŐRIZZE A MODELL NEVÉT !!
OLLAMA_MODEL: str = "llama2" 

# ==============================================================================
# 2. FŐ ELEMZŐ FUNKCIÓ (OLLAMA API HÍVÁS)
# ==============================================================================

def elemezz_szentiment_ollama(szoveg: str) -> str:
    """Elküldi a szöveget az Ollama API-nak szentiment elemzésre a prompt alapján."""
    
    # 🚨 SYSTEM PROMPT: A modell viselkedésének szigorú szabályozása.
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
            # Alacsony hőmérséklet a konzisztens, gépi válaszokért.
            "temperature": 0.01 
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=prompt_data, timeout=30)
        response.raise_for_status() 
        
        result: Dict[str, Any] = response.json()
        sentiment: str = result['response'].strip().upper()
        
        valid_sentiments: List[str] = ['POZITÍV', 'NEGATÍV', 'SEMLEGES']
        if sentiment in valid_sentiments:
            return sentiment
        else:
            return f"HIBÁS_VÁLASZ: {sentiment[:30]}..."

    except requests.exceptions.Timeout:
        return "API_IDŐTÚLLÉPÉS"
    except requests.exceptions.ConnectionError:
        return "API_HIBA (Ollama nem elérhető)"
    except requests.exceptions.RequestException as e:
        return f"API_HIBA: {e}"

# ==============================================================================
# 3. ELEMZÉS INDÍTÁSA ÉS EREDMÉNYEK ÖSSZEFOGLALÁSA
# ==============================================================================

if __name__ == "__main__":
    
    print(f"\n{'='*70}")
    print(f"--- 🇭🇺 Magyar Politikai Elemzés Indul ---")
    print(f"Modell: {OLLAMA_MODEL} | API Cím: {OLLAMA_URL}")
    print(f"{'='*70}\n")
    
    results_list: List[str] = []
    
    for index, row in df.iterrows():
        szoveg: str = row['szöveg']
        print(f"[{index + 1}/{len(df)}] Elemzés: '{szoveg[:70]}...'")
        result: str = elemezz_szentiment_ollama(szoveg)
        results_list.append(result)
        print(f"    ✅ Eredmény: {result}")
        
    df['Ollama_Szentiment'] = results_list

    # VÉGSŐ EREDMÉNYEK KIÍRÁSA
    print(f"\n{'='*70}")
    print("--- 📊 VÉGSŐ ELEMZÉSI ÖSSZEFOGLALÓ ---")
    print(f"{'='*70}")
    
    print(df[['szöveg', 'Ollama_Szentiment']])
    
    print(f"\n{'-'*70}")
    
    # Statisztika
    szentiment_eloszlas: pd.Series = df['Ollama_Szentiment'].value_counts()
    print("Szentiment Eloszlás:")
    print(szentiment_eloszlas)
    print(f"{'-'*70}")
