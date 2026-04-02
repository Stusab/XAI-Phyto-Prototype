# src/symptom_catalog.py
import sqlite3
import os
from typing import List, Set

# === KURATIERTE HAUPTSYMPTOME (Manuell gepflegt für beste UX) ===
FIXED_MAIN_SYMPTOMS = [
    "Husten",
    "Schnupfen", 
    "Halsschmerzen",
    "Heiserkeit",
    "Fieber",
    "Kopfschmerzen",
    "Migräne",
    "Ohrenschmerzen",
    "Zahnschmerzen",
    "Magenbeschwerden",
    "Übelkeit",
    "Sodbrennen",
    "Blähungen",
    "Durchfall",
    "Verstopfung",
    "Bauchkrämpfe",
    "Rückenschmerzen",
    "Gelenkschmerzen",
    "Muskelschmerzen",
    "Prellungen",
    "Verstauchungen",
    "Hautprobleme",
    "Ekzeme",
    "Wunden",
    "Verbrennungen",
    "Insektenstiche",
    "Schlafprobleme",
    "Unruhe",
    "Nervosität",
    "Erschöpfung",
    "Konzentrationsprobleme",
    "Stimmungsschwankungen",
    "Menstruationsbeschwerden",
    "Wechseljahresbeschwerden",
    "Blasenentzündung",
    "Prostatabeschwerden",
    "Kreislaufprobleme",
    "Wassereinlagerungen"
]

def get_main_symptoms(db_path: str = None) -> List[str]:
    """
    Gibt die kuratierte, fixe Liste der Hauptsymptome zurück.
    (DB-Extraktion nicht mehr nötig - zu viele Duplikate)
    """
    return sorted(FIXED_MAIN_SYMPTOMS)



def get_symptom_subtypes(main_symptom: str) -> List[dict]:
    """
    Gibt Detail-Optionen für ein Hauptsymptom zurück.
    Wird in Stufe 2 der UI genutzt.
    """
    subtype_map = {
        "Husten": [
            {"label": "Trocken/Reizhusten", "value": "dry", "keywords": ["reizhusten", "trocken"]},
            {"label": "Mit Schleim/Auswurf", "value": "productive", "keywords": ["verschleimt", "schleim", "auswurf"]},
            {"label": "Egal/Weiß nicht", "value": "any", "keywords": []}
        ],
        "Halsschmerzen": [
            {"label": "Kratzend/Trocken", "value": "scratchy", "keywords": ["kratzen", "trocken", "rauh"]},
            {"label": "Entzündet/Geschwollen", "value": "inflamed", "keywords": ["entzündet", "geschwollen"]},
            {"label": "Egal/Weiß nicht", "value": "any", "keywords": []}
        ],
        "Magenbeschwerden": [
            {"label": "Mit Krämpfen", "value": "cramps", "keywords": ["krampf", "kolik"]},
            {"label": "Übelkeit/Sodbrennen", "value": "nausea", "keywords": ["übelkeit", "sodbrennen"]},
            {"label": "Blähungen", "value": "bloating", "keywords": ["blähung", "aufgebläht"]},
            {"label": "Egal/Weiß nicht", "value": "any", "keywords": []}
        ],
        "Kopfschmerzen": [
            {"label": "Dumpf/Drückend", "value": "tension", "keywords": ["dumpf", "drückend"]},
            {"label": "Pochend/Stechend (Migräne)", "value": "migraine", "keywords": ["pochend", "migräne"]},
            {"label": "Egal/Weiß nicht", "value": "any", "keywords": []}
        ],
        # Weitere Symptome haben keine Subtypen → leere Liste
    }
    
    return subtype_map.get(main_symptom, [])
