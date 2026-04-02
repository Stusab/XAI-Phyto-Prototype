"""
Datenmodelle für das Heilpflanzen-Empfehlungssystem
"""
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class UserProfile:
    """
    Repräsentiert das Gesundheitsprofil eines Nutzers.
    Wird vom Safety Layer verwendet, um unsichere Pflanzen zu filtern.
    """
    # Basis-Daten
    age: int  # Alter in Jahren
    is_pregnant: bool = False  # Schwangerschaft
    is_breastfeeding: bool = False  # Stillzeit
    
    # Kontraindikationen (Tags aus plant_contra_tag)
    # Der User kann mehrere haben, z.B. ["HERZ", "DIABETES"]
    conditions: List[str] = field(default_factory=list)


# --- GRUPPIERTE KONTRAINDIKATIONS-TAGS ---
# Diese stammen aus deiner Datenbank (plant_contra_tag.tag)
# Gruppiert nach medizinischen Kategorien für besseres UI

CONTRAINDICATION_CATEGORIES: Dict[str, List[str]] = {
    "Allergien & Immunsystem": [
        "ALLERGIEN",
        "IMMUNSYSTEM",
    ],
    
    "Herz & Kreislauf": [
        "HERZ",
        "BLUTHOCHDRUCK",
        "BLUTDRUCK",
        "KREISLAUF",
        "BLUTGERINNUNG",
    ],
    
    "Verdauungssystem": [
        "MAGEN",
        "DARM",
        "GALLE",
        "SPEISERÖHRE",
        "SPEISEROEHRE",  # Variante in deinen Daten
    ],
    
    "Stoffwechsel & Hormone": [
        "DIABETES",
        "STOFFWECHSEL",
        "HYPOPHYSE",
    ],
    
    "Niere, Leber & Ausscheidung": [
        "NIERE",
        "LEBER",
        "HARNWEGE",
    ],
    
    "Atemwege": [
        "LUNGE",
    ],
    
    "Haut & Schleimhäute": [
        "HAUT",
        "HAUT_EKZEM",
    ],
    
    "Sinnesorgane": [
        "AUGEN",
    ],
    
    "Nervensystem & Psychiatrie": [
        "EPILEPSIE",
        "SCHIZOPHRENIE",
    ],
    
    "Onkologie": [
        "TUMOR",
    ],
    
    "Infektionen & Fieber": [
        "INFEKTIONEN",
        "FIEBER",
    ],
    
    "Sonstige": [
        "CHRONISCHE_KRANKHEIT",
        "FAHRTÜCHTIGKEIT",
        "PROSTATA",
        "SÄUGLINGE",      # Wird über age gefiltert
        "KLEINKINDER",    # Wird über age gefiltert
    ],
}


# Flache Liste aller Tags (für Validierung)
AVAILABLE_CONTRAINDICATION_TAGS = [
    tag 
    for category_tags in CONTRAINDICATION_CATEGORIES.values() 
    for tag in category_tags
]


# --- HELPER FUNCTION: Tag-Validierung ---
def validate_condition_tags(conditions: List[str]) -> List[str]:
    """
    Prüft, ob die eingegebenen Conditions gültige Tags sind.
    Gibt nur valide Tags zurück und warnt bei unbekannten.
    
    Args:
        conditions: Liste von Tag-Strings
        
    Returns:
        Liste der validen Tags (bereinigt und in Großbuchstaben)
    """
    valid_tags = []
    for tag in conditions:
        tag_upper = tag.upper()  # Normalisierung
        
        if tag_upper in AVAILABLE_CONTRAINDICATION_TAGS:
            valid_tags.append(tag_upper)
        else:
            print(f"⚠️  WARNUNG: Unbekanntes Tag '{tag}' wird ignoriert.")
    return valid_tags


# --- HELPER FUNCTION: Alle Tags einer Kategorie holen ---
def get_tags_by_category(category_name: str) -> List[str]:
    """
    Gibt alle Tags einer bestimmten Kategorie zurück.
    Nützlich für UI-Gruppierung.
    
    Args:
        category_name: Name der Kategorie (z.B. "Herz & Kreislauf")
        
    Returns:
        Liste der Tags in dieser Kategorie
    """
    return CONTRAINDICATION_CATEGORIES.get(category_name, [])
