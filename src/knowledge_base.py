"""
knowledge_base.py (neu geschrieben, buch-zentriert + komplette DB-Liste integriert)

Zentrale Quelle für Logic Injection: Gruppiert ALLE DB-Wirkstoffe in Klassen basierend auf Buch-Text.
Bleibt wissenschaftlich korrekt: Zuweisungen nur bei etablierten Zuordnungen (Buch/Phyto-Standard).
Exakte DB-Spaltennamen (case-sensitive, wie gelistet), für Boost 'chem_{name}'.

Fokus: Symptome -> Gruppen -> DB-Spalten. XAI via Mechanismen.
"""

# 1. PHYTO_GROUPS: Vollständige DB-Liste gruppiert (Buch-Klassen priorisiert)
PHYTO_GROUPS = {
    # ANTHRANOIDE (Buch: Abführend. Keine expliziten DB-Matches → leer, erweiterbar)
    "anthranoids": [],

    # CUMARINE (Buch: Gerinnung etc. DB: Cumarine)
    "coumarins": [
        "Cumarine"
    ],

    # FLAVONOIDE (Buch: Entzünd.hemmend. DB: Flavonoide, Flavone, Isoflavone, Isoflavonoide, Biflavone, lipophile Flavonoide)
    "flavonoids": [
        "Flavonoide", "Flavone", "Isoflavone", "Isoflavonoide", "Biflavone",
        "lipophile Flavonoide", "Flavolignane (Silymarin-Komplex)"
    ],

    # GERBSTOFFE (Buch: Adstringierend. DB: Gerbstoffe, Ellagitannine, Catechingerbstoffe, komplexe Gerbstoffe, Flavanoellagitannine, Lamiaceen-Gerbstoffe, Gallotannine)
    "tannins": [
        "Gerbstoffe", "Ellagitannine", "Catechingerbstoffe", "komplexe Gerbstoffe",
        "Flavanoellagitannine", "Lamiaceen-Gerbstoffe", "Gallotannine"
    ],

    # LIGNANE (Buch: Phytoöstrogene. DB: Lignane)
    "lignans": [
        "Lignane"
    ],

    # SAPONINE (Buch: Husten etc. DB: Triterpensaponine, Steroidsaponine, Hederacoside, Aescine, Glycyrrhizin, Ruscogenine, Virgaurea-Saponine, Triterpenglykoside)
    "saponins": [
        "Triterpensaponine", "Steroidsaponine", "Hederacoside", "Aescine",
        "Glycyrrhizin", "Ruscogenine", "Virgaurea-Saponine", "Triterpenglykoside"
    ],

    # SCHLEIMSTOFFE (Buch: Schleimhautschutz. DB: Schleimstoffe, Polysaccharide, Kohlenhydrate, Inulin, Saccharide)
    "mucilages": [
        "Schleimstoffe", "Polysaccharide", "Kohlenhydrate", "Inulin", "Saccharide"
    ],

    # ÄTHERISCHE ÖLE (Buch-Kontext: Desinfizierend. DB: ätherisches Öl, Eukalyptusöl, Bitterfenchelöl, Kümmelöl, ätherisches Öl (Pfefferminzöl), ätherisches Öl (Rosmarinöl), ätherisches Öl (Thymianöl) + Komponenten)
    "essential_oils": [
        "ätherisches Öl", "Eukalyptusöl", "Bitterfenchelöl", "Kümmelöl",
        "ätherisches Öl (Pfefferminzöl)", "ätherisches Öl (Rosmarinöl)", "ätherisches Öl (Thymianöl)",
        "1,8-Cineol", "Euglobale", "Macrocarpale", "trans-Anethol", "Fenchon", "Estragol",
        "D-Carvon", "Limonen", "Monoterpene", "Linalylacetat", "Linalool", "Menthol",
        "Menthon", "Menthylacetat", "Terpene", "Pulegon", "Menthofuran", "Thujon",
        "Campher", "Thymol", "p-Cymen", "γ-Terpinen", "Citral", "Citronellal",
        "Sesquiterpene", "Bisabolol", "Bisabololoxide", "Chamazulen", "Matrizin"
    ],

    # BITTERSTOFFE (Buch-Kontext: Verdauung. DB: Bitterstoffe, Bitterstoffe vom Secoiridoidtyp, Sesquiterpen-Bitterstoffe, Sesquiterpenlacton-Bitterstoffe, Sesquiterpenlactone, Gentisin, Gentianose, Diterpenlactone, Hopfenharz, bittere Phloroglucinderivate (Hopfenbitterstoffe), bittere Diterpenphenole, Helenaline, Dihydrohelenaline, Guajanolide, Iridoide, Iridoidglykoside)
    "bitters": [
        "Bitterstoffe", "Bitterstoffe vom Secoiridoidtyp", "Sesquiterpen-Bitterstoffe",
        "Sesquiterpenlacton-Bitterstoffe", "Sesquiterpenlactone", "Gentisin", "Gentianose",
        "Diterpenlactone", "Hopfenharz", "bittere Phloroglucinderivate (Hopfenbitterstoffe)",
        "bittere Diterpenphenole", "Helenaline", "Dihydrohelenaline", "Guajanolide",
        "Iridoide", "Iridoidglykoside"
    ]
    # SONSTIGE (nicht buch-spezifisch gruppiert: z.B. Arbutin, Alkaloide, Scharfstoffe etc. → bei Bedarf erweitern)
}

# 2. SYMPTOM_TO_GROUP_RULES (Buch-dominiert, unverändert)
SYMPTOM_TO_GROUP_RULES = {
    "verstopfung": ["anthranoids", "mucilages"],
    "durchfall": ["tannins"],
    "magen": ["mucilages", "bitters"],
    "bauch": ["bitters", "essential_oils"],
    "krampf": ["essential_oils"],
    "blähung": ["essential_oils", "bitters"],
    "husten": ["saponins", "essential_oils", "mucilages"],
    "reizhusten": ["mucilages"],
    "hals": ["mucilages", "tannins"],
    "schleim": ["saponins", "essential_oils"],
    "wunde": ["tannins", "flavonoids"],
    "entzündung": ["flavonoids", "tannins"],
    "haut": ["tannins", "flavonoids", "mucilages"],
    "immun": ["lignans", "saponins"],
    "herz": ["flavonoids"],
    "ödeme": ["coumarins", "flavonoids"],
    "blutzucker": ["coumarins"]
}

# Reale Symptom-Cluster für die Trainingsdaten-Erzeugung
SYMPTOM_CLUSTERS = {
    "Erkältung": {
        "keywords": ["husten", "schnupfen", "fieber", "hals", "rachen", "schleim", "bronchitis"],
        "probability": 0.35
    },
    "Magen-Darm": {
        "keywords": ["magen", "bauch", "verdauung", "blähung", "krampf", "übel", "durchfall", "verstopfung"],
        "probability": 0.25
    },
    "Psyche & Schlaf": {
        "keywords": ["schlaf", "unruh", "nervös", "angst", "stress", "nerven"],
        "probability": 0.20
    },
    "Haut & Wunden": {
        "keywords": ["haut", "wunde", "entzündung", "ekzem", "juck"],
        "probability": 0.10
    },
    "Schmerz": {
        "keywords": ["kopf", "muskel", "gelenk", "rheuma", "schmerz", "migräne"],
        "probability": 0.10
    }
}


# XAI-Erklärungen (Buch-basiert, unverändert)
GROUP_EXPLANATIONS = {
    "anthranoids": "Abführend: Reduziert Resorption von Na/Wasser im Darm → weicher Stuhl (8-12h).",
    "coumarins": "Gerinnungshemmend, entzündungshemmend, ödemreduzierend.",
    "flavonoids": "Entzündungshemmend, gefäßschützend, antioxidativ (Anthocyane).",
    "tannins": "Adstringierend: Denaturiert Proteine → austrocknend, blutstillend, Durchfall-hemmend.",
    "lignans": "Phytoöstrogene: Immunstärkend, krebspräventiv.",
    "saponins": "Schäumen, senken Oberflächenspannung: Auswurffördernd bei Husten, keimtötend.",
    "mucilages": "Schleimhautschützend: Lindert Reiz, reguliert Stuhl.",
    "essential_oils": "Desinfizierend, schleimlösend, krampflösend.",
    "bitters": "Verdauungsfördernd, appetitanregend."
}

# XAI-Erklärungen für Laien (Endnutzer-UI)
GROUP_LAYPERSON_EXPLANATIONS = {
    'essential_oils': 'Sie enthalten **ätherische Öle**, die desinfizierend wirken und helfen können, Schleim zu lösen und Krämpfe zu lindern.',
    'saponins': 'Sie enthält **Saponine** – natürliche Stoffe, die wie ein Schaummittel wirken und das Abhusten von Schleim erleichtern.',
    'flavonoids': 'Sie ist reich an **Flavonoiden**, die entzündungshemmend und gefäßschützend wirken – ähnlich wie natürliche Antioxidantien.',
    'mucilages': 'Sie enthält **Schleimstoffe**, die sich wie ein Schutzfilm über gereizte Schleimhäute legen und so den Reiz lindern.',
    'tannins': 'Sie enthält **Gerbstoffe**, die eine zusammenziehende Wirkung haben. Das hilft bei Entzündungen und kann Durchfall stoppen.',
    'bitters': 'Sie enthält **Bitterstoffe**, die die Verdauung anregen und den Appetit fördern, indem sie die Produktion von Magensaft stimulieren.',
    'anthranoids': 'Sie wirkt abführend, indem sie die Wasseraufnahme im Darm reduziert.',
    'coumarins': 'Sie enthält **Cumarine**, die gerinnungshemmend und entzündungshemmend wirken.',
    'lignans': 'Sie enthält **Lignane**, die das Immunsystem stärken können.',
}

def get_boosted_chemicals(user_text: str) -> list:
    """
    User-Symptome -> boosted DB-Spalten (chem_{name}).
    """
    text_lower = user_text.lower()
    chemicals_to_boost = []
    active_groups = set()

    for keyword, groups in SYMPTOM_TO_GROUP_RULES.items():
        if keyword in text_lower:
            for g in groups:
                active_groups.add(g)

    for group_id in active_groups:
        substances = PHYTO_GROUPS.get(group_id, [])
        for subst in substances:
            chemicals_to_boost.append(f"chem_{subst}")

    return list(set(chemicals_to_boost))


# =====================================================================
# NLP & KEYWORD MAPPINGS (Zentralisiert)
# =====================================================================

# 1. Wichtige Symptome und ihre Synonyme (Für Feature-Injection in logic.py)
IMPORTANT_SYMPTOMS_MAPPING = {
    'husten': ['husten', 'reizhusten', 'bronchitis'],
    'hals': ['halsschmerzen', 'halsweh', 'hals', 'rachen'],
    'schnupfen': ['schnupfen', 'nase', 'erkältung'],
    'fieber': ['fieber'],
    'kopf': ['kopf', 'kopfschmerzen', 'kopfschmerz', 'kopfweh', 'migräne', 'migraene'],
    'uebelkeit': ['uebelkeit', 'übelkeit', 'nausea', 'erbrechen', 'brechreiz'],
    'magen': ['magen', 'bauch', 'magenschmerzen', 'bauchschmerzen', 'verdauung', 'voellegefuehl', 'völlegefühl', 'blaehungen', 'blähungen']
}

# 2. UI-Hauptsymptome Mapping (Für Bonus-Scoring in logic.py)
PRIMARY_SYMPTOMS_MAPPING = {
    "kopfschmerzen": ["kopf", "kopfschmerz", "spannungskopfschmerzen"],
    "husten": ["husten", "bronchitis"],
    "halsschmerzen": ["hals", "halsschmerz", "rachen"],
    "magenbeschwerden": ["magen", "bauch", "verdauung"],
    "übelkeit": ["uebelkeit", "übelkeit", "nausea", "erbrechen"]
    # weitere bei Bedarf hier zentral ergänzen
}

# 3. NLP Signal-Regeln für SymptomPreprocessor (Für genaue Differenzierung)
NLP_SIGNAL_RULES = {
    "cough_indicators": ["husten", "reizhusten"],
    "cough_dry_modifiers": ["trocken", "trockener", "trockne", "ohne schleim", "reiz"],
    "cough_mucus_modifiers": ["schleim", "verschleimt", "auswurf", "produktiv", "fest"],
    
    "throat_indicators": ["hals", "halsweh", "rachen"],
    "throat_scratchy_modifiers": ["kratz", "rauh", "trocken"],
    
    "stomach_indicators": ["magen", "bauch", "verdauung"],
    "stomach_cramps_modifiers": ["krampf", "verkrampf", "kolik"],
    "stomach_bloating_modifiers": ["blaeh", "aufgeblaeh", "voellegefuehl"],
    
    "sleep_indicators": ["schlaf", "einschlaf", "durchschlaf"],
    "anxiety_indicators": ["unruh", "nervoes", "angst", "angespan"]
}

