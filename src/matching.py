"""
Relevance Layer: Stellt das SBERT-Modell bereit für semantische Suche.
Wird von logic.py verwendet für Feature-Embedding-Vergleiche.

HINWEIS: Die alte get_recommendations()-Methode wurde entfernt, da die
Recommendation-Logik vollständig in logic.py (Random Forest + SHAP) liegt.
Die SBERT-Encode-Funktion wird dort direkt über RELEVANCE_ENGINE.model.encode() genutzt.
"""
import os
from sentence_transformers import SentenceTransformer


class RelevanceLayer:
    def __init__(self, db_path: str):
        self.db_path = db_path
        
        # Lokales KI-Modell laden (Open Source, multilingual)
        print("🤖 Lade Semantic Model (kann beim 1. Mal dauern)...")
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print("✅ Relevance Layer bereit.")
