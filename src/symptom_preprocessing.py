from dataclasses import dataclass
from typing import Dict, List
import re
from src.knowledge_base import NLP_SIGNAL_RULES

from dataclasses import dataclass

@dataclass
class PreprocessResult:
    original_text: str
    normalized_text: str
    chunks: List[str]
    signals: Dict[str, bool]   # z.B. {"dry_cough": True}
    keywords: List[str]        # zusätzliche Tokens für Keyword-Injection


class SymptomPreprocessor:
    """
    Phase 1: Robuste Normalisierung + einfache Signal-Extraktion.
    (Noch keine großen Logikänderungen – nur Vorbereitung.)
    """

    def preprocess(self, text: str) -> PreprocessResult:
        original = text or ""
        norm = self._normalize(original)

        # Chunks ähnlich wie eure bestehende Logik (Punkt/Komma), aber auf normalisiertem Text
        raw_chunks = re.split(r"[.,;:\n]", norm)
        chunks = [c.strip() for c in raw_chunks if len(c.strip()) > 3]
        if not chunks:
            chunks = [norm.strip()] if norm.strip() else []

        signals = self._extract_signals(norm)
        extra_keywords = self._keywords_from_signals(signals)

        return PreprocessResult(
            original_text=original,
            normalized_text=norm,
            chunks=chunks,
            signals=signals,
            keywords=extra_keywords
        )

    def _normalize(self, text: str) -> str:
        t = text.lower().strip()

        # Umlaute/ß vereinheitlichen (hilft bei DB-Strings/Varianten)
        t = (
            t.replace("ä", "ae")
             .replace("ö", "oe")
             .replace("ü", "ue")
             .replace("ß", "ss")
        )

        # Satzzeichen vereinheitlichen, mehrfach spaces entfernen
        t = re.sub(r"[(){}\[\]\"']", " ", t)
        t = re.sub(r"[^a-z0-9\s,.;:\n-]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def _extract_signals(self, norm_text: str) -> Dict[str, bool]:
        """
        Erweiterte Signal-Extraktion für häufige Symptom-Differenzierungen.
        Benutzt die zentralen NLP_SIGNAL_RULES aus knowledge_base.py.
        """
        rules = NLP_SIGNAL_RULES
        
        # === HUSTEN ===
        has_husten = any(w in norm_text for w in rules["cough_indicators"])
        is_dry = any(w in norm_text for w in rules["cough_dry_modifiers"])
        has_mucus = any(w in norm_text for w in rules["cough_mucus_modifiers"])
        
        # === HALS ===
        has_throat = any(w in norm_text for w in rules["throat_indicators"])
        throat_scratchy = any(w in norm_text for w in rules["throat_scratchy_modifiers"])
        
        # === MAGEN/VERDAUUNG ===
        has_stomach = any(w in norm_text for w in rules["stomach_indicators"])
        has_cramps = any(w in norm_text for w in rules["stomach_cramps_modifiers"])
        has_bloating = any(w in norm_text for w in rules["stomach_bloating_modifiers"])
        
        # === SCHLAF/NERVEN ===
        has_sleep_issues = any(w in norm_text for w in rules["sleep_indicators"])
        has_anxiety = any(w in norm_text for w in rules["anxiety_indicators"])
        
        return {
            # Husten-Differenzierung
            "cough": has_husten,
            "dry_cough": has_husten and is_dry and not has_mucus,
            "productive_cough": has_husten and has_mucus,
            
            # Hals-Differenzierung
            "sore_throat": has_throat,
            "scratchy_throat": has_throat and throat_scratchy,
            
            # Verdauung
            "stomach_issues": has_stomach,
            "stomach_cramps": has_stomach and has_cramps,
            "bloating": has_bloating,
            
            # Psyche
            "sleep_problems": has_sleep_issues,
            "anxiety": has_anxiety,
        }

    def _keywords_from_signals(self, signals: Dict[str, bool]) -> List[str]:
        """
        Generiert spezifische Keywords basierend auf erkannten Signalen.
        Diese werden zu user_words hinzugefügt für besseres Keyword-Matching.
        """
        kws = []
        
        # Husten
        if signals.get("dry_cough"):
            kws += ["reizhusten", "trockener husten", "husten"]
        elif signals.get("productive_cough"):
            kws += ["verschleimt", "schleim", "husten", "auswurf"]
        elif signals.get("cough"):
            kws += ["husten"]
        
        # Hals
        if signals.get("scratchy_throat"):
            kws += ["halskratzen", "halsschmerzen", "rachen"]
        elif signals.get("sore_throat"):
            kws += ["halsschmerzen", "hals"]
        
        # Verdauung
        if signals.get("stomach_cramps"):
            kws += ["magenkraempfe", "krampf", "magen"]
        elif signals.get("bloating"):
            kws += ["blaehung", "voellegefuehl"]
        elif signals.get("stomach_issues"):
            kws += ["magen", "bauch"]
        
        # Psyche
        if signals.get("sleep_problems"):
            kws += ["schlafprobleme", "einschlafprobleme", "schlaf"]
        if signals.get("anxiety"):
            kws += ["unruhe", "nervositaet", "angst"]
        
        return list(dict.fromkeys(kws))  # Remove duplicates, keep order
