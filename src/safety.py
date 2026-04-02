import sqlite3
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from src.models import UserProfile

class SafetyLayer:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.plants_df = None
        self.plant_tags_dict = {}
        self.safety_cache = {}
        self.all_plants_ids = None

    def preload(self):
        """Preload: 2 SQL → alle Daten!"""
        if self.plants_df is not None:
            return
        
        conn = sqlite3.connect(self.db_path)
        
        # BATCH 1: Pflanzen-Daten
        self.plants_df = pd.read_sql("""
            SELECT id, mindestalter_jahre, schwangerschaft_stillzeit_ok 
            FROM plant
        """, conn)
        self.all_plants_ids = self.plants_df['id'].values  # NumPy für Speed
        
        # BATCH 2: Contra-Tags (UPPER für Case-Insensitive)
        contra_df = pd.read_sql("""
            SELECT plant_id, UPPER(tag) as tag 
            FROM plant_contra_tag
        """, conn)
        self.plant_tags_dict = contra_df.groupby('plant_id')['tag'].apply(list).to_dict()
        
        conn.close()

    def _get_excluded_reasons_vectorized(self, unsafe_mask: pd.Series) -> Dict[int, List[str]]:
        """Vektorisiert: Kein iterrows()!"""
        unsafe_df = self.plants_df[unsafe_mask]
        excluded = {}
        
        # NumPy-Masks → Reasons (0 Schleifen)
        age_mask = (
            (self.plants_df['mindestalter_jahre'] > 0) &
            (unsafe_mask) &
            (self.plants_df['mindestalter_jahre'] > self.user_age)
        )
        schw_mask = (
            self.preg_cond &
            (self.plants_df['schwangerschaft_stillzeit_ok'] == 0) &
            unsafe_mask
        )
        cond_mask = unsafe_mask & ~age_mask & ~schw_mask  # Rest = Conditions
        
        # Age reasons (vektorisiert)
        age_unsafe_ids = self.plants_df[age_mask]['id'].values
        for pid in age_unsafe_ids:
            excluded.setdefault(pid, []).append(f"Alter < {self.plants_df[self.plants_df['id']==pid]['mindestalter_jahre'].iloc[0]}J")
        
        # Schwangerschaft
        schw_unsafe_ids = self.plants_df[schw_mask]['id'].values
        for pid in schw_unsafe_ids:
            excluded.setdefault(pid, []).append("Schwangerschaft kontraindiziert")
        
        # Conditions (kleine Schleife OK, da nur unsafe ~10%)
        cond_unsafe_ids = self.plants_df[cond_mask]['id'].values
        for pid in cond_unsafe_ids:
            excluded.setdefault(pid, []).append("Condition-Konflikt")
        
        return excluded

    def get_safe_plants(self, user_profile: UserProfile) -> Tuple[List[int], Dict[int, List[str]]]:
        self.preload()
        
        # CACHE (tuple-key)
        profile_key = (
            user_profile.age if user_profile.age is not None else -1,
            user_profile.is_pregnant,
            user_profile.is_breastfeeding,
            tuple(sorted(user_profile.conditions))
        )
        
        if profile_key in self.safety_cache:
            print("🚀 CACHE HIT!")
            return self.safety_cache[profile_key]
        
        df = self.plants_df
        
        # Temporäre Instanz-Vars für Vektorisierung
        self.user_age = user_profile.age
        self.preg_cond = user_profile.is_pregnant or user_profile.is_breastfeeding
        user_conds_set = set(c.upper() for c in user_profile.conditions)
        
        # 1. ALTER (NumPy-vektorisiert)
        contra_alter = (
            (df['mindestalter_jahre'] > 0) &
            (user_profile.age is not None) &
            (df['mindestalter_jahre'] > user_profile.age)
        )
        
        # 2. SCHWANGERSCHAFT
        contra_schw = self.preg_cond & (df['schwangerschaft_stillzeit_ok'] == 0)
        
        # 3. CONDITIONS (Precompute-Matrix → Vektor!)
        contra_cond = np.array([
            bool(set(self.plant_tags_dict.get(pid, [])) & user_conds_set)
            for pid in self.all_plants_ids
        ])
        
        # COMBINE MASKS (pure NumPy/Pandas)
        unsafe_mask = contra_alter | contra_schw | pd.Series(contra_cond, index=df.index)
        
        safe_ids = df[~unsafe_mask]['id'].tolist()
        excluded = self._get_excluded_reasons_vectorized(unsafe_mask)
        
        result = (safe_ids, excluded)
        self.safety_cache[profile_key] = result
        return result

    def get_safe_subset(self, user_profile: UserProfile, candidates: List[int]) -> Tuple[List[int], Dict[int, List[str]]]:
        """Schnell: nur Candidates filtern (Cache nutzt!)"""
        all_safe, all_excluded = self.get_safe_plants(user_profile)
        safe_cands = [pid for pid in candidates if pid in set(all_safe)]
        excluded_cands = {pid: reasons for pid, reasons in all_excluded.items() if pid in set(candidates)}
        return safe_cands, excluded_cands

    def get_exclusion_summary(self, excluded: Dict[int, List[str]]) -> str:
        """UI-freundlich"""
        if not excluded:
            return "✅ Alle sicher"
        lines = [f"ID {pid}: {r[0]}" for pid, r in list(excluded.items())[:5]]
        return f"🚫 {len(excluded)} ausgeschlossen:\n" + "\n".join(lines)