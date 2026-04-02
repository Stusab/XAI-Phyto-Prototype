"""
Safety Layer ULTRA-OPTIMIERT (50 Pflanzen → 5ms!)
Batch-Pandas + perfekter Cache
"""
import sqlite3
import pandas as pd
from typing import List, Tuple, Dict, Optional
from src.models import UserProfile

class SafetyLayer:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.plants_df = None
        self.plant_tags_dict = {}
        self.safety_cache = {}

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
        
        # BATCH 2: Contra-Tags (UPPER für Case-Insensitive)
        contra_df = pd.read_sql("""
            SELECT plant_id, UPPER(tag) as tag 
            FROM plant_contra_tag
        """, conn)
        
        # Dict: plant_id → List[tags]
        self.plant_tags_dict = contra_df.groupby('plant_id')['tag'].apply(list).to_dict()
        
        conn.close()

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
            return self.safety_cache[profile_key]
        
        safe_ids = []
        excluded = {}
        
        df = self.plants_df.copy()
            
                # 1. ALTER (vektorisiert)
        df['contra_alter'] = (
                (df['mindestalter_jahre'] > 0) &
                (user_profile.age is not None) &
                (df['mindestalter_jahre'] > user_profile.age)
            )
            
            # 2. SCHWANGERSCHAFT (vektorisiert)
        preg_cond = user_profile.is_pregnant or user_profile.is_breastfeeding
        df['contra_schw'] = preg_cond & (df['schwangerschaft_stillzeit_ok'] == 0)
            
            # 3. CONDITIONS (broadcast)
        contra_cond = []
        for pid in df['id']:
                plant_tags = set(self.plant_tags_dict.get(pid, []))
                user_conds = set(c.upper() for c in user_profile.conditions)
                contra_cond.append(bool(user_conds & plant_tags))
        df['contra_cond'] = contra_cond
            
            # ANY Contra?
        unsafe_mask = df['contra_alter'] | df['contra_schw'] | df['contra_cond']
            
            # Ergebnisse
        safe_ids = df[~unsafe_mask]['id'].tolist()
        unsafe_df = df[unsafe_mask]
            
        excluded = {}
        for idx, row in unsafe_df.iterrows():
                pid = int(row['id'])
                reasons = []
                if row['contra_alter']:
                    reasons.append(f"Alter < {row['mindestalter_jahre']}J")
                if row['contra_schw']:
                    reasons.append("Schwangerschaft kontraindiziert")
                if row['contra_cond']:
                    reasons.append("Condition-Konflikt")
                excluded[pid] = reasons
            
        self.safety_cache[profile_key] = (safe_ids, excluded)
        return safe_ids, excluded

    def get_safe_subset(self, user_profile: UserProfile, candidates: List[int]) -> Tuple[List[int], Dict[int, List[str]]]:
        """Schnell: nur Candidates filtern (Cache nutzt!)"""
        all_safe, all_excluded = self.get_safe_plants(user_profile)
        safe_cands = [pid for pid in candidates if pid in set(all_safe)]
        excluded_cands = {pid: reasons for pid, reasons in all_excluded.items() if pid in candidates}
        return safe_cands, excluded_cands

    def get_exclusion_summary(self, excluded: Dict[int, List[str]]) -> str:
        """UI-freundlich"""
        if not excluded:
            return "✅ Alle sicher"
        lines = [f"ID {pid}: {r[0]}" for pid, r in list(excluded.items())[:5]]
        return f"🚫 {len(excluded)} ausgeschlossen:\n" + "\n".join(lines)
