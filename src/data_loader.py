import sqlite3
import pandas as pd
import numpy as np
import os
import re
import pickle
from collections import defaultdict

# --- ABSOLUTER PFAD ZUR DB ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "data", "heilpflanzen.db")


# --------------------------------------------------------------------------
# FEATURE SPARSITY REDUCTION
# --------------------------------------------------------------------------
def _parse_feature_col(col_name):
    """
    Zerlegt einen Spaltennamen in (prefix, channel, symptom_text).
    
    Beispiele:
        'sym_evidenz_basiert_Husten'           → ('sym_', 'evidenz_basiert', 'Husten')
        'sym_traditionell_hmpc_trockener Reizhusten' → ('sym_', 'traditionell_hmpc', 'trockener Reizhusten')
        'use_evidenz_basiert_produktiver Husten' → ('use_', 'evidenz_basiert', 'produktiver Husten')
        'chem_Flavonoide'                      → ('chem_', '', 'Flavonoide')
        'risk_MAGEN'                           → ('risk_', '', 'MAGEN')
    """
    # sym_ und use_ haben Channels
    for prefix in ['sym_', 'use_']:
        if col_name.startswith(prefix):
            rest = col_name[len(prefix):]
            for channel in ['evidenz_basiert_', 'traditionell_hmpc_']:
                if rest.startswith(channel):
                    text = rest[len(channel):]
                    return prefix, channel.rstrip('_'), text
            # Kein bekannter Channel
            return prefix, 'unknown', rest
    
    # linksym / linkind haben auch Channels
    for prefix in ['linksym', 'linkind']:
        if col_name.startswith(prefix):
            rest = col_name[len(prefix):]
            for channel in ['evidenzbasiert_', 'traditionellhmpc_']:
                if rest.startswith(channel):
                    text = rest[len(channel):]
                    return prefix, channel.rstrip('_'), text
            return prefix, 'unknown', rest
    
    # chem_, risk_ → kein Channel
    for prefix in ['chem_', 'risk_']:
        if col_name.startswith(prefix):
            return prefix, '', col_name[len(prefix):]
    
    return 'other_', '', col_name


def merge_similar_features(full_matrix, db_path, similarity_threshold=0.85):
    """
    Fasst semantisch ähnliche sym_/use_-Features zusammen.
    
    REGELN:
    - Nur sym_ und use_ Spalten werden gemerged
    - Nur innerhalb desselben Evidenz-Kanals (evidenz_basiert / traditionell_hmpc)
    - chem_, risk_, link* bleiben IMMER unverändert
    - Merge-Logik: OR (wenn mindestens eine Originalspalte = 1, dann Merged = 1)
    - Similarity-Threshold: 0.85 (konservativ, nur echte Synonyme)
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    
    cols = full_matrix.columns.tolist()
    
    # 1) Spalten in mergeable (sym_/use_) und protected (alles andere) aufteilen
    mergeable_cols = [c for c in cols if c.startswith(('sym_', 'use_'))]
    protected_cols = [c for c in cols if not c.startswith(('sym_', 'use_'))]
    
    if not mergeable_cols:
        print("⚠️ Keine sym_/use_ Spalten zum Mergen gefunden.")
        return full_matrix
    
    print(f"\n--- 3. Feature Sparsity Reduction ---")
    print(f"Mergeable Spalten (sym_/use_): {len(mergeable_cols)}")
    print(f"Geschützte Spalten (chem/risk/link): {len(protected_cols)}")
    
    # 2) Gruppiere nach (prefix, channel) → nur innerhalb einer Gruppe wird gemerged
    groups = defaultdict(list)
    col_texts = {}  # col_name → reiner Symptomtext
    
    for col in mergeable_cols:
        prefix, channel, text = _parse_feature_col(col)
        group_key = (prefix, channel)
        groups[group_key].append(col)
        col_texts[col] = text
    
    print(f"Gefundene Gruppen: {len(groups)}")
    for key, members in groups.items():
        print(f"  {key}: {len(members)} Features")
    
    # 3) SBERT laden (Cache-Check)
    cache_path = os.path.join(os.path.dirname(db_path), "merge_embeddings.pkl")
    
    # Alle Texte für Embedding sammeln
    all_texts = [col_texts[c] for c in mergeable_cols]
    
    if os.path.exists(cache_path):
        print(f"📦 Lade gecachte Merge-Embeddings...")
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        # Cache nur nutzen wenn die Spalten identisch sind
        if cached.get("cols") == mergeable_cols:
            embeddings = cached["embeddings"]
        else:
            print("⚠️ Cache veraltet, berechne neu...")
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            embeddings = model.encode(all_texts, batch_size=32, show_progress_bar=True)
            with open(cache_path, "wb") as f:
                pickle.dump({"cols": mergeable_cols, "embeddings": embeddings}, f)
    else:
        print(f"🧠 Berechne SBERT-Embeddings für {len(all_texts)} Feature-Texte...")
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        embeddings = model.encode(all_texts, batch_size=32, show_progress_bar=True)
        with open(cache_path, "wb") as f:
            pickle.dump({"cols": mergeable_cols, "embeddings": embeddings}, f)
    
    # Index-Mapping: col_name → embedding index
    col_to_idx = {col: i for i, col in enumerate(mergeable_cols)}
    
    # 4) Pro Gruppe: Ähnliche Features finden und zusammenfassen
    merge_map = {}  # old_col → new_col (Repräsentant)
    total_merged = 0
    
    for group_key, group_cols in groups.items():
        if len(group_cols) <= 1:
            # Nur 1 Feature in der Gruppe → nichts zu mergen
            for col in group_cols:
                merge_map[col] = col
            continue
        
        # Embeddings für diese Gruppe
        group_indices = [col_to_idx[c] for c in group_cols]
        group_embeddings = embeddings[group_indices]
        
        # Similarity-Matrix innerhalb der Gruppe
        sim_matrix = cosine_similarity(group_embeddings)
        
        # Union-Find / Greedy Clustering
        visited = set()
        clusters = []
        
        for i in range(len(group_cols)):
            if i in visited:
                continue
            cluster = [i]
            visited.add(i)
            for j in range(i + 1, len(group_cols)):
                if j in visited:
                    continue
                if sim_matrix[i][j] >= similarity_threshold:
                    cluster.append(j)
                    visited.add(j)
            clusters.append(cluster)
        
        # Für jeden Cluster: Repräsentant = die Spalte mit den meisten Pflanzen
        for cluster in clusters:
            cluster_cols = [group_cols[idx] for idx in cluster]
            # Repräsentant: Feature mit den meisten Pflanzen (höchste Summe)
            sums = {c: full_matrix[c].sum() for c in cluster_cols}
            representative = max(sums, key=sums.get)
            
            for col in cluster_cols:
                merge_map[col] = representative
            
            if len(cluster_cols) > 1:
                total_merged += len(cluster_cols) - 1
                merged_names = [col_texts[c] for c in cluster_cols if c != representative]
                print(f"  🔗 Merge: '{col_texts[representative]}' ← {merged_names}")
    
    # 5) Neue Matrix bauen
    # Für jede Merged-Gruppe: OR über alle Original-Spalten
    new_col_data = defaultdict(lambda: np.zeros(len(full_matrix), dtype=int))
    
    for old_col, new_col in merge_map.items():
        new_col_data[new_col] = np.maximum(
            new_col_data[new_col],
            full_matrix[old_col].values
        )
    
    # Merged sym_/use_ DataFrame
    merged_df = pd.DataFrame(new_col_data, index=full_matrix.index)
    
    # Protected Spalten unverändert anhängen
    protected_df = full_matrix[protected_cols]
    
    # Zusammenfügen
    result = pd.concat([merged_df, protected_df], axis=1)
    
    print(f"\n✅ Feature Sparsity Reduction abgeschlossen:")
    print(f"   Vorher: {len(cols)} Features")
    print(f"   Nachher: {len(result.columns)} Features")
    print(f"   Reduziert: {total_merged} redundante Spalten zusammengefasst")
    
    return result


def load_and_fuse_data(db_path=None):
    """
    Lädt alle Daten und erstellt eine 'Fused Feature Matrix'.
    Berücksichtigt 'channel' (Evidenzgrad) und korrekte Spaltennamen.
    Wendet am Ende semantisches Feature-Merging an (Sparsity Reduction).
    """
    # Wenn kein Pfad angegeben wird, nimm den Default (neben dem Skript)
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    data_store = {}
    
    # Check ob DB existiert
    if not os.path.exists(db_path):
        # Fallback: Vielleicht sind wir im Root und data/ ist ein Unterordner?
        if os.path.exists(os.path.join("data", "heilpflanzen.db")):
            db_path = os.path.join("data", "heilpflanzen.db")
        else:
            print(f"❌ FEHLER: Datenbank nicht gefunden unter: {db_path}")
            return None, None

    # 1. Verbindung herstellen
    try:
        conn = sqlite3.connect(db_path)
        print(f"--- 1. Lade Rohdaten aus {db_path} ---")
    except sqlite3.Error as e:
        print(f"Fehler bei DB Verbindung: {e}")
        return None, None

    # Alle Tabellen laden
    tables = [
        'plant', 'plant_symptom', 'plant_phytochemical', 
        'plant_usecase', 'plant_contra_tag', 'plant_synonym',
        'plant_drug_form', 'plant_evidence_source', 
        'plant_side_effect', 'plant_medical_check',
        'plant_preparation', 'plant_application_hint', 'symptom_indikation_link'
    ]

    for table in tables:
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
            data_store[table] = df
        except Exception:
            data_store[table] = pd.DataFrame()

    conn.close()
    
    print(f"DEBUG: symptom_indikation_link geladen? {len(data_store.get('symptom_indikation_link', pd.DataFrame()))} Zeilen")

    # --- 2. Feature Engineering (Die Matrix bauen) ---
    print("\n--- 2. Erstelle ML-Feature-Matrix (Fusion mit Evidenz-Channel) ---")
    
    feature_configs = [
        {'table': 'plant_symptom',       'col': 'symptom',      'prefix': 'sym_',  'use_channel': True},
        {'table': 'plant_phytochemical', 'col': 'inhaltsstoff', 'prefix': 'chem_', 'use_channel': False},
        {'table': 'plant_usecase',       'col': 'text',         'prefix': 'use_',  'use_channel': True},
        {'table': 'plant_contra_tag',    'col': 'tag',          'prefix': 'risk_', 'use_channel': False},
        {'table': 'symptom_indikation_link', 'col': 'symptom', 'prefix': 'linksym', 'use_channel': True},
        {'table': 'symptom_indikation_link', 'col': 'indikation', 'prefix': 'linkind', 'use_channel': True},
    ]

    matrix_parts = []
    
    # Basis: Alle Pflanzen-IDs
    if 'plant' in data_store and not data_store['plant'].empty:
        base_df = data_store['plant'][['id']].rename(columns={'id': 'plant_id'})
        base_df = base_df.set_index('plant_id')
    else:
        print("FEHLER: Tabelle 'plant' ist leer oder fehlt.")
        return data_store, None

    for config in feature_configs:
        tbl_name = config['table']
        text_col = config['col']
        prefix = config['prefix']
        use_channel = config['use_channel']
        
        df = data_store.get(tbl_name)
        
        if df is not None and not df.empty:
            if text_col not in df.columns:
                print(f"⚠ ACHTUNG: Spalte '{text_col}' in Tabelle '{tbl_name}' nicht gefunden! Überspringe...")
                print(f"   -> Vorhandene Spalten: {df.columns.tolist()}")
                continue
                
            if use_channel and 'channel' in df.columns:
                df['clean_channel'] = df['channel'].astype(str).str.replace(' ', '_')
                df['feature_name'] = df['clean_channel'] + '_' + df[text_col].astype(str)
            else:
                df['feature_name'] = df[text_col].astype(str)

            df['exists'] = 1
            pivot = df.pivot_table(index='plant_id', columns='feature_name', values='exists', fill_value=0)
            pivot.columns = [f"{prefix}{c}" for c in pivot.columns]
            
            matrix_parts.append(pivot)
            print(f"✔ Features aus '{tbl_name}': {pivot.shape[1]} Spalten geladen.")
            
        else:
            print(f"⚠ Warnung: Keine Daten für '{tbl_name}' gefunden.")

    # --- 3. Zusammenfügen ---
    if matrix_parts:
        full_matrix = base_df.join(matrix_parts, how='left')
        full_matrix = full_matrix.fillna(0).astype(int)
    else:
        full_matrix = None
        print("KRITISCHER FEHLER: Keine Features konnten erstellt werden.")
        return data_store, full_matrix

    # --- 4. Feature Sparsity Reduction (NEU) ---
    # Fasst semantisch ähnliche sym_/use_ Spalten zusammen
    # RESPEKTIERT: evidenz_basiert vs traditionell_hmpc (kein Cross-Channel Merge!)
    full_matrix = merge_similar_features(full_matrix, db_path, similarity_threshold=0.85)

    return data_store, full_matrix

if __name__ == "__main__":
    all_data, ml_matrix = load_and_fuse_data()
    
    if ml_matrix is not None:
        print("\n--- Ergebnis ---")
        print(f"Dimension: {ml_matrix.shape}")
        
        cols = list(ml_matrix.columns)
        risk_cols = [c for c in cols if 'risk_' in c]
        print(f"Anzahl geladener Risiko-Tags: {len(risk_cols)}")
        if risk_cols:
            print(f"Beispiel Risks: {risk_cols[:3]}")
