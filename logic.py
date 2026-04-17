import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import random
import os
import sys 
import shap
import joblib
import matplotlib.pyplot as plt
from src.knowledge_base import (
    get_boosted_chemicals, SYMPTOM_TO_GROUP_RULES, PHYTO_GROUPS, 
    SYMPTOM_CLUSTERS, IMPORTANT_SYMPTOMS_MAPPING, PRIMARY_SYMPTOMS_MAPPING
)

import plotly.graph_objects as go
import re  
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import defaultdict


from dotenv import load_dotenv
load_dotenv()  # Lade .env später erklärt
DB_PATH = os.getenv('DB_PATH', os.path.join(os.path.dirname(__file__), '..', 'data', 'heilpflanzen.db'))

# --- JETZT ERST IMPORTIEREN ---
from .data_loader import load_and_fuse_data  # <--- Jetzt findet er es!
from src.matching import RelevanceLayer
from src.symptom_preprocessing import SymptomPreprocessor
from src.safety import SafetyLayer
from src.models import UserProfile


# --- GLOBALE VARIABLEN ---
# Speichern den Zustand des Systems nach der Initialisierung
TRAINED_MODELS = {} # Hier liegt das trainierte Modell
ALL_DATA = {}       # Hier liegen die Rohdaten (für UI-Infos wie Namen)
ML_MATRIX = None    # Die Matrix, auf der wir gelernt haben
FEATURE_NAMES = []  # Liste aller Spaltennamen (662 Stück)
RELEVANCE_ENGINE = None
SAFETY_LAYER = None # Hier liegt die Instanz des Safety Layers

FEATURE_CANDIDATES = []        # Liste der Spaltennamen (sym_/use_), die SBERT nutzt
FEATURE_CANDIDATE_TEXTS = []   # Gecleante Texte zu diesen Spalten
FEATURE_EMBEDDINGS = None    

# --------------------------------------------------------------------------
# 1. TRAINING DATA GENERATOR (Data Augmentation)
# --------------------------------------------------------------------------
def create_synthetic_training_data(ml_matrix, n_samples_per_plant=50):
    """
    Erzeugt künstliche Trainingsdaten (mit Wahrscheinlichkeits-Clustern).
    UPDATE: Trainiert das Modell jetzt auch auf INHALTSSTOFFE (chem_),
    und verwendet Symptom-Cluster, damit der Random Forest echte 
    Kombinationen (z.B. Husten + Halsschmerzen) lernt statt Zufall.
    """
    X_synthetic = []
    y_synthetic = []
    feature_names = ml_matrix.columns.tolist()

    print(f"Generiere {n_samples_per_plant} synthetische User-Anfragen pro Pflanze (inkl. Cluster-Optimierung)...")

    for plant_id in ml_matrix.index:
        plant_profile = ml_matrix.loc[plant_id]
        active_features = plant_profile[plant_profile == 1].index.tolist()
        
        # Alle suchbaren Features dieser Pflanze
        searchable_features = [
            f for f in active_features 
            if f.startswith(('sym', 'use', 'chem', 'linksym', 'linkind'))]
        
        if not searchable_features:
            continue

        # 1) Finde heraus, welche Cluster diese Pflanze bedient
        plant_clusters = []
        for cluster_name, cluster_data in SYMPTOM_CLUSTERS.items():
            cluster_kws = cluster_data["keywords"]
            # Hat die Pflanze Features aus diesem Cluster?
            matching_features = [
                f for f in searchable_features 
                if any(kw in f.lower() for kw in cluster_kws)
            ]
            if matching_features:
                plant_clusters.append({
                    "name": cluster_name,
                    "prob": cluster_data["probability"],
                    "features": matching_features
                })

        # Generiere n Variationen
        for _ in range(n_samples_per_plant):
            sample_vector = pd.Series(0, index=feature_names)
            selected_inputs = []

            # 70% Wahrscheinlichkeit für "Intent-based" (Symptom-Cluster)
            # 30% Wahrscheinlichkeit für "Random" (Baseline/Regularisierung)
            if plant_clusters and random.random() < 0.70:
                # Wähle einen Cluster basierend auf Wahrscheinlichkeit
                probs = [c["prob"] for c in plant_clusters]
                total_prob = sum(probs)
                normalized_probs = [p / total_prob for p in probs]
                
                chosen_cluster = random.choices(plant_clusters, weights=normalized_probs, k=1)[0]
                cluster_features = chosen_cluster["features"]
                
                # Wähle 1-3 Features aus diesem Cluster
                n_inputs = random.randint(1, min(3, len(cluster_features)))
                selected_inputs = random.sample(cluster_features, n_inputs)
                
                # Mit 30% Wahrscheinlichkeit noch einen beliebigen Inhaltsstoff hinzufügen
                if random.random() < 0.30:
                    chem_features = [f for f in searchable_features if f.startswith('chem_')]
                    if chem_features:
                        selected_inputs.append(random.choice(chem_features))
            else:
                # Klassisches Random (Baseline)
                n_inputs = random.randint(1, min(3, len(searchable_features)))
                selected_inputs = random.sample(searchable_features, n_inputs)

            # Baue den Vektor
            sample_vector[selected_inputs] = 1
            
            X_synthetic.append(sample_vector.values)
            y_synthetic.append(plant_id)

    return np.array(X_synthetic), np.array(y_synthetic), feature_names


def evaluate_subtype_accuracy():
    """Cross-Validation: Subtyp-Genauigkeit (Husten → trocken/feucht)"""
    global ML_MATRIX, FEATURE_NAMES  # Globale Zugriffe sichern
    initialize_system()  # Vollständig initialisieren
    model = TRAINED_MODELS['global']
    X, y, feature_names = create_synthetic_training_data(ML_MATRIX, n_samples_per_plant=50)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(X_train, y_train)  # Fresh fit
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Gesamt-Accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred, target_names=[f"Plant_{i}" for i in model.classes_]))
    
        
    # Baum-Check
    husten_first = sum(1 for tree in model.estimators_ if 'husten' in feature_names[tree.tree_.feature[0]])
    print(f"Bäume mit Husten als 1. Split: {husten_first}/80")


# --------------------------------------------------------------------------
# 2. SYSTEM START & TRAINING
# --------------------------------------------------------------------------

def initialize_system():
    """
    Lädt Daten, trainiert RF und lädt SBERT.
    """
    global TRAINED_MODELS, ALL_DATA, ML_MATRIX, FEATURE_NAMES, RELEVANCE_ENGINE, SAFETY_LAYER
    
    print("--- Initialisiere System (HYBRID MODE) ---")
    
    # 1. Daten laden & RF Training (wie bisher)
    # ... (Dein existierender Code für load_and_fuse_data) ...
    ALL_DATA, ML_MATRIX = load_and_fuse_data()

    if ML_MATRIX is None:
        raise ValueError("Fehler: Konnte Datenmatrix nicht laden.")

    model_path = os.path.join(os.path.dirname(__file__), "..", "data", "rf_model.joblib")
    feature_path = os.path.join(os.path.dirname(__file__), "..", "data", "rf_features.joblib")

    if os.path.exists(model_path) and os.path.exists(feature_path):
        print("📦 Lade RandomForest-Modell aus Cache ...")
        global_model = joblib.load(model_path)
        FEATURE_NAMES_CACHE = joblib.load(feature_path)
        
        # === NEU: Validierung der Feature-Konsistenz (Auto-Retrain) ===
        current_matrix_cols = ML_MATRIX.columns.tolist()
        
        # Prüfen ob Anzahl oder Namen abweichen
        if len(current_matrix_cols) != len(FEATURE_NAMES_CACHE) or set(current_matrix_cols) != set(FEATURE_NAMES_CACHE):
            print("\n⚠️ WARNUNG: Datenbank-Änderung oder Re-Merging erkannt!")
            print(f"   Cache erwartete {len(FEATURE_NAMES_CACHE)} Features, DB liefert {len(current_matrix_cols)} Features.")
            print("🚀 Ignoriere Cache. Erzwinge automatisches Neu-Training...")
            
            X_train, y_train, FEATURE_NAMES = create_synthetic_training_data(ML_MATRIX, n_samples_per_plant=80)
            global_model = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
            global_model.fit(X_train, y_train)
            
            # Cache überschreiben
            joblib.dump(global_model, model_path)
            joblib.dump(FEATURE_NAMES, feature_path)
            print("💾 Neues RF-Modell & Feature-Namen als Cache gespeichert.\n")
        else:
            # Cache ist gültig
            FEATURE_NAMES = FEATURE_NAMES_CACHE
            print("✅ Feature-Matrix Cache validiert und geladen.")
    else:
        X_train, y_train, FEATURE_NAMES = create_synthetic_training_data(ML_MATRIX, n_samples_per_plant=80)
        print(f"Training startet mit {len(X_train)} generierten Beispielen...")
        global_model = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
        global_model.fit(X_train, y_train)
        joblib.dump(global_model, model_path)
        joblib.dump(FEATURE_NAMES, feature_path)
        print("💾 RF-Modell & Feature-Namen gespeichert.")

    TRAINED_MODELS['global'] = global_model
    print("✔ Globales RF-Modell bereit.")
   
    # NEU: Alle 80 Bäume analysieren - Erster Split pro Baum
    print("=== ANALYSE DER 80 BÄUME ===")
    model = TRAINED_MODELS['global']  # Dein geladenes/-trainiertes Modell
    for i in range(len(model.estimators_)):  # Sicher: len(model.estimators_) statt hardcode 80
        tree = model.estimators_[i]
        if tree.tree_.feature[0] != -2:  # -2 = Leaf-Only, kein Split
            first_feature_idx = tree.tree_.feature[0]
            first_split = f"{FEATURE_NAMES[first_feature_idx]} <= {tree.tree_.threshold[0]:.2f}"
            print(f"Baum {i}: Erster Split: {first_split}")
        else:
            print(f"Baum {i}: Kein Split (Leaf-Only)")

    # Optional: Ersten Baum plotten (benötigt: from sklearn.tree import plot_tree; import matplotlib.pyplot as plt)
    from sklearn.tree import plot_tree
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20,10))
    plot_tree(model.estimators_[0], feature_names=FEATURE_NAMES, max_depth=3, filled=True)
    plt.savefig('erster_baum.png')
    print("Erster Baum als 'erster_baum.png' gespeichert.")


    # 2. NEU: Relevance Layer  & Safety Layer laden
    # Wir nehmen an, die DB liegt im 'data' Ordner relativ zum Skript
    db_path = DB_PATH
    print("🤖 Lade Relevance Layer (SBERT)...")

    RELEVANCE_ENGINE = RelevanceLayer(db_path)
    print("✔ SBERT Dolmetscher bereit.")

    # Safety Layer laden
    print("🔒 Lade Safety Layer...")
    SAFETY_LAYER = SafetyLayer(DB_PATH)
    print("Safety Layer bereit.")
    
        # 3. SBERT-Embeddings für FEATURE_NAMES (sym_/use_) einmalig vorberechnen
    global FEATURE_CANDIDATES, FEATURE_CANDIDATE_TEXTS, FEATURE_EMBEDDINGS

    FEATURE_CANDIDATES = []
    FEATURE_CANDIDATE_TEXTS = []

    for col in FEATURE_NAMES:
        if col.startswith("sym_") or col.startswith("use_"):
            clean = col.replace("sym_", "").replace("use_", "").replace("_", " ")
            FEATURE_CANDIDATES.append(col)
            FEATURE_CANDIDATE_TEXTS.append(clean)

    if FEATURE_CANDIDATE_TEXTS:
        print(f"🧠 Berechne SBERT-Embeddings für {len(FEATURE_CANDIDATE_TEXTS)} Feature-Texte (Batch)...")
        FEATURE_EMBEDDINGS = RELEVANCE_ENGINE.model.encode(
            FEATURE_CANDIDATE_TEXTS,
            batch_size=32,
            show_progress_bar=True,
        )
        print("✅ Feature-Embeddings vorbereitet.")
    else:
        FEATURE_EMBEDDINGS = None
        print("⚠️ Keine sym_/use_-Features für SBERT gefunden.")
    
# --------------------------------------------------------------------------
# 4. RECOMMENDATION ENGINE
# --------------------------------------------------------------------------
def get_recommendations(user_symptoms_text, user_profile=None, primary_symptom: str | None = None):
    """
    Hauptfunktion (MIT EXPERTEN-WISSEN):
    1. Keyword-Matching & SBERT.
    2. Logic Injection: Aktiviert Inhaltsstoffe basierend auf Regeln.
    3. Random Forest Prediction.
    """
    if 'global' not in TRAINED_MODELS or RELEVANCE_ENGINE is None:
        initialize_system()
        
    model = TRAINED_MODELS['global']
    
    # --- SCHRITT 1: Input-Vektor bauen ---
    input_vector = pd.DataFrame(0, index=[0], columns=FEATURE_NAMES)
    user_requested_features = [] # Für die Anzeige "Passt zu..."

     # === NEU: PREPROCESSING (Phase 1) ===
    preprocessor = SymptomPreprocessor()
    processed = preprocessor.preprocess(user_symptoms_text)
    
    # A) Keyword Injection (Symptome)
    # ... (Dieser Teil bleibt gleich wie vorher) ...
     # Nutze normalisierten Text + extrahierte Keywords
        # Nutze normalisierten Text + extrahierte Keywords
    user_words = set(processed.normalized_text.replace(',', ' ').split())
    user_words.update(processed.keywords)  # z.B. "reizhusten", "trocken", ...

    # NEU: Auch Originaltext (mit Umlauten) als Quelle verwenden
    orig_text = (processed.original_text or "").lower()
    orig_words = set(orig_text.replace(',', ' ').split())

    # WICHTIGE SYMPTOME + SYNONYME (aus zentraler Knowledge Base)
    important_keywords = IMPORTANT_SYMPTOMS_MAPPING

    for col in FEATURE_NAMES:
        col_lower = col.lower()

        for key, synonyms in important_keywords.items():
            # 1) Prüfen, ob dieses Symptom im User-Text vorkommt
            query_hit = (
                key in user_words
                or key in orig_words
                or any(s in user_words for s in synonyms)
                or any(s in orig_words for s in synonyms)
            )
            if not query_hit:
                continue

            # 2) Prüfen, ob dieses Symptom in diesem Feature-Namen steckt
            col_hit = (
                key in col_lower
                or any(s in col_lower for s in synonyms)
            )
            if not col_hit:
                continue

            # 3) Dann Feature aktivieren
            input_vector[col] = 1
            if col not in user_requested_features:
                user_requested_features.append(col)

        # B) SBERT Semantic Search (Symptome) – nutzt vorberechnete Feature-Embeddings
    if FEATURE_EMBEDDINGS is not None and len(FEATURE_CANDIDATES) > 0:
        # 1. User-Text in Chunks vektorisiert
        user_chunks = processed.chunks if processed.chunks else [processed.normalized_text]
        chunk_embeddings = RELEVANCE_ENGINE.model.encode(user_chunks)

        # 2. Similarity-Matrix: [n_chunks x n_features]
        sims_matrix = cosine_similarity(chunk_embeddings, FEATURE_EMBEDDINGS)

        # 3. Best-Match pro Feature (Max über Chunks)
        best_scores = sims_matrix.max(axis=0)

        # 4. Aktivieren, wenn Score über Threshold
        for i, score in enumerate(best_scores):
            if score > 0.72:  # Threshold erhöht von 0.60 auf 0.72 für höhere Präzision
                col_name = FEATURE_CANDIDATES[i]
                input_vector[col_name] = 1
                if col_name not in user_requested_features:
                    user_requested_features.append(col_name)

            
    # C) EXPERTEN-LOGIK (NEU!!!)
    # Wir holen uns die "versteckten" Wirkstoffe
    chem_boosts = get_boosted_chemicals(user_symptoms_text)
    
    for chem_col in chem_boosts:
        # Prüfen, ob diese Spalte in unserer Matrix existiert
        if chem_col in FEATURE_NAMES:
            input_vector[chem_col] = 1
            # WICHTIG: Wir fügen sie NICHT zu 'user_requested_features' hinzu,
            # wenn wir nicht wollen, dass sie oben bei "Passt zu" stehen.
            # Aber wir wollen sie vielleicht im SHAP sehen.
            # Entscheidung: Wir fügen sie hinzu, damit SHAP sie erlaubt.
            if chem_col not in user_requested_features:
                user_requested_features.append(chem_col)

    active_feats = [c for c in FEATURE_NAMES if input_vector[c].iloc[0] == 1]
    print("AKTIVE FEATURES:", [f for f in active_feats if "fieber" in f.lower() or "kopf" in f.lower() or "uebel" in f.lower()])

    # --- SCHRITT 2: Vorhersage ---
    all_probs = model.predict_proba(input_vector)[0]
    plant_classes = model.classes_


    # 1) Alle Rohscores speichern (noch KEIN Threshold)
    candidate_scores = {}  # plant_id -> score
    for i, plant_id in enumerate(plant_classes):
        score = all_probs[i]
        candidate_scores[plant_id] = score

    print(f"ML-Rohscores für {len(candidate_scores)} Pflanzen berechnet")

     # === NEU: Bonus für Hauptsymptom === 25.02.2026
    if primary_symptom:
        primary = (primary_symptom or "").lower()

        # Mapping aus zentraler Knowledge Base laden
        primary_map = PRIMARY_SYMPTOMS_MAPPING

        # passende Keyword-Liste finden
        primary_kws = []
        for key, kws in primary_map.items():
            if key in primary:
                primary_kws = kws
                break

        if primary_kws:
            # Einmalig: alle Feature-Namen, die zum Hauptsymptom gehören
            primary_features = [
                col for col in FEATURE_NAMES
                if (col.startswith("sym_") or col.startswith("use_"))
                and any(kw in col.lower() for kw in primary_kws)
            ]

            alpha = 0.05  # Bonus pro Treffer
            for plant_id in list(candidate_scores.keys()):
                try:
                    row = ML_MATRIX.loc[plant_id]
                except KeyError:
                    continue

                hits = sum(int(row.get(f, 0)) == 1 for f in primary_features)
                if hits > 0:
                    candidate_scores[plant_id] += alpha * hits

    # 2) Threshold NACH dem Bonus anwenden
    threshold = 0.05
    candidate_scores = {
        pid: score for pid, score in candidate_scores.items()
        if score >= threshold
    }

    print(f"ML + Bonus: {len(candidate_scores)} Kandidaten >= {threshold}")
    
 # ========== NEU: SAFETY LAYER FILTER ==========
    safe_candidates = list(candidate_scores.keys())  # Deine Top-Kandidaten
    excluded_reasons = {}

    if user_profile is not None and SAFETY_LAYER is not None:
        safe_profile = UserProfile(
            age=user_profile.get('age'),
            is_pregnant=user_profile.get('is_pregnant', False),
            is_breastfeeding=user_profile.get('is_breastfeeding', False),
            conditions=user_profile.get('conditions', [])
        )
        safe_ids, excluded_subset = SAFETY_LAYER.get_safe_subset(safe_profile, safe_candidates)
        excluded_reasons = excluded_subset
        print(f"Safety: {len(safe_ids)} von {len(safe_candidates)} Kandidaten sicher")
    else:
        safe_ids = safe_candidates

    # 4. Nur sichere Pflanzen detailliert verarbeiten + sortieren
    results = []
    for plant_id in safe_ids:
        score = candidate_scores[plant_id]
        try:
            plant_row = ML_MATRIX.loc[plant_id]

            # Getrennte Listen
            sym_use_matches = []
            chem_matches = []

            for feat_col in user_requested_features:
                if plant_row.get(feat_col, 0) != 1:
                    continue

                raw_name = (
                    feat_col
                    .replace("sym_", "")
                    .replace("use_", "")
                    .replace("chem_", "")
                    .replace("_", " ")
                )

                if feat_col.startswith(("sym_", "use_")):
                    if raw_name not in sym_use_matches:
                        sym_use_matches.append(raw_name)
                elif feat_col.startswith("chem_"):
                    chem_name = f"Wirkstoff {raw_name}"
                    if chem_name not in chem_matches:
                        chem_matches.append(chem_name)

            try:
                name = ALL_DATA["plant"].loc[ALL_DATA["plant"]["id"] == plant_id, "name"].iloc[0]
            except (IndexError, KeyError):
                name = f"Pflanze ID {plant_id}"

            # Für Rückwärtskompatibilität: alte Liste bleibt (Symptome/Usecases zuerst, dann Wirkstoffe)
            combined_matches = sym_use_matches + chem_matches

            results.append({
                "plant_id": plant_id,
                "name": name,
                "score": float(score),
                "input_vector": input_vector,
                "matched_features": combined_matches,
                "matched_features_sym_use": sym_use_matches,
                "matched_features_chem": chem_matches,
                "safety_excluded_reasons": excluded_reasons.get(plant_id, [])
            })
        except KeyError:
            continue

    results.sort(key=lambda x: x["score"], reverse=True)
    
    # Optional: Top-10 begrenzen
    results = results[:10]
    
    return results, excluded_reasons 




def explain_prediction_shap(plant_id, input_vector):
    """
    Berechnet SHAP-Werte und filtert sie doppelt:
    1. Nur aktive Features (die der User hat/injiziert wurden).
    2. Nur positive Beiträge (die für die Pflanze sprechen).
    """
    # 1. Init & Safety Checks
    if 'global' not in TRAINED_MODELS or 'explainer' not in TRAINED_MODELS:
        if 'global' in TRAINED_MODELS:
            TRAINED_MODELS['explainer'] = shap.TreeExplainer(TRAINED_MODELS['global'])
        else:
            return None
            
    model = TRAINED_MODELS['global']
    explainer = TRAINED_MODELS['explainer']

    try:
        class_idx = list(model.classes_).index(plant_id)
    except ValueError:
        return None

    # 2. SHAP Berechnung
    try:
        shap_values_obj = explainer.shap_values(input_vector, check_additivity=False)
    except Exception as e:
        print(f"SHAP Critical Error: {e}")
        return None

    # 3. Daten extrahieren (Robust gegen Dimensions-Chaos)
    if isinstance(shap_values_obj, list):
        raw_vals = shap_values_obj[class_idx]
    else:
        if len(shap_values_obj.shape) == 3:
            raw_vals = shap_values_obj[:, :, class_idx]
        else:
            raw_vals = shap_values_obj

    # Flatten zu 1D Array
    import numpy as np
    raw_vals = np.array(raw_vals).flatten()
    
    # Dimensions-Fix
    expected_len = len(FEATURE_NAMES)
    current_len = len(raw_vals)
    
    final_shap = raw_vals
    if current_len != expected_len:
        if current_len > expected_len and current_len % expected_len == 0:
            factor = int(current_len / expected_len)
            final_shap = raw_vals.reshape(factor, expected_len).mean(axis=0)
        else:
            return None

    # 4. DataFrame bauen
    feature_importance = pd.DataFrame({
        'feature': FEATURE_NAMES,
        'shap_value': final_shap
    })
    
    # --- FILTER 1: NUR AKTIVE INPUTS (User hat es ODER es ist ein injizierter Wirkstoff) ---
    # Wir holen nur die Spalten, wo im Input-Vektor eine 1 steht.
    # Damit fliegt "Zahnfleischentzündung" raus, weil du das nicht eingegeben hast (Wert 0).
    active_features = [col for col in input_vector.columns if input_vector[col].iloc[0] == 1]
    
    if not active_features:
        return None 

    feature_importance = feature_importance[feature_importance['feature'].isin(active_features)]
    # ---------------------------------------------------------------------------------------

        # 5. Normalisierung (auf 100 skalieren, aber Vorzeichen behalten)
    max_val = feature_importance['shap_value'].abs().max()
    if max_val > 0:
        feature_importance['shap_value'] = (feature_importance['shap_value'] / max_val) * 100

    def clean_name(s):
        s = (
            s.replace('sym_', '')
             .replace('use_', '')
             .replace('chem_', '')
             .replace('evidenz_basiert_', '')
             .replace('traditionell_', '')
             .replace('_', ' ')
        )
        return s

    feature_importance['Anzeige'] = feature_importance['feature'].apply(clean_name)

    # NEU: Top-N wichtigste aktive Features nach absolutem Einfluss
    feature_importance['abs_val'] = feature_importance['shap_value'].abs()
    feature_importance = (
        feature_importance
        .sort_values('abs_val', ascending=False)
        .head(5)   # z.B. Top 5 Gründe
    )

    # Falls wirklich alles 0 ist, lieber None zurückgeben
    if feature_importance['abs_val'].max() == 0:
        return None

    # Für Output nur Anzeige & normierte SHAP-Werte behalten
    feature_importance = feature_importance[['Anzeige', 'shap_value']]

    # Für den Plot von klein nach groß sortieren (schöne Balken)
    feature_importance.sort_values('shap_value', ascending=True, inplace=True)

    return feature_importance

import time

if __name__ == "__main__":
    print("⏱️  START Initialisierung...")
    start_total = time.time()
    
    initialize_system()
    print(f"✅ Initialisierung: {time.time() - start_total:.1f}s")
    
    print("\n⏱️  START Test 1...")
    start_test = time.time()
    recs, _ = get_recommendations("Muskelschmerzen", None)
    print(f"✅ Test 1: {time.time() - start_test:.1f}s")
    
    print(f"\n🎉 TOTAL: {time.time() - start_total:.1f}s") 
    
    evaluate_subtype_accuracy()

# --------------------------------------------------------------------------
# SHAP WATERFALL PLOT (NEU - für Bachelorarbeit)
# --------------------------------------------------------------------------

def explain_prediction_shap_waterfall(plant_id, input_vector):
    """
    SHAP Waterfall Plot: Zeigt den 'Entscheidungspfad' zur Empfehlung.
    Visualisiert, wie jedes Feature die Vorhersage schrittweise beeinflusst.
    
    Args:
        plant_id: ID der zu erklärenden Pflanze
        input_vector: Der User-Input als DataFrame (wie in get_recommendations)
        
    Returns:
        plotly Figure object für Streamlit (oder None bei Fehler)
    """
    # 1. Sicherheits-Checks: Ist das Modell geladen?
    if 'global' not in TRAINED_MODELS or 'explainer' not in TRAINED_MODELS:
        if 'global' in TRAINED_MODELS:
            TRAINED_MODELS['explainer'] = shap.TreeExplainer(TRAINED_MODELS['global'])
        else:
            return None
    
    model = TRAINED_MODELS['global']
    explainer = TRAINED_MODELS['explainer']
    
    # 2. Finde die Klassen-ID für diese Pflanze
    try:
        class_idx = list(model.classes_).index(plant_id)
    except ValueError:
        return None
    
    # 3. SHAP-Werte berechnen (genau wie in deiner bestehenden Funktion)
    try:
        shap_values_obj = explainer.shap_values(input_vector, check_additivity=False)
    except Exception as e:
        print(f"SHAP Waterfall Error: {e}")
        return None
    
    # 4. SHAP-Werte extrahieren (Multi-Class Handling)
    if isinstance(shap_values_obj, list):
        raw_vals = shap_values_obj[class_idx]
    else:
        if len(shap_values_obj.shape) == 3:
            raw_vals = shap_values_obj[:, :, class_idx]
        else:
            raw_vals = shap_values_obj
    
    # Zu 1D Array konvertieren
    raw_vals = np.array(raw_vals).flatten()
    
    # 5. Dimensions-Fix (falls Array nicht passt)
    expected_len = len(FEATURE_NAMES)
    current_len = len(raw_vals)
    final_shap = raw_vals
    
    if current_len != expected_len:
        if current_len > expected_len and current_len % expected_len == 0:
            factor = int(current_len / expected_len)
            final_shap = raw_vals.reshape(factor, expected_len).mean(axis=0)
        else:
            return None
    
    # 6. Nur aktive Features behalten (die der User eingegeben hat)
    active_features = [col for col in input_vector.columns if input_vector[col].iloc[0] == 1]
    if not active_features:
        return None
    
    # 7. DataFrame erstellen
    df = pd.DataFrame({
        'feature': FEATURE_NAMES,
        'shap_value': final_shap
    })
    df = df[df['feature'].isin(active_features)]
    
    # User Testing zeigte, dass negative SHAP-Werte bei Laien stark verwirrend sind.
    # Daher kehren wir zurück zu: Nur positive Einflussfaktoren rendern.
    # Schwellenwert auf > 0.0 gesetzt, da kleine Wahrscheinlichkeiten sonst weggeschnitten werden
    df = df[df['shap_value'] > 0.0]
    
    if df.empty:
        return None
    
    # 8. Nach Wichtigkeit sortieren (negative und positive für logischen Flow)
    df = df.sort_values('shap_value', ascending=True)
    
    # 9. Feature-Namen bereinigen (für schöne Anzeige)
    def clean_name(s):
        s = s.replace('sym_', '').replace('use_', '').replace('chem_', '')
        s = s.replace('evidenz_basiert_', '').replace('traditionell_', '').replace('traditionell_hmpc_', '')
        s = s.replace('_', ' ')
        return s.title()  # Erster Buchstabe groß
    
    df['clean_name'] = df['feature'].apply(clean_name)
    
    # 9b. Nach clean_name gruppieren (fasst sym_fieber, use_fieber etc. intelligent zusammen)
    df = df.groupby('clean_name', as_index=False)['shap_value'].sum()
    
    # User Testing zeigte, dass negative SHAP-Werte bei Laien stark verwirrend sind.
    # Daher kehren wir zurück zu: Nur positive Einflussfaktoren rendern.
    # Schwellenwert auf > 0.0 gesetzt, da kleine Wahrscheinlichkeiten sonst weggeschnitten werden
    df = df[df['shap_value'] > 0.0]
    
    if df.empty:
        return None
        
    # 8. Nach Wichtigkeit sortieren (negative und positive für logischen Flow)
    df = df.sort_values('shap_value', ascending=True)
    
    # 10. Waterfall-Logik: Kumulative Summe berechnen
    # Base Value = Durchschnitts-Wahrscheinlichkeit (Expected Value des Explainers)
    try:
        if isinstance(explainer.expected_value, (list, np.ndarray)):
            base_value = float(explainer.expected_value[class_idx])
        else:
            base_value = float(explainer.expected_value)
    except Exception:
        # Fallback falls unexpected Error
        base_value = 1.0 / len(model.classes_)
    
    # Kumulative Summe: Jeder Schritt addiert zum vorherigen
    df['cumsum'] = base_value + df['shap_value'].cumsum()
    
    # 11. Plotly Waterfall Chart erstellen
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    # Schritt A: Basis-Linie (Start)
    fig.add_trace(go.Bar(
        name='Basis',
        x=[base_value],
        y=['Basis-Wahrscheinlichkeit'],
        orientation='h',
        marker=dict(color='lightgray'),
        text=[f"{base_value:.1%}"],
        textposition='outside',
        hovertemplate='Startpunkt: %{x:.2%}<extra></extra>'
    ))
    
    # Schritt B: Alle Features als EIN Trace (verhindert mikroskopisch dünne Balken!)
    fig.add_trace(go.Bar(
        name='Einflussfaktoren',
        x=df['shap_value'].tolist(),
        y=df['clean_name'].tolist(),
        orientation='h',
        marker=dict(color='#42a5f5'),  # Blau, konsistent und fokussiert
        text=[f"+{v:.4f}" for v in df['shap_value']],
        textposition='outside',
        base=(df['cumsum'] - df['shap_value']).tolist(),
        hovertemplate="%{y}<br>Einfluss: +%{x:.4f}<extra></extra>"
    ))
    
    # Schritt C: Finale Vorhersage (Ende)
    final_pred = df['cumsum'].iloc[-1]
    fig.add_trace(go.Bar(
        name='Final',
        x=[final_pred],
        y=['Finale Empfehlung'],
        orientation='h',
        marker=dict(color='#66bb6a'),  # Grün
        text=[f"{final_pred:.1%}"],
        textposition='outside',
        hovertemplate='Finale Wahrscheinlichkeit: %{x:.2%}<extra></extra>'
    ))
    
    # 12. Layout anpassen
    fig.update_layout(
        title="📊 Entscheidungspfad: Wie kam die Empfehlung zustande?",
        xaxis_title="Einfluss auf Empfehlung",
        yaxis_title="",
        showlegend=False,
        height=400,
        barmode='overlay', # Verhindert Platzreservierung für unsichtbare Kategorien
        template='plotly_white',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def explain_wirkstoff_interactions(plant_id: int, input_vector: pd.DataFrame, user_text: str, top_k: int = 12):
    """
    Erklärt Interaktionen zwischen User-Keywords und (injizierten) Wirkstoffen:
    - Welches Keyword im Text hat welche Wirkstoff-Gruppe aktiviert?
    - Welche dieser Wirkstoffe besitzt die empfohlene Pflanze?
    - Optional: SHAP-Beitrag dieser chem_ Features für genau diese Pflanze.
    """
    # 1) Trigger im User-Text finden (Keywords -> Gruppen)
    text_lower = (user_text or "").lower()
    triggered = []  # Liste von (keyword, group_id)
    for keyword, groups in SYMPTOM_TO_GROUP_RULES.items():
        if keyword in text_lower:
            for g in groups:
                triggered.append((keyword, g))

    if not triggered:
        return pd.DataFrame(columns=["Trigger", "Gruppe", "Wirkstoff", "Pflanze_hat_Wirkstoff", "Im_Input_aktiv", "SHAP"])

    # 2) Pflanze laden (welche Features hat sie wirklich?)
    try:
        plant_row = ML_MATRIX.loc[plant_id]
    except Exception:
        return pd.DataFrame(columns=["Trigger", "Gruppe", "Wirkstoff", "Pflanze_hat_Wirkstoff", "Im_Input_aktiv", "SHAP"])

    # 3) SHAP roh berechnen (damit wir SHAP-Werte pro chem_ Feature abfragen können)
    shap_map = {}
    if 'global' in TRAINED_MODELS:
        if 'explainer' not in TRAINED_MODELS:
            TRAINED_MODELS['explainer'] = shap.TreeExplainer(TRAINED_MODELS['global'])
        model = TRAINED_MODELS['global']
        explainer = TRAINED_MODELS['explainer']

        try:
            class_idx = list(model.classes_).index(plant_id)
            shap_values_obj = explainer.shap_values(input_vector, check_additivity=False)

            if isinstance(shap_values_obj, list):
                raw_vals = shap_values_obj[class_idx]
            else:
                raw_vals = shap_values_obj[:, :, class_idx] if len(shap_values_obj.shape) == 3 else shap_values_obj

            raw_vals = np.array(raw_vals).flatten()

            expected_len = len(FEATURE_NAMES)
            if len(raw_vals) != expected_len:
                if len(raw_vals) > expected_len and len(raw_vals) % expected_len == 0:
                    factor = int(len(raw_vals) / expected_len)
                    raw_vals = raw_vals.reshape(factor, expected_len).mean(axis=0)
                else:
                    raw_vals = None

            if raw_vals is not None:
                shap_map = dict(zip(FEATURE_NAMES, raw_vals))
        except Exception:
            shap_map = {}

    # 4) Interaktions-Zeilen bauen (nur Wirkstoffe, die (a) getriggert sind und (b) in Matrix existieren)
    rows = []
    for keyword, group_id in triggered:
        substances = PHYTO_GROUPS.get(group_id, [])
        for subst in substances:
            feat = f"chem_{subst}"
            if feat not in FEATURE_NAMES:
                continue

            plant_has = int(plant_row.get(feat, 0)) == 1
            in_input = int(input_vector.get(feat, pd.Series([0])).iloc[0]) == 1
            shap_val = float(shap_map.get(feat, 0.0))

            # Wir zeigen nur Stoffe, die die Pflanze wirklich hat (sonst wird's riesig und verwirrend)
            if plant_has:
                rows.append({
                    "Trigger": keyword,
                    "Gruppe": group_id,
                    "Wirkstoff": subst,
                    "Pflanze_hat_Wirkstoff": plant_has,
                    "Im_Input_aktiv": in_input,
                    "SHAP": shap_val
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["Trigger", "Gruppe", "Wirkstoff", "Pflanze_hat_Wirkstoff", "Im_Input_aktiv", "SHAP"])

    # 5) Optional: sortieren (stärkste SHAP-Beiträge zuerst) und begrenzen
    df = df.sort_values("SHAP", ascending=False).head(top_k).reset_index(drop=True)

    return df


# --------------------------------------------------------------------------
# LAYPERSON EXPLANATION GENERATOR (NEU - für Bachelorarbeit XAI)
# --------------------------------------------------------------------------
def generate_layperson_explanation(
    plant_name: str,
    plant_id: int,
    input_vector,
    matched_features_sym_use: list,
    matched_features_chem: list,
    user_text: str,
    shap_data=None,
):
    """
    Erzeugt eine verständliche, dreistufige Erklärung für Laien.
    
    Ebene 1: Warum diese Pflanze? (Symptom-Match + Evidenz-Kanal)
    Ebene 2: Wie wirkt sie? (Wirkmechanismus in einfachen Worten)
    Ebene 3: Transparenz (Modell-Limitationen + Disclaimer)
    
    Returns:
        dict mit Keys: 'ebene1', 'ebene2', 'ebene3', 'konfidenz_level'
    """
    from src.knowledge_base import (
        SYMPTOM_TO_GROUP_RULES, PHYTO_GROUPS, GROUP_EXPLANATIONS, GROUP_LAYPERSON_EXPLANATIONS
    )
    
    # Hilfsfunktion für saubere Text-Ausgaben (entfernt alle technischen Präfixe)
    def clean_feature_name(name):
        return (
            name.replace('sym_', '')
                .replace('use_', '')
                .replace('chem_', '')
                .replace('evidenz_basiert_', '')
                .replace('traditionell_hmpc_', '')
                .replace('traditionell_', '')
                .replace('evidenz basiert ', '')
                .replace('traditionell hmpc ', '')
                .replace('Wirkstoff ', '')
                .replace('_', ' ')
        )
    
    # ================================================================
    # EBENE 1: "Warum passt diese Pflanze zu dir?"
    # ================================================================
    
    # Evidenz-Kanal aus Feature-Namen extrahieren
    evidenz_features = []
    traditionell_features = []
    
    # Aktive Feature-Spalten im Input-Vektor finden
    active_cols = [col for col in input_vector.columns if input_vector[col].iloc[0] == 1]
    
    for col in active_cols:
        if col.startswith(('sym_', 'use_')):
            if 'evidenz_basiert' in col:
                evidenz_features.append(clean_feature_name(col))
            elif 'traditionell_hmpc' in col:
                traditionell_features.append(clean_feature_name(col))
    
    # Konfidenz bestimmen
    if evidenz_features and not traditionell_features:
        konfidenz = "hoch"
        konfidenz_emoji = ""
        evidenz_text = (
            f"**{plant_name}** wird in der wissenschaftlichen Fachliteratur "
            f"**vollständig geprüft und nachgewiesen** (durch klinische Studien belegt) bei deinen Beschwerden eingesetzt."
        )
    elif evidenz_features and traditionell_features:
        konfidenz = "mittel"
        konfidenz_emoji = ""
        evidenz_text = (
            f"**{plant_name}** wird bei deinen Beschwerden eingesetzt – die Wirksamkeit "
            f"ist **vollständig geprüft und nachgewiesen** (evidenzbasierte klinische Studien & traditionelle Anwendung)."
        )
    elif traditionell_features:
        konfidenz = "traditionell"
        konfidenz_emoji = ""
        evidenz_text = (
            f"**{plant_name}** wird **traditionell** bei deinen Beschwerden angewendet. "
            f"Das bedeutet: Die Wirksamkeit ist durch eine über 30 Jahre dokumentierte, medizinische Anwendungserfahrung (davon mind. 15 Jahre in der EU) plausibel belegt. Die unbedenkliche Sicherheit ist durch bibliografische Daten garantiert, weshalb für diese Einstufung keine zusätzlichen klinischen Studien zwingend nötig sind."
        )
    else:
        konfidenz = "indirekt"
        konfidenz_emoji = ""
        evidenz_text = (
            f"**{plant_name}** wurde aufgrund ihrer Inhaltsstoffe als passend erkannt."
        )
    
    # Symptom-Match Text
    if matched_features_sym_use:
        symptom_list = ", ".join(f"**{clean_feature_name(s)}**" for s in matched_features_sym_use[:4])
        match_text = f"Du hast Beschwerden angegeben, die zu folgenden Einsatzgebieten passen: {symptom_list}."
    else:
        match_text = ""
    
    ebene1 = f"{match_text}\n\n{evidenz_text}"
    
    # ================================================================
    # EBENE 2: "Wie wirkt diese Pflanze?"
    # ================================================================
    
    # Finde aktive Wirkstoffgruppen über den User-Text
    text_lower = (user_text or "").lower()
    active_groups = set()
    for keyword, groups in SYMPTOM_TO_GROUP_RULES.items():
        if keyword in text_lower:
            for g in groups:
                active_groups.add(g)
    
    # Gruppen-Erklärungen für Laien entfallen hier (jetzt zentral in GROUP_LAYPERSON_EXPLANATIONS)
    
    ebene2_parts = []
    
    if matched_features_chem:
        chem_list = ", ".join(f"**{clean_feature_name(c)}**" for c in matched_features_chem[:3])
        ebene2_parts.append(f"Diese Pflanze enthält unter anderem die Wirkstoffe {chem_list}.")
    
    for group_id in active_groups:
        if group_id in GROUP_LAYPERSON_EXPLANATIONS:
            ebene2_parts.append(GROUP_LAYPERSON_EXPLANATIONS[group_id])
    
    if not ebene2_parts:
        ebene2 = "Für diese Pflanze konnten keine spezifischen Wirkmechanismen zu deinen Symptomen zugeordnet werden."
    else:
        ebene2 = " ".join(ebene2_parts)
    
    # ================================================================
    # EBENE 3: Transparenz & SHAP-Einordnung
    # ================================================================
    
    # SHAP-basierte Einordnung
    shap_text = ""
    if shap_data is not None and not shap_data.empty:
        top_feature = shap_data.sort_values('shap_value', ascending=False).iloc[0]
        feature_name = top_feature['Anzeige']
        shap_text = (
            f"Der stärkste Einflussfaktor für diese Empfehlung war **{feature_name}**. "
            f"Das bedeutet: Dieses Merkmal hat im KI-Modell am meisten dazu beigetragen, "
            f"dass diese Pflanze für dich vorgeschlagen wurde."
        )
    
    disclaimer = (
        "**Wichtig:** Diese Empfehlung basiert auf einem KI-Modell (Random Forest + SBERT), "
        "das Muster in einer phytotherapeutischen Datenbank erkennt. Die angezeigten Einflussfaktoren "
        "(SHAP-Werte) zeigen, *wie das Modell entschieden hat* – sie sagen nicht aus, "
        "wie *medizinisch wirksam* eine Pflanze in der Realität ist. "
        "Bei anhaltenden oder schweren Beschwerden konsultiere bitte einen Arzt oder Apotheker."
    )
    
    ebene3 = f"{shap_text}\n\n{disclaimer}" if shap_text else disclaimer
    
    ebene3 = f"{shap_text}\n\n{disclaimer}" if shap_text else disclaimer
    
    return {
        'ebene1': ebene1,
        'ebene2': ebene2,
        'ebene3': ebene3,
        'konfidenz_level': konfidenz,
        'konfidenz_emoji': konfidenz_emoji,
    }


def explain_prediction_lime(plant_id: int, input_vector: pd.DataFrame):
    """
    Berechnet eine lokale LIME-Erklärung für die Empfehlung.
    """
    import numpy as np
    import lime
    import lime.lime_tabular
    
    global TRAINED_MODELS, FEATURE_NAMES, ML_MATRIX
    if 'global' not in TRAINED_MODELS or ML_MATRIX is None:
        return None
        
    model = TRAINED_MODELS['global']
    
    # Baue X_train für den Explainer aus der Datenbank
    X_train = ML_MATRIX[FEATURE_NAMES].values
    
    # Feature-Namen für die Anzeige bereinigen
    def clean_name_lime(s):
        s = s.replace('sym_', '').replace('use_', '').replace('chem_', '')
        s = s.replace('evidenz_basiert_', '').replace('traditionell_hmpc_', '').replace('traditionell_', '')
        s = s.replace('_', ' ')
        return s.title()
        
    cleaned_feature_names = [clean_name_lime(f) for f in FEATURE_NAMES]
    
    # Definiere Explainer
    explainer_lime = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=cleaned_feature_names,
        class_names=model.classes_,
        mode='classification'
    )
    
    # Die Instanz zum Erklären (Nutzer-Input)
    instance = input_vector.values.flatten()
    
    # Erkläre die Instanz - hole alle Features, um filtern zu können
    try:
        exp = explainer_lime.explain_instance(
            data_row=instance,
            predict_fn=model.predict_proba,
            num_features=len(FEATURE_NAMES),
            top_labels=1
        )
        
        # Hole die Wichtigkeiten für die am höchsten bewertete Klasse
        label = exp.available_labels()[0]
        feature_weights = exp.local_exp[label]
        
        # Filtern: Nur Features anzeigen, die beim User auch aktiv sind (Wert > 0)
        # und die einen positiven Beitrag leisten (optional, aber User fand negative verwirrend)
        active_weights = []
        for idx, weight in feature_weights:
            if instance[idx] > 0:
                active_weights.append({
                    'feature': cleaned_feature_names[idx],
                    'weight': weight
                })
        
        if not active_weights:
            return None
            
        import pandas as pd
        df = pd.DataFrame(active_weights)
        df = df.sort_values('weight', ascending=True) # Kleinste zuerst für Plotly
        
        # Erstelle ein Plotly Bar Chart analog zu SHAP, aber simpler
        import plotly.graph_objects as go
        
        fig = go.Figure(go.Bar(
            x=df['weight'],
            y=df['feature'],
            orientation='h',
            marker=dict(
                color=['#66bb6a' if w > 0 else '#ef5350' for w in df['weight']]
            ),
            text=[f"{w:+.3f}" for w in df['weight']],
            textposition='outside',
            hovertemplate="%{y}<br>Einfluss (LIME): %{x:+.3f}<extra></extra>"
        ))
        
        fig.update_layout(
            title="🔬 Lokale Feature Wichtigkeit (LIME Approximation)",
            xaxis_title="Einfluss auf die Empfehlung (auf lokaler Ebene)",
            yaxis_title="",
            height=max(300, len(df) * 40), # Dynamische Höhe
            template='plotly_white',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
        
    except Exception as e:
        import traceback
        print(f"LIME Error: {e}\n{traceback.format_exc()}")
        return None


def get_plant_preparations(plant_id: int):
    """
    Lädt Vorbereitungen und Anwendungs-Hinweise aus der Datenbank.
    """
    import sqlite3
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # Vorbereitungen
        prep_query = f"SELECT typ, anweisung FROM plant_preparation WHERE plant_id = {plant_id}"
        prep_df = pd.read_sql(prep_query, conn)
        
        # Hinweise
        hint_query = f"SELECT hint FROM plant_application_hint WHERE plant_id = {plant_id}"
        hint_df = pd.read_sql(hint_query, conn)
        
        conn.close()
        
        return prep_df, hint_df['hint'].tolist() if not hint_df.empty else []
    except Exception as e:
        print(f"Fetch Prep Error: {e}")
        return pd.DataFrame(), []

def get_plant_medical_checks(plant_id: int):
    """
    Lädt harte Trigger (Gründe für Arztbesuch) aus plant_medical_check.
    """
    import sqlite3
    try:
        conn = sqlite3.connect(DB_PATH)
        query = f"SELECT reason FROM plant_medical_check WHERE plant_id = {plant_id}"
        df = pd.read_sql(query, conn)
        conn.close()
        return df['reason'].tolist() if not df.empty else []
    except Exception as e:
        print(f"Fetch Medical Check Error: {e}")
        return []

def get_plant_side_effects(plant_id: int):
    """
    Lädt Nebenwirkungen (Risikokommunikation) aus plant_side_effect.
    """
    import sqlite3
    try:
        conn = sqlite3.connect(DB_PATH)
        query = f"SELECT side_effect FROM plant_side_effect WHERE plant_id = {plant_id}"
        df = pd.read_sql(query, conn)
        conn.close()
        return df['side_effect'].tolist() if not df.empty else []
    except Exception as e:
        print(f"Fetch Side Effect Error: {e}")
        return []

def get_plant_safety_data(plant_ids: list) -> dict:
    """
    Gibt ein Mapping für die übergebenen IDs zurück (für Safety-Layer-Transparenz und Details).
    """
    if not plant_ids:
        return {}
    import sqlite3
    import pandas as pd
    import os
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, "..", "data", "heilpflanzen.db")
    if not os.path.exists(db_path):
        db_path = os.path.join(current_dir, "data", "heilpflanzen.db")
        
    try:
        conn = sqlite3.connect(db_path)
        ids_str = ",".join(map(str, plant_ids))
        query = f"""
            SELECT id, name, schwangerschaft_stillzeit_text, 
                   kontraindikation_text, wechselwirkungen_details, nebenwirkungen_hinweis_text,
                   dosierung_text
            FROM plant 
            WHERE id IN ({ids_str})
        """
        df = pd.read_sql(query, conn)
        conn.close()
        
        res = {}
        for _, row in df.iterrows():
            res[row['id']] = {
                'name': row['name'],
                'safety_text': row['schwangerschaft_stillzeit_text'],
                'contra_text': row['kontraindikation_text'],
                'interaction_text': row['wechselwirkungen_details'],
                'side_effect_note': row['nebenwirkungen_hinweis_text'],
                'dosage_text': row['dosierung_text']
            }
        return res
    except Exception as e:
        print(f"Fetch Safety Data Error: {e}")
        return {}

def get_plant_medical_details(plant_id: int) -> dict:
    """
    Gibt alle medizinischen Detailtexte für eine spezifische Pflanze zurück.
    """
    data = get_plant_safety_data([plant_id])
    return data.get(plant_id, {})
