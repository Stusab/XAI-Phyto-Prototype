import sys
import os

# Pfad Setup
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import streamlit as st
import pandas as pd
import plotly.express as px
from src.symptom_catalog import get_main_symptoms  # NEU


# --- IMPORTS ---
try:
    # HIER IST DER FIX: Wir importieren load_and_fuse_data und nennen es load_data
    from src.data_loader import load_and_fuse_data
    from src.models import UserProfile
    from src.logic import (
    get_recommendations, 
    explain_prediction_shap, 
    explain_prediction_shap_waterfall,
    explain_wirkstoff_interactions,
    generate_layperson_explanation,
    explain_prediction_lime,
    get_plant_preparations
    )
    from src.knowledge_base import GROUP_EXPLANATIONS
except ImportError as e:
    st.error(f"Kritischer Import-Fehler: {e}")
    st.stop()

# Page Config
st.set_page_config(
    page_title="PhytoMatch AI",
    page_icon="🪴",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    div[data-testid="stExpander"] {
        background-color: #f8f9fa;
        border-radius: 10px;
        border: 1px solid #e9ecef;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.2rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Daten laden
@st.cache_resource
def get_data():
    return load_and_fuse_data()

try:
    data = get_data()
except Exception as e:
    st.error(f"Fehler beim Laden der Datenbank: {e}")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.header("👤 Dein Profil")
    age = st.number_input("Alter", min_value=0, max_value=120, value=30)
    st.markdown("---")
    is_pregnant = st.checkbox("Schwangerschaft")
    is_breastfeeding = st.checkbox("Stillzeit")
    
    st.markdown("---")
    st.subheader("⚠️ Vorerkrankungen")
    conditions_list = [
        "Allergien & Immunsystem", "Herz & Kreislauf", "Verdauungssystem", 
        "Stoffwechsel & Hormone", "Niere, Leber & Ausscheidung", "Atemwege",
        "Haut & Schleimhäute", "Sinnesorgane", "Nervensystem & Psychiatrie",
        "Onkologie", "Infektionen & Fieber"
    ]
    selected_conditions = []
    for cat in conditions_list:
        if st.checkbox(cat, key=cat):
            selected_conditions.append(cat)

    
   
# --- MAIN PAGE ---
c1, c2 = st.columns([1, 6])
with c1:
    st.image("assets/logo.png", width=80)
with c2:
    st.title("PhytoMatch AI")

st.markdown("**Dein evidenzbasierter Pflanzen-Assistent.**")

# === SYMPTOM-EINGABE: 3-STUFEN-MODELL ===
st.subheader("🩺 Beschreibe deine Beschwerden")

from src.symptom_catalog import get_main_symptoms, get_symptom_subtypes

@st.cache_data
def load_symptoms():
    return get_main_symptoms()

MAIN_SYMPTOMS = load_symptoms()

# Initialize Session State
if "main_symptom" not in st.session_state:
    st.session_state["main_symptom"] = ""
if "symptom_detail" not in st.session_state:
    st.session_state["symptom_detail"] = ""
if "additional_symptoms" not in st.session_state:
    st.session_state["additional_symptoms"] = ""

# ===== STUFE 1: HAUPTSYMPTOM =====
st.markdown("### 1️⃣ Hauptbeschwerde")
st.markdown("**💡 Häufig gesucht:**")

quick_picks = ["Husten", "Magenbeschwerden", "Unruhe", "Halsschmerzen", "Kopfschmerzen"]
cols = st.columns(5)
for i, sym in enumerate(quick_picks):
    with cols[i]:
        is_active = st.session_state.get("main_symptom") == sym
        btn_type = "primary" if is_active else "secondary"
        
        if st.button(sym, key=f"qp_{sym}", use_container_width=True, type=btn_type):
            st.session_state["main_symptom"] = sym
            st.rerun()

# Sichere Index-Berechnung
current_symptom = st.session_state.get("main_symptom", "")
if current_symptom and current_symptom in MAIN_SYMPTOMS:
    safe_index = MAIN_SYMPTOMS.index(current_symptom) + 1  # +1 wegen "Bitte wählen..."
else:
    safe_index = 0

main_symptom = st.selectbox(
    "Oder wähle aus der Liste:",
    options=["Bitte wählen..."] + MAIN_SYMPTOMS,
    index=safe_index,

    key="main_selector"
)

if main_symptom != "Bitte wählen...":
    st.session_state["main_symptom"] = main_symptom

st.markdown("---")

# ===== STUFE 2: DETAILLIERUNG (nur wenn Hauptsymptom gewählt) =====
if st.session_state["main_symptom"] and st.session_state["main_symptom"] != "Bitte wählen...":
    subtypes = get_symptom_subtypes(st.session_state["main_symptom"])
    
    if subtypes:
        st.markdown("### 2️⃣ Genauere Beschreibung")
        st.caption(f"Wie würdest du dein **{st.session_state['main_symptom']}** beschreiben?")
        
        detail_choice = st.radio(
            "Art des Symptoms:",
            options=[s["label"] for s in subtypes],
            key="detail_radio",
            horizontal=True
        )
        
        # Speichere die Keywords für Backend
        selected_subtype = next((s for s in subtypes if s["label"] == detail_choice), None)
        if selected_subtype:
            st.session_state["symptom_detail"] = selected_subtype["value"]
            st.session_state["detail_keywords"] = selected_subtype["keywords"]
        
        st.markdown("---")
    else:
        st.session_state["symptom_detail"] = ""
        st.session_state["detail_keywords"] = []

# ===== STUFE 3: BEGLEITSYMPTOME =====
if st.session_state["main_symptom"] and st.session_state["main_symptom"] != "Bitte wählen...":
    st.markdown("### 3️⃣ Weitere Symptome (optional)")
    additional = st.text_area(
        "Begleitsymptome:",
        placeholder="z.B. 'Fieber, Kopfschmerzen, Müdigkeit'",
        height=80,
        key="additional_input"
    )
    st.session_state["additional_symptoms"] = additional

st.markdown("---")

# Submit
submit = st.button("🔍 Empfehlung finden", type="primary", use_container_width=True)

# Query zusammenbauen
if st.session_state["main_symptom"] and st.session_state["main_symptom"] != "Bitte wählen...":
    query_parts = [st.session_state["main_symptom"]]
    
    # Detail-Keywords hinzufügen (z.B. "trocken" bei trockenem Husten)
    if st.session_state.get("detail_keywords"):
        query_parts.extend(st.session_state["detail_keywords"])
    
    # Begleitsymptome
    if st.session_state["additional_symptoms"].strip():
        query_parts.append(st.session_state["additional_symptoms"])
    
    query = " ".join(query_parts)
else:
    query = ""


if submit:
    if not query:
        st.warning("Bitte gib deine Beschwerden ein.")
    else:
        with st.spinner('🔍 Analysiere...'):
            user = UserProfile(age=age, is_pregnant=is_pregnant, is_breastfeeding=is_breastfeeding, conditions=selected_conditions)
            recommendations, excluded_reasons = get_recommendations(
                query,
                user_profile=user.__dict__,
                primary_symptom=st.session_state.get("main_symptom")
            )
            
            # --- 1. Global Safety Feedback (Iteratives Feedback) ---
            if excluded_reasons:
                num_excluded = len(excluded_reasons)
                st.info(f"👤 **Dein Profil:** Es wurden **{num_excluded} Pflanzen** zu deiner Sicherheit (Safety Layer) proaktiv herausgefiltert.")
            
            if not recommendations:
                st.warning("Keine passenden Pflanzen gefunden.")
            else:
                st.success(f"Ich habe {len(recommendations)} Optionen gefunden. Top 3:")
                for rec in recommendations[:3]:
                    with st.container():
                        st.markdown("---")
                        
                        sym_use_for_expl = rec.get("matched_features_sym_use", [])
                        chem_for_expl = rec.get("matched_features_chem", [])
                        shap_for_expl = explain_prediction_shap(rec['plant_id'], rec['input_vector'])
                        
                        explanation = generate_layperson_explanation(
                            plant_name=rec['name'],
                            plant_id=rec['plant_id'],
                            input_vector=rec['input_vector'],
                            matched_features_sym_use=sym_use_for_expl,
                            matched_features_chem=chem_for_expl,
                            user_text=query,
                            shap_data=shap_for_expl,
                        )
                        
                        # HEADER & CONFIDENCE
                        c1, c2 = st.columns([4, 1])
                        with c1:
                            conf = explanation['konfidenz_emoji']
                            conf_label = {
                                'hoch': 'Vollständig geprüft und nachgewiesen',
                                'mittel': 'Vollständig geprüft und nachgewiesen (inkl. traditionell)',
                                'traditionell': 'Traditionell angewendet',
                                'indirekt': 'Indirekte Zuordnung',
                            }.get(explanation['konfidenz_level'], '')
                            st.subheader(f"🌿 {rec['name']} {conf}")
                            st.caption(f"Evidenz-Stufe: **{conf_label}**")
                            if rec.get('botanik'): st.caption(f"*{rec['botanik']}*")
                        with c2:
                            st.metric("Relevanz", f"{int(rec['score'] * 100)}%")

                        # PROMINENT WARNINGS (Outside Tabs)
                        st.warning("⚠️ **Hinweis zur Selbstmedikation:** Bei anhaltenden, unklaren oder neu auftretenden Beschwerden sowie bei Fieber über 39°C ist immer ein Arzt aufzusuchen.")

                        # EXPLANATION IN PLAIN LANGUAGE (Directly visible)
                        st.markdown("#### 🔍 Warum passt diese Pflanze zu dir?")
                        st.markdown(explanation['ebene1'])
                        
                        # ACTIONABILITY (Dosage directly visible)
                        prep_df, hints = get_plant_preparations(rec['plant_id'])
                        if not prep_df.empty:
                            st.markdown("#### 🍵 Anwendung & Dosierung")
                            prep_display = prep_df.copy()
                            prep_display.columns = ["Anwendungsform", "Zubereitung / Dosierung"]
                            st.table(prep_display)
                        
                        if hints:
                            for h in hints:
                                st.markdown(f"🔹 {h}")
                                
                        # KI-TRANSPARENCY (Expander)
                        with st.expander("📊 KI- und Wirkstoff-Details ansehen"):
                            tab1, tab2 = st.tabs(["💡 Feature-Einfluss (SHAP)", "🔗 Wirkstoff-Logik"])
                            
                            with tab1:
                                st.markdown("#### Wie kam diese Empfehlung zustande?")
                                st.caption("Dieses Diagramm zeigt, wie das KI-Modell Schritt für Schritt zur finalen Empfehlung gelangt ist.")
                                waterfall_fig = explain_prediction_shap_waterfall(rec['plant_id'], rec['input_vector'])
                                if waterfall_fig is not None:
                                    st.plotly_chart(waterfall_fig, use_container_width=True)
                                
                                st.markdown("#### 🧪 Wirkmechanismus laut KI:")
                                st.markdown(explanation['ebene2'])

                            with tab2:
                                st.write("Welche Wirkstoffe in dieser Pflanze wurden durch deine Symptome aktiviert?")
                                interactions_df = explain_wirkstoff_interactions(rec['plant_id'], rec['input_vector'], query)
                                if interactions_df is not None and not interactions_df.empty:
                                    display_df = interactions_df[['Trigger', 'Gruppe', 'Wirkstoff', 'SHAP']].copy()
                                    display_df.columns = ['Dein Symptom', 'Wirkstoff-Gruppe', 'Enthaltener Stoff', 'Einflussstärke']
                                    group_map = {
                                        'essential_oils': 'Ätherische Öle', 'saponins': 'Saponine', 
                                        'flavonoids': 'Flavonoide', 'mucilages': 'Schleimstoffe',
                                        'tannins': 'Gerbstoffe', 'bitters': 'Bitterstoffe', 'pungent_alkaloids': 'Scharfstoffe'
                                    }
                                    display_df['Wirkstoff-Gruppe'] = display_df['Wirkstoff-Gruppe'].map(group_map).fillna(display_df['Wirkstoff-Gruppe'])
                                    display_df['Einflussstärke'] = display_df['Einflussstärke'].apply(
                                        lambda x: f"+{x:.4f}" if x > 0.01 else "Gering/Indirekt"
                                    )
                                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                                else:
                                    st.caption("Keine spezifischen Wirkstoff-Interaktionen gefunden.")

# Footer (immer sichtbar)
st.markdown("<br><br>", unsafe_allow_html=True)
st.caption("Bachelorarbeit Prototyp • ZHAW 2026 • Generiert durch Hybrid AI (SBERT + Random Forest)")
