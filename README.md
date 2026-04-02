# XAI-Phyto: Explainable AI für Phytotherapie

Dieses Repository beinhaltet den interaktiven Prototyp für eine Bachelorarbeit im Bereich **Explainable Medical AI (XAI)**.
Das System nutzt kaskadierende Hybridisierung, um bei Nutzer-Symptomen evidenzbasierte pflanzliche Heilmittel (Phytotherapie) zu empfehlen. Dabei kommen strikte Sicherheitsfilter (Safety Layer) zum Einsatz, um Kontraindikationen zu vermeiden.

## Systemarchitektur
- **Frontend:** Streamlit 
- **Relevance Layer:** Sentence-BERT (`sentence-transformers`) für das Semantic Matching unstrukturierter Nutzereingaben.
- **Classification:** Random Forest Model
- **Explainability (XAI):** SHAP (SHapley Additive exPlanations) zur Berechnung exakter Feature-Beiträge. Die SHAP-Werte werden in ein auf Laien ausgerichtetes Text-Template übersetzt (UCXAISD-Prinzipien).

## Projektstruktur
- `app.py`: Einstiegspunkt und Streamlit-Frontend
- `src/logic.py`: Backend, SBERT, Random Forest und XAI (SHAP) Kalkulation
- `src/knowledge_base.py`: Zentraer Mapping-Katalog für Laien-Erklärungen und linguistische NLP-Regeln
- `data/`: Ordner für die phytotherapeutische Matrix und exportierte Joblib-Modelle

## Installation & Start (Reproduzierbarkeit)

1. **Virtuelles Environment (optional aber empfohlen):**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Mac/Linux:
   source venv/bin/activate
   ```

2. **Abhängigkeiten installieren:**
   ```bash
   pip install -r requirements.txt
   ```

3. **App starten:**
   ```bash
   streamlit run app.py
   ```
