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

   ##  Rechtliche Hinweise & Nutzungsbedingungen (Copyright & Terms of Use)

**© 2026 Sabrina Studer. Alle Rechte vorbehalten. (All Rights Reserved)**

Dieses Projekt – einschließlich des gesamten Quellcodes, der Systemarchitektur, der Machine-Learning-Modelle (SBERT, Random Forest), der Datenbank, der XAI-Erklärungslogik sowie des UI-Designs – ist geistiges Eigentum des Autors und im Rahmen einer Bachelorarbeit an der ZHAW (Zürcher Hochschule für Angewandte Wissenschaften) entstanden.

**Erlaubte Nutzung (Strictly Limited Use):**
* Dieses Repository und das darin enthaltene System ("PhytoMatch AI") dürfen **ausschließlich zu Zwecken des vorgesehenen, betreuten User-Testings** und der akademischen Begutachtung im Rahmen dieser Bachelorarbeit verwendet werden.

**Ausdrückliche Verbote:**
* Es wird **keine Open-Source-Lizenz** erteilt. 
* Jegliche andere Nutzung, Vervielfältigung, Verbreitung, Modifikation, Weiterentwicklung, kommerzielle Nutzung oder das Reverse Engineering des Codes, der Modelle oder der Datenbankstruktur ist ohne vorherige schriftliche Zustimmung des Autors **strikt untersagt**.

**Medizinischer Disclaimer:**
Dieser Prototyp dient ausschließlich Forschungs- und Demonstrationszwecken. Er ersetzt keine professionelle medizinische Beratung, Diagnose oder Behandlung. Für gesundheitliche Schäden oder Folgen aus der Nutzung des Systems wird jede Haftung ausgeschlossen.

