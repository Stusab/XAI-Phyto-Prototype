import sqlite3
import pandas as pd

conn = sqlite3.connect('heilpflanzen.db')

# 1. Clean Start
conn.execute("DELETE FROM symptom_indikation_link;")
conn.commit()

# 2. CSV laden
df = pd.read_csv('kandidaten_all_filtered.csv')
df['matches'] = pd.to_numeric(df['matches'], errors='coerce').fillna(1).astype(int)

# 3. Regelbasierter typ (Constraint-sicher: unique Labels)
def get_typ(matches):
    if matches >= 3: return 'typisch'
    if matches == 2: return 'häufig'
    return 'selten'

df_import = df[['symptom', 'indikation', 'channel']].copy()
df_import['evidenzscore'] = df['matches']
df_import['typ'] = df['matches'].apply(get_typ)

print(f"44 Rows bereit:\n{df_import[['symptom', 'typ', 'evidenzscore']].head()}")

# 4. Insert
df_import.to_sql('symptom_indikation_link', conn, if_exists='append', index=False)
conn.commit()

# 5. Validierung
val1 = pd.read_sql("SELECT COUNT(*) as rows, MIN(evidenzscore), MAX(evidenzscore) FROM symptom_indikation_link;", conn)
val2 = pd.read_sql("SELECT symptom, indikation, typ, evidenzscore FROM symptom_indikation_link ORDER BY evidenzscore DESC LIMIT 5;", conn)
print(val1)
print(val2)

conn.close()
print("✅ Erfolg! Tabelle gefüllt.")
