import sqlite3
import pandas as pd

conn = sqlite3.connect('heilpflanzen.db')  # Deine DB hier

# Kategorien mit FIXTEN Queries (ein LIKE pro OR)
kategorien = {
    'schlaf': "ps.symptom LIKE '%schlaf%' OR ps.symptom LIKE '%insomnie%' OR ps.symptom LIKE '%einschlaf%' OR ps.symptom LIKE '%durchschlaf%' OR pu.text LIKE '%schlaf%' OR pu.text LIKE '%insomnie%' OR pu.text LIKE '%einschlaf%' OR pu.text LIKE '%durchschlaf%'",
    'verdauung': "ps.symptom LIKE '%bauch%' OR ps.symptom LIKE '%krampf%' OR ps.symptom LIKE '%bläh%' OR ps.symptom LIKE '%dyspepsie%' OR pu.text LIKE '%bauch%' OR pu.text LIKE '%krampf%' OR pu.text LIKE '%bläh%' OR pu.text LIKE '%dyspepsie%'",
    'schmerz': "ps.symptom LIKE '%schmerz%' OR ps.symptom LIKE '%kopfschmerz%' OR ps.symptom LIKE '%migräne%' OR pu.text LIKE '%schmerz%' OR pu.text LIKE '%kopfschmerz%' OR pu.text LIKE '%migräne%'",
    'erkältung': "ps.symptom LIKE '%husten%' OR ps.symptom LIKE '%hal%' OR ps.symptom LIKE '%erkält%' OR pu.text LIKE '%husten%' OR pu.text LIKE '%hal%' OR pu.text LIKE '%erkält%'",
    'nervös': "ps.symptom LIKE '%nervös%' OR ps.symptom LIKE '%unruhe%' OR ps.symptom LIKE '%angst%' OR pu.text LIKE '%nervös%' OR pu.text LIKE '%unruhe%' OR pu.text LIKE '%angst%'",
    'stimmung': "ps.symptom LIKE '%depress%' OR ps.symptom LIKE '%stimmung%' OR pu.text LIKE '%depress%' OR pu.text LIKE '%stimmung%'",
    'magen': "ps.symptom LIKE '%mag%' OR ps.symptom LIKE '%darm%' OR ps.symptom LIKE '%übel%' OR pu.text LIKE '%mag%' OR pu.text LIKE '%darm%' OR pu.text LIKE '%übel%'",
    'haut': "ps.symptom LIKE '%haut%' OR ps.symptom LIKE '%wund%' OR pu.text LIKE '%haut%' OR pu.text LIKE '%wund%'",
    'urin': "ps.symptom LIKE '%harn%' OR ps.symptom LIKE '%blase%' OR pu.text LIKE '%harn%' OR pu.text LIKE '%blase%'",
    'mund': "ps.symptom LIKE '%hal%' OR ps.symptom LIKE '%mund%' OR pu.text LIKE '%hal%' OR pu.text LIKE '%mund%'"
}

for kategorie, where_clause in kategorien.items():
    query = f"""
    SELECT ps.symptom, pu.text as indikation, p.name as pflanze, ps.channel, 
           COUNT(*) as matches
    FROM plant_symptom ps
    JOIN plant_usecase pu ON ps.plant_id = pu.plant_id
    JOIN plant p ON ps.plant_id = p.id
    WHERE {where_clause}
    GROUP BY ps.symptom, pu.text, p.name, ps.channel
    HAVING matches >= 1
    ORDER BY matches DESC
    LIMIT 30
    """
    
    try:
        df = pd.read_sql_query(query, conn)
        if not df.empty:
            df.to_csv(f'kandidaten_{kategorie}.csv', index=False)
            print(f"{kategorie.upper()}: {len(df)} Kandidaten → kandidaten_{kategorie}.csv")
        else:
            print(f"{kategorie.upper()}: Keine Treffer")
    except Exception as e:
        print(f"Fehler bei {kategorie}: {e}")

# Alle Paare (top 100)
query_all = """
SELECT ps.symptom, pu.text as indikation, p.name as pflanze, ps.channel, COUNT(*) as matches
FROM plant_symptom ps 
JOIN plant_usecase pu ON ps.plant_id = pu.plant_id 
JOIN plant p ON ps.plant_id = p.id
GROUP BY ps.symptom, pu.text, p.name, ps.channel
HAVING COUNT(*) >=1
ORDER BY COUNT(*) DESC
LIMIT 100
"""

df_all = pd.read_sql_query(query_all, conn)
df_all.to_csv('kandidaten_all.csv', index=False)
print(f"ALLE: {len(df_all)} Kandidaten → kandidaten_all.csv")

conn.close()
print("✅ FERTIG! CSVs sind im Ordner.")
