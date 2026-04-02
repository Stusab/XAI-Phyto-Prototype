#!/usr/bin/env python3
"""
import_symptom_indikation_links.py
Importiert kandidaten_all_filtered.csv in symptom_indikation_link Tabelle.

Schema-Anpassung:
- plant_id sollte FOREIGN KEY sein, nicht PRIMARY KEY
- Das Skript arbeitet mit deinem aktuellen Schema (plant_id als PK)
- WARNUNG: Mit plant_id als PRIMARY KEY kann jede Pflanze nur EINMAL vorkommen!
"""

import sqlite3
import pandas as pd
import sys

CSV_PATH = 'kandidaten_all_filtered.csv'
DB_PATH = 'heilpflanzen.db'

def normalize_channel(channel):
    """Konvertiert CSV-Format zu DB-Format."""
    channel = str(channel).strip().lower()
    if 'evidenz' in channel:
        return 'evidenzbasiert'
    elif 'traditionell' in channel:
        return 'traditionellhmpc'
    return channel

def map_typ_from_score(score):
    """Mappt evidenzscore zu typ (typisch/häufig/selten)."""
    if score >= 4:
        return 'häufig'  # Top-Links (matches=2)
    elif score >= 3:
        return 'typisch'  # Normale Links (matches=1)
    else:
        return 'selten'

def main():
    # 1. CSV laden
    try:
        df = pd.read_csv(CSV_PATH)
        print(f"✓ CSV geladen: {len(df)} Zeilen")
    except Exception as e:
        print(f"❌ FEHLER beim CSV-Laden: {e}")
        sys.exit(1)
    
    # 2. Daten vorbereiten
    df['channel'] = df['channel'].apply(normalize_channel)
    df['evidenzscore'] = (df['matches'] * 2).astype(int)
    df['typ'] = df['evidenzscore'].apply(map_typ_from_score)
    
    print(f"✓ Channel-Werte: {df['channel'].unique()}")
    print(f"✓ Evidenzscores: Min={df['evidenzscore'].min()}, Max={df['evidenzscore'].max()}")
    
    # 3. Verbindung zur DB
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        print(f"✓ DB verbunden: {DB_PATH}")
    except Exception as e:
        print(f"❌ DB-Fehler: {e}")
        sys.exit(1)
    
    # 4. Tabelle leeren (falls gewünscht)
    cursor.execute("DELETE FROM symptom_indikation_link")
    conn.commit()
    print("✓ Alte Daten gelöscht")
    
    # 5. Plant-ID Mapping (alle Pflanzen vorab laden)
    plant_map = {}
    unique_plants = df['pflanze'].unique()
    
    for plant_name in unique_plants:
        # Exakte Suche mit LIKE
        query = "SELECT id, name FROM plant WHERE LOWER(name) LIKE ? LIMIT 1"
        result = cursor.execute(query, (f'%{plant_name.lower()}%',)).fetchone()
        if result:
            plant_map[plant_name] = result[0]
            print(f"  → {plant_name}: ID {result[0]}")
        else:
            print(f"  ⚠ WARNUNG: '{plant_name}' nicht in plant-Tabelle gefunden!")
    
    print(f"✓ {len(plant_map)}/{len(unique_plants)} Pflanzen gefunden")
    
    # 6. INSERT - ALLE Zeilen einzeln
    inserted = 0
    skipped = 0
    
    for idx, row in df.iterrows():
        plant_id = plant_map.get(row['pflanze'])
        
        if not plant_id:
            print(f"  ✗ Zeile {idx}: Pflanze '{row['pflanze']}' übersprungen (keine ID)")
            skipped += 1
            continue
        
        try:
            cursor.execute("""
                INSERT INTO symptom_indikation_link 
                (plant_id, symptom, indikation, typ, evidenzscore, channel)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                plant_id,
                row['symptom'],
                row['indikation'],
                row['typ'],
                int(row['evidenzscore']),
                row['channel']
            ))
            inserted += 1
            
        except sqlite3.IntegrityError as e:
            # Primary Key Konflikt: plant_id doppelt
            print(f"  ✗ Zeile {idx}: {e} (plant_id={plant_id}, Pflanze={row['pflanze']})")
            skipped += 1
        except Exception as e:
            print(f"  ✗ Zeile {idx}: Unbekannter Fehler: {e}")
            skipped += 1
    
    conn.commit()
    conn.close()
    
    # 7. Ergebnis
    print(f"\n{'='*60}")
    print(f"✅ Import abgeschlossen!")
    print(f"   Eingefügt: {inserted}/{len(df)}")
    print(f"   Übersprungen: {skipped}")
    print(f"{'='*60}")
    
    # 8. Verify
    conn = sqlite3.connect(DB_PATH)
    verify = pd.read_sql("SELECT COUNT(*) as total, COUNT(CASE WHEN evidenzscore>=4 THEN 1 END) as top_links FROM symptom_indikation_link", conn)
    conn.close()
    
    print(f"DB-Inhalt: {verify['total'][0]} Links | Top-Links (Score>=4): {verify['top_links'][0]}")
    
    if inserted < len(df):
        print(f"\n⚠ WICHTIG: plant_id ist PRIMARY KEY - nur 1 Link pro Pflanze möglich!")
        print(f"   Lösung: Ändere Schema zu AUTO-INCREMENT ID + plant_id als FOREIGN KEY")

if __name__ == "__main__":
    main()
