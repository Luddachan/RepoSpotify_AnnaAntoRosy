#!/usr/bin/env python3
"""
Script per aggiungere le feature mancanti a spotify_clean.csv
"""
import pandas as pd
import numpy as np
import joblib

print("="*70)
print("🔧 RIGENERAZIONE FEATURE - Aggiungi Feature Mancanti")
print("="*70)

# 1. Carica dataset esistente
print("\n📂 Caricamento spotify_clean.csv...")
df = pd.read_csv('spotify_clean.csv')
print(f"✅ Dataset caricato: {df.shape[0]} righe × {df.shape[1]} colonne")
print(f"\n📋 Colonne presenti: {list(df.columns)}")

# 2. Carica X_columns per sapere cosa serve
print("\n📂 Caricamento X_columns.pkl...")
X_columns = joblib.load('X_columns.pkl')
print(f"✅ X_columns caricato: {len(X_columns)} features richieste")

# 3. Identifica feature mancanti
missing = [col for col in X_columns if col not in df.columns]
print(f"\n⚠️  Feature mancanti: {len(missing)}")
for col in missing:
    print(f"   - {col}")

# 4. Crea feature mancanti
print("\n🔨 Creazione feature mancanti...")

# Feature numeriche derivate
if 'release_year' in df.columns and 'release_age' not in df.columns:
    df['release_age'] = 2025 - df['release_year']
    print("✅ Creato: release_age")

if 'danceability' in df.columns and 'energy' in df.columns:
    if 'dance_energy_product' not in df.columns:
        df['dance_energy_product'] = df['danceability'] * df['energy']
        print("✅ Creato: dance_energy_product")
    
    if 'dance_energy_ratio' not in df.columns:
        df['dance_energy_ratio'] = df['danceability'] / (df['energy'] + 1e-5)
        print("✅ Creato: dance_energy_ratio")

if 'energy' in df.columns and 'energy_x_tempo' not in df.columns:
    # Se non c'è tempo, stimiamo da energy
    if 'tempo' in df.columns:
        df['energy_x_tempo'] = df['energy'] * df['tempo']
    else:
        # Stima: energy alta = tempo alto
        estimated_tempo = df['energy'] * 150
        df['energy_x_tempo'] = df['energy'] * estimated_tempo
    print("✅ Creato: energy_x_tempo")

if 'energy' in df.columns and 'high_energy_fast' not in df.columns:
    if 'tempo' in df.columns:
        df['high_energy_fast'] = ((df['tempo'] > 140) & (df['energy'] > 0.7)).astype(int)
    else:
        df['high_energy_fast'] = (df['energy'] > 0.7).astype(int)
    print("✅ Creato: high_energy_fast")

if 'loudness' in df.columns and 'duration_s' in df.columns and 'loudness_per_sec' not in df.columns:
    df['loudness_per_sec'] = df['loudness'] / (df['duration_s'] + 1e-5)
    print("✅ Creato: loudness_per_sec")

if 'danceability' in df.columns and 'loudness_per_sec' in df.columns and 'dance_x_loud' not in df.columns:
    df['dance_x_loud'] = df['danceability'] * df['loudness_per_sec']
    print("✅ Creato: dance_x_loud")

if 'tempo_loudness_ratio' not in df.columns and 'loudness' in df.columns:
    if 'tempo' in df.columns:
        df['tempo_loudness_ratio'] = df['tempo'] / (abs(df['loudness']) + 1e-5)
    else:
        estimated_tempo = df['energy'] * 150 if 'energy' in df.columns else 120
        df['tempo_loudness_ratio'] = estimated_tempo / (abs(df['loudness']) + 1e-5)
    print("✅ Creato: tempo_loudness_ratio")

# Feature categoriche
if 'tempo_cat' not in df.columns:
    if 'tempo' in df.columns:
        df['tempo_cat'] = pd.cut(df['tempo'], bins=[0, 80, 140, 250], labels=['slow', 'medium', 'fast'])
    elif 'energy' in df.columns:
        # Stima da energy
        df['tempo_cat'] = pd.cut(df['energy'], bins=[0, 0.4, 0.7, 1.0], labels=['slow', 'medium', 'fast'])
    else:
        df['tempo_cat'] = 'medium'
    df['tempo_cat'] = df['tempo_cat'].astype(str)
    print("✅ Creato: tempo_cat")

if 'label_grouped' not in df.columns:
    if 'label' in df.columns:
        label_counts = df['label'].value_counts()
        rare_labels = label_counts[label_counts < 50].index
        df['label_grouped'] = df['label'].replace(rare_labels, 'Other')
    else:
        df['label_grouped'] = 'Unknown'
    df['label_grouped'] = df['label_grouped'].astype(str)
    print("✅ Creato: label_grouped")

if 'high_stream' not in df.columns:
    if 'stream_count' in df.columns:
        df['high_stream'] = (df['stream_count'] > df['stream_count'].median()).astype(int)
    else:
        df['high_stream'] = 0
    print("✅ Creato: high_stream")

# 5. Verifica che ora abbiamo tutte le colonne
print(f"\n🔍 Verifica finale...")
still_missing = [col for col in X_columns if col not in df.columns]

if still_missing:
    print(f"⚠️  Ancora mancanti ({len(still_missing)}):")
    for col in still_missing:
        print(f"   - {col}")
        # Aggiungi con valore di default
        df[col] = 0
        print(f"      → Aggiunto con valore 0")
else:
    print("✅ Tutte le colonne richieste sono ora presenti!")

# 6. Salva nuovo dataset
print(f"\n💾 Salvataggio...")
df.to_csv('spotify_clean_BACKUP.csv', index=False)
print("✅ Backup salvato: spotify_clean_BACKUP.csv")

df.to_csv('spotify_clean.csv', index=False)
print("✅ Nuovo dataset salvato: spotify_clean.csv")
print(f"   Dimensioni finali: {df.shape[0]} righe × {df.shape[1]} colonne")

# 7. Test finale
print(f"\n🧪 Test finale...")
try:
    preprocessor = joblib.load('scaler_preprocessor.pkl')
    test_row = df.iloc[0:1][X_columns].copy()
    transformed = preprocessor.transform(test_row)
    print(f"✅ Test riuscito! Il preprocessor funziona correttamente")
    print(f"   Input: {len(X_columns)} features → Output: {transformed.shape[1]} features")
except Exception as e:
    print(f"❌ Test fallito: {e}")

print("\n" + "="*70)
print("🏁 Rigenerazione completata!")
print("="*70)
print("\n💡 Ora puoi eseguire: python main.py")