import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import sys

# --- 1. PREPARAZIONE DATI E FUNZIONI DI FEATURE ENGINEERING ---

def prepare_data_and_train_model(csv_file):
    """Carica i dati, esegue Feature Engineering, Preprocessing e Addestra il Modello."""
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Errore: File '{csv_file}' non trovato.")
        sys.exit()

    # --- FEATURE ENGINEERING (Stesse regole dell'addestramento) ---
    
    # Tempo Categoria (Fast/Medium/Slow)
    tempo_median = df['tempo'].median()
    def tempo_category(tempo):
        if tempo < 100:
            return 'slow'
        elif tempo < 140:
            return 'medium'
        else:
            return 'fast'
    df['tempo_cat'] = df['tempo'].apply(tempo_category)

    # Loudness Binning
    bins = [-np.inf, -10, -5, np.inf]
    labels = ['low_loudness', 'medium_loudness', 'high_loudness']
    df['loudness_cat'] = pd.cut(df['loudness'], bins=bins, labels=labels, right=False)

    # Termine di Interazione: Danceability-Loudness
    df['dance_loud_interact'] = df['danceability'] * df['loudness']

    # --- SELEZIONE FEATURE E SPLIT ---
    features_num = ['danceability', 'energy', 'tempo', 'instrumentalness', 'duration_s', 
                    'dance_loud_interact'] 
    features_cat = ['key', 'mode', 'tempo_cat', 'loudness_cat'] 

    X = df[features_num + features_cat]
    y = df['popularity']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- PREPROCESSING ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), features_num),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), features_cat)
        ]
    )

    X_train_proc = preprocessor.fit_transform(X_train)

    # --- ADDESTRAMENTO MODELLO ---
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_proc, y_train)
    
    # Restituisce gli oggetti necessari per la predizione
    return model, preprocessor, features_num, features_cat, tempo_category, bins, labels

# --- 2. FUNZIONE INTERATTIVA PER LA PREDIZIONE (IL TUO "MENU") ---

def predict_new_song_interactive(model, preprocessor, features_num, features_cat, tempo_category, bins, labels):
    """Chiede i valori delle feature all'utente e predice la popolarità."""
    
    print("\n" + "="*50)
    print(" 💿 MENU DI PREDIZIONE POPOLARITÀ SPOTIFY 💿")
    print(" Inserisci i valori del tuo nuovo brano.")
    print("="*50)
    
    new_song_data = {}
    
    # 1. Input delle Feature Numeriche
    print("\n--- Caratteristiche Acustiche (Valori da 0.0 a 1.0 o reali) ---")
    
    for feature in ['danceability', 'energy', 'instrumentalness']:
        while True:
            try:
                val = float(input(f"Inserisci {feature} (0.0 a 1.0): "))
                if 0.0 <= val <= 1.0:
                    new_song_data[feature] = [val]
                    break
                else:
                    print("Valore non valido. Deve essere tra 0.0 e 1.0.")
            except ValueError:
                print("Input non valido. Inserisci un numero.")

    # Input delle altre Feature Numeriche
    while True:
        try:
            new_song_data['loudness'] = [float(input("Inserisci loudness (dB, es. -5.0): "))]
            new_song_data['tempo'] = [float(input("Inserisci tempo (BPM, es. 120.0): "))]
            new_song_data['duration_s'] = [float(input("Inserisci durata_s (secondi, es. 210.0): "))]
            break
        except ValueError:
            print("Input non valido. Inserisci un numero.")
            
    # 2. Input delle Feature Categoriche
    print("\n--- Caratteristiche Musicali ---")
    while True:
        try:
            key_val = int(input("Inserisci key (0-11, dove 0=C, 1=C#, ecc.): "))
            if 0 <= key_val <= 11:
                new_song_data['key'] = [key_val]
                break
            else:
                print("Valore non valido. Inserisci un numero tra 0 e 11.")
        except ValueError:
            print("Input non valido. Inserisci un numero intero.")
            
    while True:
        try:
            mode_val = int(input("Inserisci mode (0=Minore, 1=Maggiore): "))
            if mode_val in [0, 1]:
                new_song_data['mode'] = [mode_val]
                break
            else:
                print("Valore non valido. Inserisci 0 o 1.")
        except ValueError:
            print("Input non valido. Inserisci un numero intero.")

    # Crea il DataFrame
    new_song_df = pd.DataFrame(new_song_data)
    
    # --- 3. APPLICAZIONE DELLA FEATURE ENGINEERING ---
    # Applicare le stesse trasformazioni del training
    
    # tempo_cat
    new_song_df['tempo_cat'] = new_song_df['tempo'].apply(tempo_category)
    
    # loudness_cat
    new_song_df['loudness_cat'] = pd.cut(new_song_df['loudness'], bins=bins, labels=labels, right=False)

    # dance_loud_interact
    new_song_df['dance_loud_interact'] = new_song_df['danceability'] * new_song_df['loudness']

    # --- 4. PREPROCESSING E PREDIZIONE ---
    
    # Assicurarsi che le colonne siano nell'ordine corretto
    all_features = features_num + features_cat
    new_song_X = new_song_df[all_features]
    
    # Trasformazione dei dati utilizzando il preprocessor fit-tato
    new_song_proc = preprocessor.transform(new_song_X)
    
    # Predizione finale
    predicted_popularity = model.predict(new_song_proc)

    print("\n" + "="*50)
    print(f"La Popolarità Prevista per il tuo Brano è: **{predicted_popularity[0]:.2f}**")
    print("="*50)
    print("Ricorda: La Popolarità Spotify è un punteggio da 0 a 100.")
    
# --- 3. ESECUZIONE PRINCIPALE ---

if __name__ == '__main__':
    print("FASE 1: Addestramento del Modello...")
    model, preprocessor, features_num, features_cat, tempo_category, bins, labels = prepare_data_and_train_model('spotify_clean.csv')
    print("Modello pronto per le previsioni!")
    
    # Avvia il "menu" di predizione
    predict_new_song_interactive(model, preprocessor, features_num, features_cat, tempo_category, bins, labels)