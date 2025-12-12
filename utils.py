# utils.py - VERSIONE ADATTIVA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from IPython.display import display


def get_available_columns(df):
    """Restituisce un dizionario con le colonne disponibili nel dataset."""
    return {
        'numerical': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'all': df.columns.tolist()
    }


def get_valid_input(prompt, min_val, max_val):
    """Richiede input numerico con validazione."""
    while True:
        try:
            value = float(input(prompt))
            if min_val <= value <= max_val:
                return value
            print(f"Valore deve essere tra {min_val} e {max_val}")
        except ValueError:
            print("Inserisci un numero valido")
        except KeyboardInterrupt:
            raise


def crea_input_da_colonne_disponibili(df, user_inputs):
    """
    Crea un dizionario di input usando solo le colonne disponibili nel dataset.
    user_inputs: dict con i valori forniti dall'utente
    """
    input_dict = {}
    available = get_available_columns(df)
    
    # Liste di colonne comuni che potrebbero essere presenti
    numeric_cols = [
        'danceability', 'energy', 'loudness', 'speechiness', 
        'acousticness', 'instrumentalness', 'liveness', 'valence',
        'duration_s', 'explicit', 'key', 'release_year', 'release_month',
        'release_weekday', 'release_quarter', 'release_age',
        'dance_energy_product', 'dance_energy_ratio', 'tempo_loudness_ratio',
        'high_stream', 'high_energy_fast', 'loudness_per_sec',
        'dance_x_loud', 'energy_x_tempo'
    ]
    
    # Aggiungi valori numerici
    for col in numeric_cols:
        if col in df.columns:
            if col in user_inputs:
                # Usa valore fornito dall'utente
                input_dict[col] = user_inputs[col]
            else:
                # Usa media/mediana del dataset
                if df[col].dtype in ['int64', 'int32', 'uint8', 'uint16']:
                    input_dict[col] = int(df[col].median())
                else:
                    input_dict[col] = float(df[col].mean())
    
    # Aggiungi colonne categoriche con valore più frequente
    for col in available['categorical']:
        mode_val = df[col].mode()
        if len(mode_val) > 0:
            input_dict[col] = str(mode_val[0])
        else:
            input_dict[col] = 'Unknown'
    
    return input_dict


def fix_categorical_types(df_input, df_original, preprocessor):
    """
    Assicura che le colonne categoriche abbiano i tipi corretti per il preprocessor.
    """
    # Identifica le colonne categoriche dal preprocessor
    for name, transformer, columns in preprocessor.transformers:
        if 'OneHot' in str(type(transformer)) or hasattr(transformer, 'categories_'):
            for col in columns:
                if col in df_input.columns and col in df_original.columns:
                    # Copia il dtype dalla colonna originale
                    original_dtype = df_original[col].dtype
                    
                    # Se è category, converti
                    if original_dtype.name == 'category':
                        df_input[col] = df_input[col].astype('category')
                        # Assicurati che abbia le stesse categorie
                        df_input[col] = df_input[col].cat.set_categories(
                            df_original[col].cat.categories
                        )
                    elif original_dtype == 'object':
                        # Converti a stringa pulita
                        df_input[col] = df_input[col].astype(str)
                    
    return df_input


def predici_popolarita_interattiva(df, X_columns, preprocessor, final_system):
    """Predice la popolarità di una traccia basandosi su input utente."""
    print("\n🎵 Predizione popolarità Spotify 🎵")
    
    # Filtra X_columns per includere solo colonne presenti nel dataset
    X_columns_available = [col for col in X_columns if col in df.columns]
    
    if len(X_columns_available) < len(X_columns):
        missing = len(X_columns) - len(X_columns_available)
        print(f"{missing} colonne non trovate nel dataset (verranno ignorate)")
    
    # Mostra colonne disponibili
    available = get_available_columns(df)
    print(f"Dataset ha {len(available['numerical'])} colonne numeriche e {len(available['categorical'])} categoriche")
    
    # Input utente con validazione
    try:
        user_inputs = {}
        
        # Input richiesti (adatta in base alle colonne disponibili)
        if 'danceability' in df.columns:
            user_inputs['danceability'] = get_valid_input(
                "Inserisci danceability (0-1): ", 0, 1
            )
        
        if 'energy' in df.columns:
            user_inputs['energy'] = get_valid_input(
                "Inserisci energy (0-1): ", 0, 1
            )
        
        if 'loudness' in df.columns:
            user_inputs['loudness'] = get_valid_input(
                "Inserisci loudness (dB, -60 a 5): ", -60, 5
            )
        
        # Calcola feature derivate se esistono nel dataset
        if 'danceability' in user_inputs and 'energy' in user_inputs:
            if 'dance_energy_product' in df.columns:
                user_inputs['dance_energy_product'] = user_inputs['danceability'] * user_inputs['energy']
            if 'dance_energy_ratio' in df.columns:
                user_inputs['dance_energy_ratio'] = user_inputs['danceability'] / (user_inputs['energy'] + 1e-5)
        
    except KeyboardInterrupt:
        print("\nOperazione annullata.")
        return
    except Exception as e:
        print(f"\nErrore nell'input: {e}")
        return
    
    # STRATEGIA: Usa una riga del dataset come template
    try:
        print("\n Creazione input basato su template del dataset...")
        
        # Prendi una riga casuale come template (USA SOLO COLONNE DISPONIBILI)
        template = df.sample(1)[X_columns_available].copy().reset_index(drop=True)
        
        # Sostituisci con i valori utente
        for col, val in user_inputs.items():
            if col in template.columns:
                template.loc[0, col] = val
        
        # Se X_columns ha colonne extra non nel dataset, aggiungile con 0
        for col in X_columns:
            if col not in template.columns:
                template[col] = 0
        
        # Riordina secondo X_columns
        template = template[X_columns]
        
        print(" Input creato")
        
        # Trasforma e predici
        print(" Trasformazione dati...")
        df_input_pre = preprocessor.transform(template)
        
        print(" Predizione...")
        pred = final_system.predict(df_input_pre)[0]
        pred = max(0, min(100, pred))
        
        print(f"\n Predizione popolarità stimata: {pred:.2f}/100")
        
        # Feedback qualitativo
        if pred >= 80:
            print(" Potenziale HIT!")
        elif pred >= 60:
            print(" Buone possibilità di successo")
        elif pred >= 40:
            print(" Popolarità media")
        else:
            print(" Probabile bassa popolarità")
            
    except Exception as e:
        print(f"\n Errore durante la predizione: {e}")
        print("\n Suggerimenti:")
        print("   1. Esegui: python fix_columns.py")
        print("   2. Oppure rigenera i file .pkl da ml.ipynb")
        import traceback
        traceback.print_exc()


def paesi_hit(df, soglia_hit=80):
    """Mostra i top 10 paesi con più hit."""
    if 'popularity' not in df.columns:
        print(" Colonna 'popularity' non trovata nel dataset")
        return
    
    if 'country' not in df.columns:
        print(" Colonna 'country' non trovata nel dataset")
        # Prova con altre colonne simili
        possible_cols = [col for col in df.columns if 'country' in col.lower() or 'nation' in col.lower()]
        if possible_cols:
            print(f" Trovate colonne alternative: {possible_cols}")
        return
    
    hits = df[df['popularity'] >= soglia_hit]
    
    if len(hits) == 0:
        print(f"  Nessuna traccia trovata con popolarità >= {soglia_hit}")
        max_pop = df['popularity'].max()
        print(f" Popolarità massima nel dataset: {max_pop:.2f}")
        return
    
    top_paesi = hits['country'].value_counts().head(10)
    
    print(f"\n Top 10 Paesi con più hit (pop >= {soglia_hit}):")
    for i, (paese, count) in enumerate(top_paesi.items(), 1):
        print(f"  {i:2d}. {paese:20s}: {count:4d} tracce")
    
    # Grafico
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_paesi.values, y=top_paesi.index, palette='viridis')
    plt.title(f"Top 10 Paesi per hit Spotify (pop >= {soglia_hit})")
    plt.xlabel("Numero di hit")
    plt.ylabel("Paese")
    plt.tight_layout()
    plt.show()


def genera_traccia_casuale(df, X_columns):
    """Genera una traccia con valori casuali basati sul dataset - USA TEMPLATE."""
    
    # Filtra X_columns per includere solo colonne presenti nel dataset
    X_columns_available = [col for col in X_columns if col in df.columns]
    
    # STRATEGIA: Prendi una riga esistente e modifica solo i valori numerici
    template = df.sample(1)[X_columns_available].copy().reset_index(drop=True)
    
    # Modifica solo le colonne numeriche
    for col in template.columns:
        if template[col].dtype in [np.float32, np.float64, np.int32, np.int64, 
                                    np.uint8, np.uint16, np.uint32]:
            col_min = df[col].min() if col in df.columns else 0
            col_max = df[col].max() if col in df.columns else 1
            
            if template[col].dtype in [np.int32, np.int64, np.uint8, np.uint16, np.uint32]:
                template.loc[0, col] = random.randint(int(col_min), int(col_max))
            else:
                template.loc[0, col] = random.uniform(col_min, col_max)
    
    # Aggiungi colonne mancanti con 0 se necessario
    for col in X_columns:
        if col not in template.columns:
            template[col] = 0
    
    # Riordina secondo X_columns
    template = template[X_columns]
    
    return template


def generatore_hit(df, X_columns, preprocessor, final_system, n=1):
    """Genera N tracce casuali e predice la loro popolarità."""
    if n < 1 or n > 100:
        print("  Genera tra 1 e 100 tracce")
        return
    
    print(f"\n Generazione di {n} tracce casuali...")
    
    tracce = []
    preds = []
    errors = 0
    
    for i in range(n):
        try:
            df_traccia = genera_traccia_casuale(df, X_columns)
            df_traccia_pre = preprocessor.transform(df_traccia)
            pred = final_system.predict(df_traccia_pre)[0]
            pred = max(0, min(100, pred))
            preds.append(pred)
            tracce.append(df_traccia)
        except Exception as e:
            errors += 1
            if errors <= 3:  # Mostra solo i primi 3 errori
                print(f"⚠️  Errore traccia {i+1}: {e}")
            continue
    
    if not tracce:
        print(" Nessuna traccia generata con successo")
        return
    
    success_rate = len(tracce) / n * 100
    print(f"\n {len(tracce)}/{n} tracce generate con successo ({success_rate:.1f}%)")
    
    if errors > 0:
        print(f"  {errors} errori durante la generazione")
    
    # Statistiche
    print(f"\nStatistiche popolarità:")
    print(f"  • Media:    {np.mean(preds):.2f}")
    print(f"  • Mediana:  {np.median(preds):.2f}")
    print(f"  • Min/Max:  {np.min(preds):.2f} / {np.max(preds):.2f}")
    print(f"  • Dev.Std:  {np.std(preds):.2f}")
    
    # Conta hit potenziali
    hits = sum(1 for p in preds if p >= 80)
    print(f"  • Hit potenziali (≥80): {hits} ({hits/len(preds)*100:.1f}%)")
    
    # Mostra solo le prime tracce
    df_preds = pd.concat(tracce, ignore_index=True)
    df_preds['predicted_popularity'] = preds
    
    display_limit = min(10, len(df_preds))
    print(f"\n Prime {display_limit} tracce:")
    display(df_preds.head(display_limit)[['predicted_popularity'] + df_preds.columns[:5].tolist()])
    
    # Grafico
    plt.figure(figsize=(10, 6))
    sns.histplot(preds, bins=20, kde=True, color='orange', edgecolor='black')
    plt.axvline(np.mean(preds), color='red', linestyle='--', linewidth=2, label=f'Media: {np.mean(preds):.2f}')
    plt.axvline(80, color='green', linestyle=':', linewidth=2, label='Soglia Hit (80)')
    plt.title(f"Distribuzione Popolarità di {len(preds)} Tracce Generate")
    plt.xlabel("Popolarità stimata")
    plt.ylabel("Conteggio")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    


from matplotlib.animation import FuncAnimation

def visualizza_predizioni_animate(df, X_columns, preprocessor, final_system, n_tracce=50):
    """
    Genera tracce casuali, predice la popolarità e crea un'animazione 
    che mostra le predizioni in tempo reale.
    """
    print(f"\n Generazione di {n_tracce} tracce e visualizzazione animata...")
    
    # --- GENERA TRACCE E PREDIZIONI ---
    tracce_data = []
    
    for i in range(n_tracce):
        try:
            df_traccia = genera_traccia_casuale(df, X_columns)
            df_traccia_pre = preprocessor.transform(df_traccia)
            pred = final_system.predict(df_traccia_pre)[0]
            pred = max(0, min(100, pred))
            
            # Estrai alcune feature per la visualizzazione
            energy = df_traccia['energy'].iloc[0] if 'energy' in df_traccia.columns else 0.5
            danceability = df_traccia['danceability'].iloc[0] if 'danceability' in df_traccia.columns else 0.5
            
            tracce_data.append({
                'pred': pred,
                'energy': energy,
                'danceability': danceability,
                'index': i
            })
        except Exception:
            continue
    
    if not tracce_data:
        print(" Errore nella generazione delle tracce")
        return
    
    print(f" {len(tracce_data)} tracce generate con successo!")
    
    # --- SETUP ANIMAZIONE ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Grafico 1: Barra di popolarità che cresce
    bar = ax1.barh([0], [0], color='orange', height=0.5)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(-1, 1)
    ax1.set_xlabel('Popolarità Predetta', fontsize=12)
    ax1.set_title('🎵 Predizione Popolarità in Tempo Reale', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Linee di riferimento
    ax1.axvline(80, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Soglia Hit')
    ax1.axvline(60, color='yellow', linestyle='--', linewidth=1, alpha=0.5, label='Buono')
    ax1.legend(loc='upper right')
    
    # Testo con info traccia
    text_info = ax1.text(50, 0.7, '', ha='center', va='center', fontsize=10, 
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Grafico 2: Storia delle predizioni (linea che si forma)
    ax2.set_xlim(0, n_tracce)
    ax2.set_ylim(0, 100)
    ax2.set_xlabel('Traccia #', fontsize=12)
    ax2.set_ylabel('Popolarità', fontsize=12)
    ax2.set_title(' Andamento Predizioni', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(80, color='green', linestyle='--', linewidth=1, alpha=0.5)
    
    line, = ax2.plot([], [], 'o-', color='orange', linewidth=2, markersize=4)
    scatter_hits = ax2.scatter([], [], c='red', s=100, marker='*', zorder=5, label='Hit!')
    ax2.legend()
    
    # Dati per l'animazione
    x_data, y_data = [], []
    hit_x, hit_y = [], []
    
    # --- FUNZIONE DI AGGIORNAMENTO ---
    def update(frame):
        if frame >= len(tracce_data):
            return bar, line, scatter_hits, text_info
        
        traccia = tracce_data[frame]
        pred = traccia['pred']
        energy = traccia['energy']
        dance = traccia['danceability']
        
        # Aggiorna barra di popolarità
        bar[0].set_width(pred)
        
        # Colore in base alla popolarità
        if pred >= 80:
            bar[0].set_color('limegreen')
            emoji = '🔥'
        elif pred >= 60:
            bar[0].set_color('gold')
            emoji = '🎵'
        elif pred >= 40:
            bar[0].set_color('orange')
            emoji = '📻'
        else:
            bar[0].set_color('coral')
            emoji = '💤'
        
        # Aggiorna testo
        text_info.set_text(f'{emoji} Traccia #{frame+1}\nPop: {pred:.1f} | Energy: {energy:.2f} | Dance: {dance:.2f}')
        
        # Aggiorna linea storico
        x_data.append(frame)
        y_data.append(pred)
        line.set_data(x_data, y_data)
        
        # Segna gli hit
        if pred >= 80:
            hit_x.append(frame)
            hit_y.append(pred)
            scatter_hits.set_offsets(np.c_[hit_x, hit_y])
        
        return bar, line, scatter_hits, text_info
    
    # --- CREA E MOSTRA ANIMAZIONE ---
    ani = FuncAnimation(fig, update, frames=len(tracce_data), interval=100, 
                       blit=False, repeat=False)
    
    plt.tight_layout()
    plt.show()
    
    # --- STATISTICHE FINALI ---
    preds = [t['pred'] for t in tracce_data]
    hits = sum(1 for p in preds if p >= 80)
    
    print(f"\n STATISTICHE FINALI:")
    print(f"   • Tracce analizzate: {len(tracce_data)}")
    print(f"   • Popolarità media: {np.mean(preds):.2f}")
    print(f"   • Hit potenziali (≥80): {hits} ({hits/len(preds)*100:.1f}%)")
    print(f"   • Range: {np.min(preds):.1f} - {np.max(preds):.1f}")


def visualizza_onda_sonora_da_predizione(df, X_columns, preprocessor, final_system):
    """
    Genera una traccia casuale, predice la popolarità e crea un'onda sonora
    la cui ampiezza e frequenza sono influenzate dalla predizione.
    """
    print("\n🎵 Generazione onda sonora basata su predizione ML...")
    
    # --- GENERA TRACCIA E PREDICI ---
    try:
        df_traccia = genera_traccia_casuale(df, X_columns)
        df_traccia_pre = preprocessor.transform(df_traccia)
        pred = final_system.predict(df_traccia_pre)[0]
        pred = max(0, min(100, pred))
        
        # Estrai feature
        energy = df_traccia['energy'].iloc[0] if 'energy' in df_traccia.columns else 0.5
        danceability = df_traccia['danceability'].iloc[0] if 'danceability' in df_traccia.columns else 0.5
        loudness = df_traccia['loudness'].iloc[0] if 'loudness' in df_traccia.columns else -10
        
    except Exception as e:
        print(f" Errore: {e}")
        return
    
    print(f" Traccia generata:")
    print(f"   • Popolarità predetta: {pred:.1f}/100")
    print(f"   • Energy: {energy:.2f}")
    print(f"   • Danceability: {danceability:.2f}")
    print(f"   • Loudness: {loudness:.1f} dB")
    
    # --- PARAMETRI ONDA INFLUENZATI DALLA PREDIZIONE ---
    # Ampiezza basata su popolarità (hit = ampiezza maggiore)
    ampiezza = 0.5 + (pred / 100) * 1.5  # Range: 0.5-2.0
    
    # Frequenza basata su energy (più energy = più veloce)
    frequenza = 3 + energy * 15  # Range: ~3-18 Hz
    
    # Numero di componenti armoniche basato su danceability
    n_armoniche = int(1 + danceability * 4)  # Range: 1-5
    
    # Velocità animazione basata su energy
    velocita = 0.1 + energy * 0.4  # Range: 0.1-0.5
    
    print(f"\n Parametri onda:")
    print(f"   • Ampiezza: {ampiezza:.2f} (da popolarità)")
    print(f"   • Frequenza: {frequenza:.1f} Hz (da energy)")
    print(f"   • Armoniche: {n_armoniche} (da danceability)")
    
    # --- SETUP GRAFICO ---
    fig, ax = plt.subplots(figsize=(12, 6))
    t = np.linspace(0, 4 * np.pi, 300)
    line, = ax.plot(t, np.sin(t), color='red', linewidth=2)
    
    ax.set_ylim(-ampiezza * 1.2, ampiezza * 1.2)
    ax.set_xlim(0, 4 * np.pi)
    ax.set_title(f"🎵 Onda Sonora - Popolarità: {pred:.1f}/100", 
                fontsize=14, fontweight='bold')
    ax.set_xlabel("Tempo", fontsize=12)
    ax.set_ylabel("Ampiezza", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Colore barra di sfondo basato su popolarità
    if pred >= 80:
        ax.set_facecolor('#e8f5e9')  # Verde chiaro
    elif pred >= 60:
        ax.set_facecolor('#fff9c4')  # Giallo chiaro
    else:
        ax.set_facecolor('#fce4ec')  # Rosa chiaro
    
    # Testo con info
    info_text = (f"Energy: {energy:.2f} | Danceability: {danceability:.2f} | "
                f"Loudness: {loudness:.1f}dB")
    ax.text(0.5, 0.95, info_text, transform=ax.transAxes, 
           ha='center', va='top', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # --- FUNZIONE ANIMAZIONE ---
    def update(frame):
        # Crea onda complessa con armoniche
        onda = np.zeros_like(t)
        for i in range(1, n_armoniche + 1):
            # Ogni armonica con ampiezza decrescente
            amp_armonica = ampiezza / i
            onda += amp_armonica * np.sin(frequenza * i * t + velocita * frame)
        
        line.set_ydata(onda)
        return line,
    
    # --- CREA ANIMAZIONE ---
    ani = FuncAnimation(fig, update, frames=150, interval=50, blit=True)
    
    plt.tight_layout()
    plt.show()
    
    # Feedback finale
    if pred >= 80:
        print("\n Questa traccia ha il potenziale per essere un HIT!")
    elif pred >= 60:
        print("\n Buone vibrazioni! Popolarità interessante.")
    else:
        print("\n Onda tranquilla, popolarità moderata.")