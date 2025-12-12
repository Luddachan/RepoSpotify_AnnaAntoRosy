#!/usr/bin/env python3
# main.py - VERSIONE COMPLETA CON ANIMAZIONI

from utils import (
    predici_popolarita_interattiva, 
    paesi_hit, 
    generatore_hit,
    visualizza_predizioni_animate, 
    visualizza_onda_sonora_da_predizione
)
import joblib
import pandas as pd
import sys
import os


def carica_risorse():
    """Carica tutti i file necessari per il funzionamento."""
    try:
        print("📦 Caricamento risorse...")
        
        # Carica dataset
        if not os.path.exists("spotify_clean.csv"):
            print("❌ File 'spotify_clean.csv' non trovato!")
            print("💡 Esegui prima: python regenerate_features.py")
            return None, None, None, None
        
        df = pd.read_csv("spotify_clean.csv")
        print(f"✅ Dataset caricato: {len(df)} righe × {df.shape[1]} colonne")
        
        # Carica preprocessor
        if not os.path.exists("scaler_preprocessor.pkl"):
            print("❌ File 'scaler_preprocessor.pkl' non trovato!")
            print("💡 Esegui il notebook ml.ipynb per generare i file")
            return None, None, None, None
        
        preprocessor = joblib.load("scaler_preprocessor.pkl")
        print("✅ Preprocessor caricato")
        
        # Carica modello
        if not os.path.exists("rf_model.pkl"):
            print("❌ File 'rf_model.pkl' non trovato!")
            print("💡 Esegui il notebook ml.ipynb per generare i file")
            return None, None, None, None
        
        final_system = joblib.load("rf_model.pkl")
        print("✅ Modello caricato")
        
        # Carica colonne
        if not os.path.exists("X_columns.pkl"):
            print("❌ File 'X_columns.pkl' non trovato!")
            print("💡 Esegui il notebook ml.ipynb per generare i file")
            return None, None, None, None
        
        X_columns = joblib.load("X_columns.pkl")
        print(f"✅ Colonne caricate: {len(X_columns)} features")
        
        # Verifica allineamento colonne
        missing_cols = [col for col in X_columns if col not in df.columns]
        if missing_cols:
            print(f"\n⚠️  {len(missing_cols)} colonne mancanti nel dataset:")
            for col in missing_cols[:5]:
                print(f"   - {col}")
            if len(missing_cols) > 5:
                print(f"   ... e altre {len(missing_cols) - 5}")
            print("\n💡 Esegui: python regenerate_features.py")
            
            risposta = input("\nVuoi continuare comunque? (s/n): ").strip().lower()
            if risposta != 's':
                return None, None, None, None
        
        print("🎉 Tutte le risorse caricate con successo!\n")
        return df, X_columns, preprocessor, final_system
        
    except Exception as e:
        print(f"❌ Errore durante il caricamento: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def menu_interattivo(df, X_columns, preprocessor, final_system):
    """Menu principale dell'applicazione."""
    while True:
        print("\n" + "="*55)
        print("🎵  SPOTIFY AI - MENU PRINCIPALE  🎵".center(55))
        print("="*55)
        print("  1. 🎤  Predici popolarità traccia")
        print("  2. 🌍  Top paesi con più hit")
        print("  3. 🎲  Genera tracce casuali e statistiche")
        print("  4. 🎬  Animazione predizioni in tempo reale")
        print("  5. 🎵  Onda sonora da predizione ML")
        print("  6. 👋  Esci")
        print("="*55)
        
        scelta = input("\n➤ Scegli un'opzione (1-6): ").strip()
        
        if scelta == "1":
            print("\n" + "="*55)
            predici_popolarita_interattiva(df, X_columns, preprocessor, final_system)
            
        elif scelta == "2":
            print("\n" + "="*55)
            try:
                soglia = input("Inserisci soglia popolarità (default 80): ").strip()
                soglia = int(soglia) if soglia else 80
                soglia = max(0, min(100, soglia))
                paesi_hit(df, soglia_hit=soglia)
            except ValueError:
                print("⚠️  Valore non valido, uso soglia 80")
                paesi_hit(df, soglia_hit=80)
            except KeyboardInterrupt:
                print("\n⚠️  Operazione annullata.")
                
        elif scelta == "3":
            print("\n" + "="*55)
            try:
                n_str = input("Quante tracce vuoi generare? (1-100, default 10): ").strip()
                n = int(n_str) if n_str else 10
                n = max(1, min(100, n))
                generatore_hit(df, X_columns, preprocessor, final_system, n=n)
            except ValueError:
                print("⚠️  Valore non valido, genero 10 tracce")
                generatore_hit(df, X_columns, preprocessor, final_system, n=10)
            except KeyboardInterrupt:
                print("\n⚠️  Operazione annullata.")
        
        elif scelta == "4":
            print("\n" + "="*55)
            try:
                n_str = input("Quante tracce visualizzare? (10-100, default 50): ").strip()
                n = int(n_str) if n_str else 50
                n = max(10, min(100, n))
                visualizza_predizioni_animate(df, X_columns, preprocessor, final_system, n_tracce=n)
            except ValueError:
                print("⚠️  Valore non valido, uso 50 tracce")
                visualizza_predizioni_animate(df, X_columns, preprocessor, final_system, n_tracce=50)
            except KeyboardInterrupt:
                print("\n⚠️  Operazione annullata.")
            except Exception as e:
                print(f"\n❌ Errore durante l'animazione: {e}")
        
        elif scelta == "5":
            print("\n" + "="*55)
            try:
                visualizza_onda_sonora_da_predizione(df, X_columns, preprocessor, final_system)
            except KeyboardInterrupt:
                print("\n⚠️  Operazione annullata.")
            except Exception as e:
                print(f"\n❌ Errore durante l'animazione: {e}")
                
        elif scelta == "6":
            print("\n" + "="*55)
            print("👋 Grazie per aver usato Spotify AI!".center(55))
            print("🎶 A presto!".center(55))
            print("="*55 + "\n")
            break
            
        else:
            print("⚠️  Opzione non valida. Scegli un numero tra 1 e 6.")


def stampa_banner():
    """Stampa banner di benvenuto."""
    banner = """
    ╔═══════════════════════════════════════════════════╗
    ║                                                   ║
    ║           🎵  SPOTIFY AI ANALYZER  🎵            ║
    ║                                                   ║
    ║     Predici la popolarità delle tue tracce      ║
    ║          con Machine Learning & AI               ║
    ║                                                   ║
    ╚═══════════════════════════════════════════════════╝
    """
    print(banner)


# --- MAIN ---
if __name__ == "__main__":
    try:
        stampa_banner()
        
        # Carica risorse
        df, X_columns, preprocessor, final_system = carica_risorse()
        
        # Verifica che tutto sia stato caricato correttamente
        if df is None or X_columns is None or preprocessor is None or final_system is None:
            print("\n❌ Impossibile avviare l'applicazione.")
            print("\n📋 File necessari:")
            print("   • spotify_clean.csv")
            print("   • scaler_preprocessor.pkl")
            print("   • rf_model.pkl")
            print("   • X_columns.pkl")
            print("\n💡 Suggerimenti:")
            print("   1. Esegui: python regenerate_features.py")
            print("   2. Oppure riesegui il notebook ml.ipynb")
            sys.exit(1)
        
        # Avvia menu
        menu_interattivo(df, X_columns, preprocessor, final_system)
        
    except KeyboardInterrupt:
        print("\n\n👋 Applicazione interrotta dall'utente. Ciao!")
    except Exception as e:
        print(f"\n❌ Errore inaspettato: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)