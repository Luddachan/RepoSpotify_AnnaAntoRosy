# 🎵 Analisi Dataset Spotify

## 1. Obiettivi del progetto

Il nostro gruppo si propone di analizzare il dataset Spotify per esplorare e comprendere le caratteristiche dei brani musicali e la loro relazione con la popolarità.

Gli obiettivi principali sono:

- Analizzare la distribuzione dei generi più popolari per country
- Studiare la relazione tra caratteristiche audio dei brani (ad esempio: `danceability`, `energy`, `tempo`, `instrumentalness`) e popolarità.
- Valutare la distribuzione dei brani espliciti vs non espliciti e la loro popolarità.
- Analizzare i principali paesi di streaming e il contributo degli artisti e delle etichette discografiche.
- Produrre visualizzazioni chiare e intuitive per supportare l’analisi.

## 2. Deliverable del progetto

Per la consegna mattutina del progetto saranno prodotti i seguenti file:

- **dataset_sporco.csv** → il dataset originale così com’è stato fornito.
- **dataset_pulito.csv** → dataset ripulito, normalizzato e senza duplicati.
- **visualizzazioni_grafiche** → grafici esplorativi dei dati, in formato immagine o notebook.
- **traccia.md** → definizione degli obiettivi funzionali e delle domande di ricerca del gruppo.
- **README.md** → questo file, contenente obiettivi, ruoli, riferimenti e istruzioni iniziali.
## 🎯 Predizioni con Machine Learning

Spotify AI Analyzer permette di stimare la **popolarità di una traccia musicale** utilizzando un modello di **Random Forest** addestrato sulle feature audio e dati categoriali del dataset Spotify.

---

### 🔹 Come funziona

1. **Input utente**
   - L’utente può inserire manualmente valori per alcune feature chiave del brano:
     - `danceability` (0-1)
     - `energy` (0-1)
     - `loudness` (dB, tipicamente da -60 a 5)
   - Il sistema calcola automaticamente eventuali feature derivate presenti nel dataset (ad esempio: `dance_energy_product`, `dance_energy_ratio`).

2. **Template dal dataset**
   - Il tool prende una riga casuale del dataset come **template**.
   - I valori delle colonne presenti nel dataset vengono sostituiti con quelli forniti dall’utente o con la media/mediana se non specificati.
   - Colonne mancanti vengono aggiunte con valori default (0).

3. **Preprocessing**
   - I dati vengono trasformati tramite il **preprocessor** (scaling per numeriche, encoding per categoriche) per adattarli al modello.

4. **Predizione**
   - Il modello Random Forest calcola la popolarità stimata del brano (range 0-100).
   - La predizione viene corretta per rimanere sempre entro il range valido.

5. **Feedback qualitativo**
   - Pop ≥ 80 → 🔥 Potenziale HIT
   - Pop ≥ 60 → 🎵 Buone possibilità di successo
   - Pop ≥ 40 → 📻 Popolarità media
   - Pop < 40 → 💤 Probabile bassa popolarità

---

### 🔹 Generazione tracce casuali

- È possibile generare **N tracce casuali** basate sul dataset.
- Ogni traccia casuale:
  - Mantiene valori realistici per le feature numeriche.
  - Viene trasformata e passata al modello per stimare la popolarità.
- Output fornito:
  - Distribuzione delle predizioni (media, mediana, min/max, deviazione standard)
  - Percentuale di hit potenziali (pop ≥ 80)
  - Visualizzazione istogramma con soglia hit e linea media

---

### 🔹 Animazioni delle predizioni

1. **Animazione interattiva**
   - Mostra barre di popolarità che si aggiornano in tempo reale per ogni traccia generata.
   - Linea storica delle predizioni.
   - Evidenzia le hit con stelle rosse.

2. **Onda sonora basata su predizione**
   - Genera un’onda animata:
     - **Ampiezza** proporzionale alla popolarità.
     - **Frequenza** legata all’energy.
     - **Numero di armoniche** legato alla danceability.
   - Offre una rappresentazione visiva e “musicale” della predizione.

---

### 🔹 File principali coinvolti

| File                        | Descrizione                                      |
|-----------------------------|------------------------------------------------|
| `utils.py`                  | Contiene funzioni per predizione, generazione, animazioni |
| `main.py`                   | Menu interattivo per predizioni e visualizzazioni |
| `rf_model.pkl`              | Modello Random Forest addestrato                |
| `scaler_preprocessor.pkl`   | Preprocessor delle feature                      |
| `X_columns.pkl`             | Lista delle colonne/features usate dal modello |
| `spotify_clean.csv`         | Dataset pulito usato come base                  |

---

### 🔹 Suggerimenti


- La soglia per considerare una traccia “hit” è **80 per default**, ma può essere modificata.
