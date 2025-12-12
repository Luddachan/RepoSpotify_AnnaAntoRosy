# 🎵 Analisi Dataset Spotify

## 1. Obiettivi del progetto

## 🎯 Obiettivi del Progetto

Il progetto ha l’obiettivo di analizzare il dataset **Spotify 2015–2025** per identificare pattern, relazioni e fattori che influenzano la popolarità dei brani musicali.  
In particolare, l’analisi si concentra su:

- **Esplorazione della distribuzione dei generi musicali nei diversi Paesi**, identificando i generi più rappresentativi per area geografica.
- **Studio delle principali audio-features** (*danceability*, *energy*, *tempo*, *instrumentalness*, *valence*) e della loro correlazione con il punteggio di popolarità.
- **Analisi dell’impatto del contenuto esplicito** sulla popolarità, confrontando brani *explicit* e *non explicit*.
- **Esplorazione del ruolo di artisti ed etichette discografiche** nella diffusione e nel successo dei brani.
- **Produzione di visualizzazioni chiare e interpretative** per supportare l’identificazione di trend, outlier e comportamenti ricorrenti nel dataset.
- **Preparazione dei dati per modelli predittivi di Machine Learning**, attraverso pulizia, trasformazione e selezione delle variabili più informative.


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
