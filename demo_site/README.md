# Dimostratore web per la tesi

Questa cartella contiene una demo **Streamlit** che usa la pipeline finale del progetto e produce un output **video-level** nello stile di `test_joint`:

- input: video `.mp4` / `.avi` / `.mov` ...
- preprocessing: crop `10` con **letterbox 224x224**
- estrazione feature CLIP `ViT-B/16`
- ricostruzione dei segmenti fino a `visual_length = 256`
- inferenza con il modello finale
- output: **Incidente / Non incidente**, confidence, threshold, timeline score

## File inclusi

- `app.py`: interfaccia web Streamlit
- `demo_backend.py`: caricamento modello, estrazione feature, inferenza singolo video
- `run_inference_cli.py`: test da terminale senza interfaccia web
- `config.example.json`: esempio di configurazione
- `requirements.txt`: dipendenze principali

## 1) Dove mettere questi file

Metti questa cartella **fuori oppure dentro** il tuo progetto, ma devi conoscere il percorso della cartella `src` originale della tesi.

Esempio:

```text
Codice_RescignoAndrea/
├── list/
├── src/
│   ├── model_with_MLP.py
│   ├── test_joint.py
│   ├── crop.py
│   └── ...
└── tesi_demo_site/
    ├── app.py
    ├── demo_backend.py
    └── ...
```

## 2) Installa le dipendenze

Dentro la cartella della demo:

```bash
pip install -r requirements.txt
```

## 3) Crea il file `config.json`

Copia `config.example.json` in `config.json` e modifica i percorsi.

Campi essenziali:

- `src_root`: cartella `src` del tuo progetto
- `weights_path`: checkpoint finale `.pth`
- `val_list`: CSV di validation, se vuoi che la soglia venga ricalcolata come nel test finale
- `fixed_threshold`: se vuoi forzare una soglia precisa, ad esempio `0.47`

### Scelta soglia

Hai due possibilità:

- `fixed_threshold = null` e `val_list` valorizzato: la demo calcola la soglia sulla validation
- `fixed_threshold = numero`: la demo usa quella soglia direttamente

Per la discussione della tesi è spesso meglio usare la **soglia finale fissata**, così il comportamento della demo è stabile.

## 4) Avvio del sito

```bash
streamlit run app.py
```

Poi si apre il browser. Carichi il video, premi **Analizza video** e ottieni:

- prediction finale
- confidence
- threshold usata
- segmento più sospetto
- timeline degli score

## 5) Test rapido da terminale

Puoi provare prima senza sito:

```bash
python run_inference_cli.py --config config.json --video "C:/video/prova.mp4"
```

oppure:

```bash
python run_inference_cli.py \
  --src-root "C:/Users/andre/Desktop/Codice_RescignoAndrea/src" \
  --weights-path "C:/Users/andre/Desktop/Codice_RescignoAndrea/src/model/model_best.pth" \
  --val-list "C:/Users/andre/Desktop/Codice_RescignoAndrea/list/CLIP_rgbval.csv" \
  --video "C:/video/prova.mp4"
```

## 6) Come funziona internamente

La demo esegue questi passi:

1. apre il video con OpenCV;
2. se gli FPS sono alti, fa downsampling verso `target_fps`;
3. prende un frame ogni `stride=16` frame validi;
4. applica il crop `10` con letterbox;
5. estrae le feature CLIP `ViT-B/16`;
6. costruisce la sequenza di feature del video;
7. usa il modello finale per ottenere il logit video-level;
8. applica la threshold e restituisce la classe finale.

## 7) Nota importante

Nel tuo `test_joint.py` originale c'è un punto da correggere: la funzione `load_model` seleziona `weight_path`, ma poi carica un file fisso (`model/model_cur.pth`). In questa demo il caricamento è già corretto: viene caricato davvero il checkpoint indicato in `weights_path`.

## 8) Consigli pratici per la presentazione

Per la discussione conviene usare 3-5 video già pronti:

- 2 video con incidente
- 2 video normali
- 1 video borderline

Così fai vedere sia i casi facili sia quelli difficili.

## 9) Possibili estensioni

Puoi aggiungere facilmente:

- salvataggio dei risultati in CSV
- batch di più video
- esportazione di un report
- evidenziazione del tratto più sospetto direttamente nel player
