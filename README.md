# DeepNIR: Analisi Spettrale e Modellazione Predittiva
Questa repository contiene il codice per l'addestramento, la valutazione e l'analisi di interpretabilità (SHAP) e ripetibilità di modelli predittivi (XGBoost e SVM) basati su dati spettrali NIR. Il progetto include pipeline di preprocessing, selezione delle bande spettrali e valutazione della robustezza in diverse condizioni di temperatura.

📁 Struttura del Progetto
```text
├── __init__.py
├── utils/                  # Moduli di utilità condivisi
│   ├── data.py             # Caricamento, pulizia e gestione dei dataset
│   ├── models.py           # Definizione e recupero dei modelli (XGB, SVM)
│   ├── preprocessing.py    # Trasformazioni spettroscopiche (SNV, MSC, Savitzky-Golay)
│   └── visual.py           # Funzioni per la generazione di grafici e plot
├── outputs/                # Cartella di output generati
│   ├── svm/                # Risultati specifici per il modello SVM
│   │   ├── performance/    # Metriche di valutazione (R², RMSE, ecc.)
│   │   ├── shap_analysis/  # Grafici e dati SHAP per l'interpretabilità
│   │   └── trained_models/ # Modelli serializzati (pickle)
│   └── xgb/                # Risultati specifici per il modello XGBoost
│       ├── performance/    # Metriche di valutazione
│       ├── shap_analysis/  # Analisi SHAP
│       └── trained_models/ # Modelli serializzati
├── train.py                # Script principale per l'addestramento dei modelli
├── train.sh                # Script bash per lanciare il training in batch
├── robust_temperature.py   # Valutazione della robustezza rispetto alla temperatura
├── robust_temperature_unif.py # Valutazione del modello unificato
├── robust_repeat1.py       # Script di ripetibilità (run 1)
├── robust_repeat2.py       # Script di ripetibilità (run 2)
├── shap_analysis.py        # Analisi SHAP per modelli specifici per coltura
├── shap_analysis_unif.py   # Analisi SHAP per il modello unificato
├── plot_temp_var.ipynb     # Notebook per visualizzare la variazione della temperatura
├── plot_test_metrics.ipynb # Notebook per visualizzare le metriche di test
├── test.py                 # Script di test rapido
└── requirements.txt        # Dipendenze Python necessarie
```

## Prerequisiti
Assicurati di avere Python 3.8 o superiore installato. Installa le dipendenze necessarie:

```bash
pip install -r requirements.txt
```

Dipendenze principali:

scikit-learn
xgboost
shap
pandas
openpyxl (per la lettura/scrittura di Excel)
matplotlib, seaborn (per la visualizzazione)
⚙️ Configurazione
Prima di eseguire gli script, assicurati che:

La variabile d'ambiente DATA_PATH punti alla cartella dei dati grezzi.
La variabile d'ambiente MODELS_PATH e OUT_PATH siano configurate correttamente (o modificate direttamente nel codice se necessario).
I file Excel dei risultati (.xlsx) siano presenti nelle sottocartelle di outputs/ se si desidera visualizzare i risultati pre-calcolati.
📊 Esecuzione degli Script

## 1. Addestramento Modelli
Per addestrare i modelli (XGBoost e SVM) con selezione delle bande e preprocessing:

```bash
# Esegui lo script di training
python train.py
# Oppure usa lo script bash (se configurato)
bash train.sh
```

## 2. Valutazione Robustezza Temperatura
Per valutare come i modelli si comportano al variare della temperatura:

```bash
# Valutazione per singola coltura
python robust_temperature.py

# Valutazione per modello unificato (tutte le colture insieme)
python robust_temperature_unif.py
```

Output: I file Excel (rip_temperatura.xlsx, ecc.) verranno salvati in outputs/xgb/ o outputs/svm/.

## 3. Analisi SHAP (Interpretabilità)
Per generare le analisi di importanza delle feature e i grafici SHAP:

```bash
# Analisi SHAP per modelli specifici
python shap_analysis.py

# Analisi SHAP per il modello unificato
python shap_analysis_unif.py
```

Output: I grafici e i dati verranno salvati in outputs/xgb/shap_analysis/ o outputs/svm/shap_analysis/.

## 4. Visualizzazione
Utilizza i notebook Jupyter per esplorare i risultati:

