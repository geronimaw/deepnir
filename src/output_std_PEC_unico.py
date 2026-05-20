import os
import pandas as pd
from pathlib import Path

# Percorsi delle cartelle principali
cartella_A = "VRAI2/outputs_std_PEC"
cartella_B = "VRAI2/outputs_std_PEC_svm"

# Lista per raccogliere tutti i DataFrame
dfs = []

# Funzione per leggere i CSV da una cartella e sottocartelle
def leggi_csv_dalla_cartella(cartella_principale):
    for root, _, files in os.walk(cartella_principale):
        if root != "multi_dataset_training_xgb":
            for file in files:
                if file.endswith(".csv"):
                    percorso_file = os.path.join(root, file)
                    df = pd.read_csv(percorso_file)
                    dfs.append(df)
    return df

# Leggi i CSV da A e B
leggi_csv_dalla_cartella(cartella_A)
leggi_csv_dalla_cartella(cartella_B)

# Unisci tutti i DataFrame
tabella_unica = pd.concat(dfs, ignore_index=True)

# Salva il risultato in un nuovo file CSV
tabella_unica.to_csv("./tabella_unica_std_PEC.csv", index=False)

print("Tabella unica creata con successo: tabella_unica.csv")