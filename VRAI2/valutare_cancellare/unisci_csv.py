import os
import pandas as pd
from pathlib import Path

folder_path = "/mnt/c/Users/aless/Desktop/ricerca/deep-nir/src/deep_nir/VRAI2/outputs/xgb_performance"

csvs = {}
for file in os.listdir(folder_path):
    if file.endswith(".csv"):
        file_path = os.path.join(folder_path, file)

        # key = parte prima del primo "_"
        nutriente = file.split("_")[0]
        csvs[nutriente] = {}

        # leggi csv
        df = pd.read_csv(file_path)

        for col in list(df.columns):
            csvs[nutriente][col] = df[col] 


# salva csv finale
import csv
with open(os.path.join(folder_path, "finale.csv"), 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csvs.keys())
    writer.writeheader()  # Write header row
    writer.writerows(csvs)  # Write data rows
