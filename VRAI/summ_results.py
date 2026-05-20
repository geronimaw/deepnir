import os
import re
import pandas as pd
from pathlib import Path

path = "/mnt/c/users/aless/Desktop/ricerca/deep-nir/src/deep_nir/VRAI/outputs_noPrep"

METRICS = ["SEP", "Bias", "SEPC", "R2", "Slope"]
PREFIXES = ["val_", "train_"]  # validation e training

def extract_metrics_from_sheet(df, file_name, sheet_name):
    """
    df: dataframe letto dal foglio, deve contenere colonna A e B (0 e 1)
    Estrae i valori in colonna B dove colonna A contiene le chiavi richieste.
    """
    # Prendiamo solo le prime due colonne come A e B
    df = df.iloc[:, :2].copy()
    df.columns = ["A", "B"]

    # Normalizza la colonna A per matching robusto
    df["A"] = df["A"].astype(str).str.strip()

    rows = []
    for pref in PREFIXES:
        for m in METRICS:
            key = f"{pref}{m}"
            # match "contains" case-insensitive
            mask = df["A"].str.contains(re.escape(key), case=False, na=False)

            value = None
            if mask.any():
                # se ce ne sono più di uno, prendo il primo (modifica qui se vuoi altro criterio)
                value = df.loc[mask, "B"].iloc[0]

            rows.append({
                "file": file_name.split("_")[1],
                "sheet": sheet_name.split("_SVM")[0],
                "metric_key": key,           # es. train_SEP / performance_R2
                "value": value
            })
    return rows

def main(path):
    path = Path(path)
    global_rows = []  # Lista per raccogliere tutti i dati

    for fold in os.listdir(path):
        # dividi per "_" e prendi il secondo elemento
        name = fold.split("_")[1]
        full_path = os.path.join(path, fold)
        if not "multi" in fold and os.path.isdir(full_path):

            xlsx_files = sorted(Path(full_path).glob("*.xlsx"))

            all_rows = []

            for xlsx in xlsx_files:
                # legge i nomi fogli senza caricare tutto
                xl = pd.ExcelFile(xlsx)

                # solo fogli con "_SVM" nel nome. prendi la parte prima di "_SVM" come nome sheet
                svm_sheets = [s for s in xl.sheet_names if "_SVM" in s]

                for sh in svm_sheets:
                    # Leggi il foglio (senza header), così la riga con performance_* o train_* può stare ovunque
                    df = pd.read_excel(xlsx, sheet_name=sh, header=None, engine="openpyxl")
                    extracted_rows = extract_metrics_from_sheet(df, xlsx.name, sh)
                    all_rows.extend(extracted_rows)
                    global_rows.extend(extracted_rows)  # Aggiungi i dati alla lista globale

            long_df = pd.DataFrame(all_rows)

            # Wide: una riga per file+sheet, colonne = metric_key
            wide_df = (long_df
                    .pivot_table(index=["file", "sheet"], columns="metric_key", values="value", aggfunc="first")
                    .reset_index())

            # Ordina colonne in modo “pulito”
            desired_cols = ["file", "sheet"] + [f"{p}{m}" for p in PREFIXES for m in METRICS]
            for c in desired_cols:
                if c not in wide_df.columns:
                    wide_df[c] = pd.NA
            wide_df = wide_df[desired_cols]

            out = path / f"summary_{name}_SVM_metrics.xlsx"
            with pd.ExcelWriter(out, engine="openpyxl") as w:
                wide_df.to_excel(w, index=False, sheet_name="wide_summary")
                long_df.to_excel(w, index=False, sheet_name="long_summary")

            print(f"Creato: {out}")

    # Salva la tabella globale
    global_long_df = pd.DataFrame(global_rows)
    global_wide_df = (global_long_df
                      .pivot_table(index=["file", "sheet"], columns="metric_key", values="value", aggfunc="first")
                      .reset_index())

    # Ordina colonne in modo “pulito”
    desired_cols = ["file", "sheet"] + [f"{p}{m}" for p in PREFIXES for m in METRICS]
    for c in desired_cols:
        if c not in global_wide_df.columns:
            global_wide_df[c] = pd.NA
    global_wide_df = global_wide_df[desired_cols]

    global_out = path / "global_summary_SVM_metrics.xlsx"
    with pd.ExcelWriter(global_out, engine="openpyxl") as w:
        global_wide_df.to_excel(w, index=False, sheet_name="wide_summary")
        global_long_df.to_excel(w, index=False, sheet_name="long_summary")

    print(f"Creato file globale: {global_out}")

# Esegui:
main(path)
