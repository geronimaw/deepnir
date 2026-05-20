import pandas as pd

# Define target columns for the dataset in Grain > 08_CornGrain 
y_columns = {
    "DATASET": ["DM", "Starch", "Protein", "ADF", "NDF", "Ash", "Crude Fat", "Crude Fib."],
    "VALID": ["DM", "Starch", "Crude Protein", "ADF", "NDF", "Ash", "Crude Fat", "Crude Fiber"],
    "ALL_FOSS": ["SS", "AMIDO_SS", "PG_SS", "ADF_SS", "NDF_SS", "CEN_SS", "EE_SS", "FG_SS"],
    "ALL_Gonzaga": ["SS", "AMIDO_SS", "PG_SS", "ADF_SS", "NDF_SS", "CEN_SS", "EE_SS", "FG_SS"],
    "ALL_LUFA": ["SS", "AMIDO_SS", "PG_SS", "ADF_SS", "NDF_SS", "CEN_SS", "EE_SS", "FG_SS"],
    "DATASET_new": ["SS", "AMIDO_SS", "PG_SS", "ADF_SS", "NDF_SS", "CEN_SS", "EE_SS"]
}
col_names_switch = {
    "SS": "DM",
    "AMIDO_SS": "Starch",
    "PG_SS": "Protein",
    "ADF_SS": "ADF",
    "NDF_SS": "NDF",
    "CEN_SS": "Ash",
    "EE_SS": "Crude Fat",
    "FG_SS": "Crude Fib.",
}

# Select features and targets
def get_x_y_labels(data, y_column, first_freq):
    # risolvi nome colonna y
    if y_column not in data.columns:
        matches = [k for k, v in col_names_switch.items() if v == y_column]
        if not matches:
            raise KeyError(
                f"y_column='{y_column}' non trovata in data.columns e nessuna mappatura in col_names_switch. "
                f"Esempi colonne: {list(data.columns)[:10]}"
            )
        y_column = matches[0]

    y = pd.to_numeric(data[y_column], errors="coerce")

    # risolvi colonne X (vedi sotto: il tuo slicing '1100':'1800' dipende dai nomi!)
    x_columns = data.loc[:, "1100":"1800"].columns.tolist()
    if first_freq > 0:
        print(f"\tCropping spectrum to the first {first_freq} frequencies")
        x_columns = x_columns[:first_freq]
        print(f"\t\tusing {len(x_columns)} frequencies")
    X = data[x_columns]
    return X, y