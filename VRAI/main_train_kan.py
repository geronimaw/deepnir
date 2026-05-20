import os
import pandas as pd
import xgboost as xgb
import numpy as np

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from VRAI.utils.models import train_svm_classif, train_svm_regr
from VRAI.utils.data import y_columns, get_x_y_labels, col_names_switch

HERE = Path(__file__).parent
ROOT = HERE.parents[2]

out_path = ROOT / "src/deep_nir/VRAI"
in_path = ROOT / "data/raw/Grain"

def group_wavelengths(data, beams_step):
    """
    Groups wavelengths in the dataset based on the beams_step value.
    """
    if beams_step == 1:
        return data  # No grouping, use all wavelengths as-is
    
    data = data.select_dtypes(include="number")

    # Group columns by step
    grouped_data = pd.DataFrame()
    num_wavelengths = data.shape[1]
    for i in range(0, num_wavelengths, beams_step):
        group_cols = data.iloc[:, i:i + beams_step]
        grouped_data[f"group_{i // beams_step}"] = group_cols.mean(axis=1)
    return grouped_data

def main(dataset, path, beams_step):
    # Load training and validation data from Excel sheets
    for sheet_name in sheet_names:
        if sheet_name == "DATASET":
            data = pd.read_excel(path, sheet_name=sheet_name)
            val_data = pd.read_excel(path, sheet_name="VALID")
            val_data.columns = val_data.columns.map(str)
        else:
            data = pd.read_excel(path, sheet_name=sheet_name, header=1)
            val_data = None
        data.columns = data.columns.map(str)
        
        # Get dtrain and dtest 
        train_data = data
        if val_data is None:
            train_data = data.sample(frac=0.7, random_state=42)
            val_data = data.drop(train_data.index)

        # Train a different model for each target
        X_train, X_val, y_train, y_val = {}, {}, {}, {}
        col_names = y_columns[sheet_name] if dataset == "08_CornGrain" else y_columns["ALL_FOSS"]

        print("\tTarget columns:", col_names)
        for col_idx, y_col in enumerate(col_names):
            # print("\t\tGetting values for target:", y_col)

            if y_col not in y_columns["DATASET"]:
                y_col_act = col_names_switch[y_col]
            else:
                y_col_act = y_col
            
            # Training set
            X_train[y_col_act], y_train[y_col_act] = get_x_y_labels(train_data, y_col)

            # Validation set
            if dataset == "08_CornGrain":
                X_val[y_col_act], y_val[y_col_act] = get_x_y_labels(val_data, y_col if sheet_name != "DATASET" else y_columns["VALID"][col_idx])
            else:
                X_val[y_col_act], y_val[y_col_act] = get_x_y_labels(val_data, y_col)
            # remove all rows with 0 in y_train and y_val
            mask_train = y_train[y_col_act] != 0
            mask_val = y_val[y_col_act] != 0

            # Count non-zero values in each target column
            # print(f"\t\t\tNon-zero values in training set for {y_col_act}: {len(y_train[y_col_act])}/{len(train_data)}")
            # print(f"\t\t\tNon-zero values in validation set for {y_col_act}: {len(y_val[y_col_act])}/{len(val_data)}")
            
            # Beams grouping
            X_train[y_col_act] = group_wavelengths(X_train[y_col_act], beams_step)
            X_val[y_col_act] = group_wavelengths(X_val[y_col_act], beams_step)

            # Apply masks to filter out zero values. Only if non-zero values > half of total
            if len(y_train[y_col_act]) > len(train_data) / 2:
                X_train[y_col_act] = X_train[y_col_act][mask_train]
                y_train[y_col_act] = y_train[y_col_act][mask_train]
                X_val[y_col_act] = X_val[y_col_act][mask_val]
                y_val[y_col_act] = y_val[y_col_act][mask_val]
        return X_train, X_val, y_train, y_val
            

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train models on NIR data.")
    parser.add_argument("--beams_step", type=int, default=1, help="Step size for grouping wavelengths. Use 1 for no grouping.")
    
    args = parser.parse_args()
    
    if args.beams_step < 1:
        raise ValueError("beams_step must be greater than or equal to 1.")

    sheet_names = ["DATASET"]#, "ALL_FOSS", "ALL_Gonzaga", "ALL_LUFA"]
    # sheet_names = ["ALL_FOSS", "ALL_Gonzaga", "ALL_LUFA"]

    # Save everything related to this training to a csv file
    output_csv = os.path.join(out_path, "outputs", "multi_dataset_training_kan")
    os.makedirs(output_csv, exist_ok=True)
    
    X_train_all, X_val_all, y_train_all, y_val_all = {}, {}, {}, {}
    for dir in os.listdir(in_path):
        dataset_name = dir.split('_')[1][:-len('Grain')]
        for file in os.listdir(os.path.join(in_path, dir)):
            if file.endswith("DATASET.xlsx"):
                print(f"\nProcessing dataset: {dataset_name}, file: {file}")
                X_train_all[dataset_name], X_val_all[dataset_name], y_train_all[dataset_name], y_val_all[dataset_name] = main(dir, os.path.join(in_path, dir, file), args.beams_step)
    
    for y_col in y_train_all[list(y_train_all.keys())[0]].keys():
        print(f"\nTraining combined models for target: {y_col}")
        X_train_combined = pd.concat([X_train_all[dataset_name][y_col] for dataset_name in X_train_all])
        y_train_combined = pd.concat([y_train_all[dataset_name][y_col] for dataset_name in y_train_all])
        X_val_combined = pd.concat([X_val_all[dataset_name][y_col] for dataset_name in X_val_all])
        y_val_combined = pd.concat([y_val_all[dataset_name][y_col] for dataset_name in y_val_all])

        # Add dataset classification labels
        train_dirs = pd.Series([dataset_name for dataset_name in X_train_all for _ in range(len(X_train_all[dataset_name][y_col]))], name="dataset_label")
        val_dirs = pd.Series([dataset_name for dataset_name in X_val_all for _ in range(len(X_val_all[dataset_name][y_col]))], name="dataset_label")
        
        # Train regression and classification models
        print(f"\nTraining combined SVM model for target: {y_col}")
        train_svm_regr(X_train_combined, X_val_combined, y_train_combined, y_val_combined, 
                       y_col, output_csv, args.beams_step)

        # Combine dataset labels with features
        X_train_combined["dataset_label"] = train_dirs.values
        X_val_combined["dataset_label"] = val_dirs.values

        # Train dataset classification model 
        # X_train_combined does not change based on y_col, so we can train once per y_col (take DM)
        if y_col == "DM" or y_col == "SS": 
            print(f"\nTraining SVM model for dataset classification (target: {y_col})")
            train_svm_classif(X_train_combined.drop(columns=["dataset_label"]), X_val_combined.drop(columns=["dataset_label"]),
                    train_dirs, val_dirs, f"_fam_classif", output_csv, args.beams_step)
