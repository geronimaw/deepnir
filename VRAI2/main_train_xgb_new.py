import os
import sys
import csv
import pandas as pd

from joblib import dump
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from VRAI2.utils.models import train_xgb_regr, train_xgb_class
from VRAI2.utils.data import y_columns, get_x_y_labels, col_names_switch

HERE = Path(__file__).parent
ROOT = HERE.parents[2]

out_path = ROOT / "src/deep_nir/VRAI2"
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

def main(dataset, sheet_names, path, beams_step):
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
    parser.add_argument("--multi_dataset", action="store_true", help="If set, train a single model on all datasets. Otherwise, train separate models for each directory.")

    args = parser.parse_args()

    if args.beams_step < 1:
        raise ValueError("beams_step must be greater than or equal to 1.")

    sheet_names = ["DATASET"]  # , "ALL_FOSS", "ALL_Gonzaga", "ALL_LUFA"

    X_train_all, X_val_all, y_train_all, y_val_all = {}, {}, {}, {}

    for dir in os.listdir(in_path):
        dataset_name = dir.split('_')[1][:-len('Grain')]
        dir_path = os.path.join(in_path, dir)
        if not os.path.isdir(dir_path):
            continue

        for file in os.listdir(dir_path):
            if file.endswith("DATASET.xlsx"):
                print(f"\nProcessing dataset: {dataset_name}, file: {file}")
                X_train_all[dataset_name], X_val_all[dataset_name], y_train_all[dataset_name], y_val_all[dataset_name] = main(
                    dir, sheet_names, os.path.join(dir_path, file), args.beams_step)

    if args.multi_dataset:
        print("---Training a unified model on all families---")
        # Define output paths based on the training mode
        output_csv = os.path.join(out_path, "outputs_std_PEC", "multi_dataset_training_xgb")
        os.makedirs(output_csv, exist_ok=True)
        out_models_path = os.path.join(out_path, "trained_xgb", "multi_dataset_training_xgb")
        os.makedirs(out_models_path, exist_ok=True)

        # Train a single model on all datasets
        for y_col in y_train_all[list(y_train_all.keys())[0]].keys():
            X_train_combined = pd.concat([X_train_all[dataset_name][y_col] for dataset_name in X_train_all])
            y_train_combined = pd.concat([y_train_all[dataset_name][y_col] for dataset_name in y_train_all])
            X_val_combined = pd.concat([X_val_all[dataset_name][y_col] for dataset_name in X_val_all])
            y_val_combined = pd.concat([y_val_all[dataset_name][y_col] for dataset_name in y_val_all])

            # Add dataset classification labels
            train_dirs = pd.Series(
                [dataset_name for dataset_name in X_train_all for _ in range(len(X_train_all[dataset_name][y_col]))],
                name="dataset_label"
            )
            val_dirs = pd.Series(
                [dataset_name for dataset_name in X_val_all for _ in range(len(X_val_all[dataset_name][y_col]))],
                name="dataset_label"
            )

            # Train for regression
            print(f"\nTraining combined XGBoost model for target: {y_col}")
            results = train_xgb_regr(
                X_train_combined, X_val_combined, y_train_combined, y_val_combined,
                y_col, output_csv, args.beams_step
            )

            model_out_path = os.path.join(out_models_path, f"{y_col}_regr.joblib")
            dump(results["model"], model_out_path)

            with open(os.path.join(out_models_path, f"{y_col}_regr_selected_freq.csv"), 'w') as fil:
                wr = csv.writer(fil)
                wr.writerow(results["selected_indices"])

            if y_col == "DM" or y_col == "SS":
                print("\nTraining combined XGBoost classification model")
                results = train_xgb_class(
                    X_train_combined, X_val_combined, train_dirs, val_dirs,
                    output_csv, args.beams_step
                )

                model_out_path = os.path.join(out_models_path, f"class.joblib")
                dump(results["model"], model_out_path)

                with open(os.path.join(out_models_path, f"class_selected_freq.csv"), 'w') as fil:
                    wr = csv.writer(fil)
                    wr.writerow(results["selected_indices"])
    else:
        print("---Training a  model for each family---")
        for dataset_name in X_train_all:
            # Define output paths based on the training mode
            output_csv = os.path.join(out_path, "outputs_std_PEC", dataset_name)
            os.makedirs(output_csv, exist_ok=True)
            out_models_path = os.path.join(out_path, "trained_xgb")
            os.makedirs(out_models_path, exist_ok=True)

            # Train separate models for each directory
            for y_col in X_train_all[dataset_name].keys():
                X_train = X_train_all[dataset_name][y_col]
                y_train = y_train_all[dataset_name][y_col]
                X_val = X_val_all[dataset_name][y_col]
                y_val = y_val_all[dataset_name][y_col]
                
                if X_train.empty or X_val.empty:
                    continue

                # Create a subdirectory for this dataset
                dataset_out_path = os.path.join(out_models_path, dataset_name)
                os.makedirs(dataset_out_path, exist_ok=True)

                # Train for regression
                print(f"\nTraining XGBoost model for dataset: {dataset_name}, target: {y_col}")
                results = train_xgb_regr(
                    X_train, X_val, y_train, y_val,
                    y_col, output_csv, args.beams_step
                )

                model_out_path = os.path.join(dataset_out_path, f"{y_col}_regr.joblib")
                dump(results["model"], model_out_path)

                with open(os.path.join(dataset_out_path, f"{y_col}_regr_selected_freq.csv"), 'w') as fil:
                    wr = csv.writer(fil)
                    wr.writerow(results["selected_indices"])

                # if y_col == "DM" or y_col == "SS":
                #     print(f"\nTraining XGBoost classification model for dataset: {dataset_name}")
                #     train_dirs = pd.Series([dataset_name] * len(X_train), name="dataset_label")
                #     val_dirs = pd.Series([dataset_name] * len(X_val), name="dataset_label")

                #     results = train_xgb_class(
                #         X_train, X_val, train_dirs, val_dirs,
                #         output_csv, args.beams_step
                #     )

                #     model_out_path = os.path.join(dataset_out_path, f"class.joblib")
                #     dump(results["model"], model_out_path)

                #     with open(os.path.join(dataset_out_path, f"class_selected_freq.csv"), 'w') as fil:
                #         wr = csv.writer(fil)
                #         wr.writerow(results["selected_indices"])