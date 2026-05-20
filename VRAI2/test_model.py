import os
import sys
import csv
import joblib
import pandas as pd

from joblib import dump
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from VRAI2.main_train_xgb_new import main
from VRAI2.utils.models import train_xgb_regr, train_xgb_class
from VRAI2.utils.data import y_columns, get_x_y_labels, col_names_switch

HERE = Path(__file__).parent
ROOT = HERE.parents[2]

out_path = ROOT / "src/deep_nir/VRAI2"
in_path = ROOT / "data/raw/Grain"

##################### CUSTOMIZE HERE
model_path = os.path.join(out_path, "trained_xgb/58freqs/multi_dataset_training_xgb/DM_regr.joblib")
train_on_X_first_freq = 58

loaded_model = joblib.load(model_path)

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
                dir, sheet_names, os.path.join(dir_path, file), 1, train_on_X_first_freq)

for dataset_name in X_val_all:
    predictions = loaded_model.predict(X_new)
