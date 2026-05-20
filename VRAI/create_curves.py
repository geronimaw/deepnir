"""
For each regression model, plot: x (sample from the dataset) vs y_true and y_pred"
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from utils import train_xgb, train_svm, y_columns, get_x_y_labels

def create_curves(dataset, sheet_name, nutrient, model_type="xgb", scale=False):
    # Load training and validation data from Excel sheets
    path = "/mnt/c/Users/aless/Desktop/ricerca/deep-nir/data/raw/Grain"
    # file_name is the file inside os.path.join(path, dataset) that ends with "DATASET.xlsx"
    file_name = [f for f in os.listdir(os.path.join(path, dataset)) if f.endswith("DATASET.xlsx")][0] 
    data = pd.read_excel(os.path.join(path, dataset, file_name), sheet_name=sheet_name)
    val_data = pd.read_excel(os.path.join(path, dataset, file_name), sheet_name="VALID")
    val_data.columns = val_data.columns.map(str)
    data.columns = data.columns.map(str)
    
    # Get dtrain and dtest 
    train_data = data.sample(frac=0.7, random_state=42)
    val_data = data.drop(train_data.index)
        
    # Train a different model for each target
    for col_idx, y_col in enumerate(y_columns[sheet_name]):
        print("\nCreating curves for target:", y_col)
        
        # Training set
        X_train, y_train = get_x_y_labels(train_data, y_col)

        # Validation set
        X_val, y_val = get_x_y_labels(val_data, y_col if sheet_name != "DATASET" else y_columns["VALID"][col_idx])
        
        # remove all rows with 0 in y_train
        mask = y_train != 0
        X_train = X_train[mask]
        y_train = y_train[mask]

        # Train model
        if model_type == "xgb":
            model = train_xgb(X_train, X_val, y_val, y_col, "", scale)
        else:
            model = train_svm(X_train, y_train, X_val, y_val, y_col, "", scale)
        
        # Predict on validation set
        y_pred = model.predict(X_val)
        
        # Plot curves
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(y_val)), y_val, color='blue', label='True Values', alpha=0.6)
        plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted Values', alpha=0.6)
        plt.title(f'{model_type.upper()} Predictions vs True Values for {y_col} ({sheet_name})')
        plt.xlabel('Sample Index')
        plt.ylabel(y_col)
        plt.legend()
        plt.grid()
        
        # Save plot
        plt.savefig(os.path.join(models_path, f"{dataset}_{sheet_name}_{nutrient}_curves.png"))
        plt.close()

if __name__ == "__main__":
    models_path = "/mnt/c/Users/aless/Desktop/ricerca/deep-nir/src/deep_nir/VRAI/outputs"
    
    for model in os.listdir(models_path):
        print("Processing model:", model)
        dataset = model.split("_")[0] + "_" + model.split("_")[1]
        sheet_name = model.split("_")[2]
        nutrient = model.split("_")[3]
        model_type = "xgb" if model.split("_")[4] == "XGB" else "svm"
        create_curves(dataset, sheet_name, nutrient, model_type=model_type, scale=True)