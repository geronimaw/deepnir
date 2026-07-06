import os
import sys
import shap
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# sys.modules['utils'] = deepnir_utils
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from deepnir.test import main

ROOT = Path(__file__).parent

training_set = '58freqs'
model_name = 'svm'

OUT_PATH = ROOT / "outputs"
DATA_PATH  = ROOT / ".." / "raw" / "Grain"
MODELS_PATH  = ROOT / "outputs" / model_name / "trained_models"
OUT_PATH = ROOT / "outputs" / model_name / "shap_analysis"

nutrients = ['ADF', 'Ash', 'Crude Fat', 'Crude Fib.', 'DM', 'NDF', 'Protein', 'Starch']


def get_models(root_dir):
    # Store all frequencies per nutrient
    models = {}
    sel_freqs = {}

    # Traverse directory tree
    for crop_folder in os.listdir(root_dir):
        if "Store" in crop_folder:
            continue

        models[crop_folder] = {}
        sel_freqs[crop_folder] = {}

        crop_path = os.path.join(root_dir, crop_folder)
        
        if not os.path.isdir(crop_path) or "DS_Store" in crop_path:
            continue

        for nutrient in nutrients:
            # Match CSV by nutrient in filename
            if os.path.exists(os.path.join(crop_path, f"{nutrient}_regr.joblib")):
                models[crop_folder][nutrient] = joblib.load(os.path.join(crop_path, f"{nutrient}_regr.joblib"))
            else:
                models[crop_folder][nutrient] = None
            if os.path.exists(os.path.join(crop_path, f"{nutrient}_regr_selected_freq.csv")):        
                sel_freqs[crop_folder][nutrient] = pd.read_csv(
                    os.path.join(crop_path, f"{nutrient}_regr_selected_freq.csv"), header=None, sep="\t")  # Shape: (N_spectra, wavelengths)
            else:
                sel_freqs[crop_folder][nutrient] = None

    # for k, v in models.items():
    #     for nutrient in v:
    #         print(k, "\n\t", nutrient, "\t", sel_freqs[k][nutrient])

    return models, sel_freqs

if __name__ == "__main__":

    sheet_names = ["DATASET"]

    X_val_all, y_val_all, X_train_all, y_train_all = {}, {}, {}, {}

    models_path = MODELS_PATH / training_set

    for dir_entry in DATA_PATH.iterdir():
        if not dir_entry.is_dir():
            continue

        # Robustly extract dataset name: strip leading digits + underscore, trailing "Grain"
        raw_name = dir_entry.name.split("_", 1)[-1]
        dataset_name = raw_name.removesuffix("Grain")

        for file in dir_entry.iterdir():
            if file.suffix == ".xlsx" and file.stem.endswith("DATASET"):
                print(f"\nProcessing dataset: {dataset_name}  |  file: {file.name}")
                (X_train_all[dataset_name],
                 X_val_all[dataset_name],
                 y_train_all[dataset_name],
                 y_val_all[dataset_name],
                ) = main(dir_entry.name, sheet_names, file, 1, 58 if '58' in training_set else -1)
    
    models, sel_freqs = get_models(models_path)

    # Identify if a unified model exists
    unified_model_name = "multi_dataset_training"
    has_unified_model = unified_model_name in models

    # Pre-calculate the list of all crops available in the loaded data
    all_crops = list(X_train_all.keys())
    
    for crop, nutrient in models.items():
        
        # Determine the output path
        output_path = OUT_PATH / training_set / crop
        output_path.mkdir(parents=True, exist_ok=True)
        
        crop_shap_data = {}
        features_names = None

        for nutr, model in nutrient.items():
            if model is None:
                continue
                
            print(f"\nProcessing SHAP for: {crop} | Nutrient: {nutr}")

            # --- DATA PREPARATION LOGIC ---
            if crop == unified_model_name and has_unified_model:
                # CASE: Unified Model
                # Concatenate all crops' data into single DataFrames
                X_background_list = []
                X_test_list = []
                
                print("\t-> Merging data from all crops for unified model...")
                
                for c in all_crops:
                    if nutr in X_train_all[c]:
                        X_background_list.append(X_train_all[c] [nutr])
                    if nutr in X_val_all[c]:
                        X_test_list.append(X_val_all[c] [nutr])
                
                if not X_background_list or not X_test_list:
                    print(f"\t-> ERROR: No data found for nutrient '{nutr}' in any crop. Skipping SHAP.")
                    continue

                # Concatenate along rows (axis=0)
                # This assumes all DataFrames have the same columns (features)
                X_background = pd.concat(X_background_list, ignore_index=True)
                X_test = pd.concat(X_test_list, ignore_index=True)
                
                # Ensure column order is consistent (optional but good practice)
                # X_background = X_background.sort_index(axis=1)
                # X_test = X_test.sort_index(axis=1)

            else:
                # CASE: Single Crop Model
                if crop not in X_train_all or nutr not in X_train_all[crop]:
                    print(f"\t-> WARNING: Data for {crop}/{nutr} not found. Skipping.")
                    continue

                X_background = X_train_all[crop] [nutr]
                X_test = X_val_all[crop] [nutr]
            
            X_test_selected = X_test
            feature_names_target = X_test.columns#[30:]
            
            # Calcolo dei valori SHAP
            if model_name == "xgb":
                best_model = model.best_estimator_.named_steps['model']
                explainer = shap.TreeExplainer(best_model, data=X_background)
            else:
                if hasattr(model, 'named_steps'):
                    best_model = model.named_steps['model']
                else:
                    # Fallback if it's a dict or other structure
                    best_model = model['model']
                
                selector = model.named_steps['band_select']

                selected_mask = selector.support_
                original_features = X_background.columns.tolist()
                X_background_selected = X_background.loc[:, selected_mask]
                X_test_selected = X_test.loc[:, selected_mask]

                feature_names_target = X_background_selected.columns.tolist()

                K = 100  # Target number of background samples
                if len(X_background_selected) > K:
                    print(f"\t-> Summarizing {len(X_background_selected)} background samples to {K} using kmeans...")
                    X_background_selected = shap.kmeans(X_background_selected, K)

                explainer = shap.KernelExplainer(best_model.predict, X_background_selected, nsamples=500)

            shap_values = explainer(X_test_selected)
            
            # Estrazione globale delle feature (dalla 30 in poi)
            shap_values_target = shap_values.values#[:, 30:]
            
            # --- PLOT INDIVIDUALE PER NUTRIENTE (FEATURE >= 30) ---
            if shap_values_target.shape[1] > 0:
                mean_abs_shap = np.mean(np.abs(shap_values_target), axis=0)
                
                importance_df = pd.DataFrame({
                    'Feature': feature_names_target,
                    'Importance': mean_abs_shap
                }).sort_values(by='Importance', ascending=True)
                
                plt.figure(figsize=(10, max(4, len(feature_names_target) * 0.3)))
                plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
                plt.xlabel('Mean |SHAP value| (Significatività)')
                plt.title(f'Importanza lambda | {crop} - {nutr}')
                plt.tight_layout()
                
                plt.savefig(output_path / f"shap_importance_{nutr}.png", dpi=150)
                plt.close()
                
                # --- ACCUMULO DATI PER HEATMAP (SOLO FEATURE >= 30) ---
                if features_names is None:
                    features_names = feature_names_target
                
                crop_shap_data[nutr] = mean_abs_shap
            else:
                print(f"[WARNING] {crop} - {nutr}: Il dataset ha meno di 31 feature.")

        # --- GENERAZIONE HEATMAP ACCORPATA (SOLO FEATURE >= 30) ---
        if crop_shap_data:
            print(f"Generazione heatmap per {crop}")
            df_heatmap = pd.DataFrame(crop_shap_data, index=features_names).T
            
            # Opzionale: decommenta la riga sotto se vuoi nascondere le feature che sono a zero per TUTTI i nutrienti
            # df_heatmap = df_heatmap.loc[:, (df_heatmap != 0).any(axis=0)]
            
            plt.figure(figsize=(max(12, df_heatmap.shape[1] * 0.4), max(5, df_heatmap.shape[0] * 0.8)))
            
            sns.heatmap(
                df_heatmap, 
                fmt=".4f",          
                cmap="YlGnBu",      
                cbar_kws={'label': 'Mean |SHAP value|'},
                linewidths=0.5
            )
            
            plt.title(f'Riepilogo Importanza lambda per {crop}', fontsize=14, pad=20)
            plt.xlabel('Features', fontsize=12)
            plt.ylabel('Nutrienti (nutr)', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            plt.savefig(str(output_path) + "_shap_heatmap.png", dpi=150)
            plt.close()