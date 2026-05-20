# Plot scattered IR light sampled at various wavelengths
import os
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from deep_nir.VRAI.utils.data import y_columns, get_x_y_labels
from deep_nir.VRAI.utils.visual import plot_wavelengths

HERE = Path(__file__).parent
ROOT = HERE.parents[2]

out_path = ROOT / "src" / "deep_nir" / "VRAI"
in_path = ROOT / "data" / "raw" / "Grain"

def main(dataset, path):
    # Load training and validation data from Excel sheets
    for sheet_name in sheet_names:
        if sheet_name == "DATASET":
            data = pd.read_excel(path, sheet_name=sheet_name)
            val_data = pd.read_excel(path, sheet_name="VALID")
            val_data.columns = val_data.columns.map(str)
        else:
            data = pd.read_excel(path, sheet_name=sheet_name, header=1)
            val_data = None
        # print(f"Getting sheet: {sheet_name}")
        data.columns = data.columns.map(str)
        
        # Save everything related to this training to a csv file
        output_csv = os.path.join(out_path, "data_inspection")
        os.makedirs(output_csv, exist_ok=True)

        # Get dtrain and dtest 
        train_data = data
        if val_data is None:
            train_data = data.sample(frac=0.7, random_state=42)
            val_data = data.drop(train_data.index)
            
        # Get wavelengths data
        # Training set
        X_train, y_train = get_x_y_labels(train_data, y_columns[sheet_name][0])

        # Validation set
        X_val, y_val = get_x_y_labels(val_data, y_columns["VALID"][0])

        plot_wavelengths(X_train, output_csv, f"{dataset.split('_')[1][:-len('Grain')]}", sheet_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train models on NIR data.")
    parser.add_argument("--model", type=str, choices=["xgb", "svm"], default="xgb", help="Type of model to train.")
    
    args = parser.parse_args()

    merge_plots = True  # whether to merge all plots into a single one
    
    sheet_names = ["DATASET"]#, "ALL_FOSS", "ALL_Gonzaga", "ALL_LUFA"]
    # sheet_names = ["ALL_FOSS", "ALL_Gonzaga", "ALL_LUFA"]
    
    if not merge_plots:
        for dir in os.listdir(in_path):
            # if dir == "08_CornGrain":
                for file in os.listdir(os.path.join(in_path, dir)):
                    if file.endswith("DATASET.xlsx"):
                        print(f"Processing dataset: {dir}, file: {file}")
                        main(dir, path=os.path.join(in_path, dir, file))
    else:
        # Merge all plots into a single folder
        spectra = {}
        for dir in os.listdir(in_path):
            for file in os.listdir(os.path.join(in_path, dir)):
                if file.endswith("DATASET.xlsx"):
                    print(f"Processing dataset: {dir}, file: {file}")
                    for sheet_name in sheet_names:
                        if sheet_name == "DATASET":
                            data = pd.read_excel(os.path.join(in_path, dir, file), sheet_name=sheet_name)
                            val_data = pd.read_excel(os.path.join(in_path, dir, file), sheet_name="VALID")
                            val_data.columns = val_data.columns.map(str)
                        # else:
                        #     data = pd.read_excel(os.path.join(in_path, dir, file), sheet_name=sheet_name, header=1)
                        #     val_data = None
                        data.columns = data.columns.map(str)
                        
                        # Get dtrain and dtest 
                        train_data = data
                        if val_data is None:
                            train_data = data.sample(frac=0.7, random_state=42)
                            val_data = data.drop(train_data.index)
                            
                        # Get wavelengths data
                        # Training set
                        X_train, y_train = get_x_y_labels(train_data, y_columns[sheet_name][0])

                        # Validation set
                        X_val, y_val = get_x_y_labels(val_data, y_columns["VALID"][0])

                        spectra[dir.split('_')[1][:-len('Grain')]] = X_train

        # Now plot all spectra in a single plot
        # plot wavelengths as mean with std shading
        plt.figure(figsize=(12, 6))
        for dataset, X in spectra.items():
            mean_spectrum = X.mean()
            std_spectrum = X.std()
            plt.plot(X.columns.astype(float), mean_spectrum, label=dataset)
            plt.fill_between(X.columns.astype(float), mean_spectrum - std_spectrum, mean_spectrum + std_spectrum, alpha=0.2)
        plt.title(f'Scattered NIR Spectra - Merged Datasets')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Absorbance')
        plt.legend()
        plt.grid()
        output_csv = os.path.join(out_path, "data_inspection")
        os.makedirs(output_csv, exist_ok=True)
        plt.savefig(os.path.join(output_csv, f"spectrum_merged_datasets.png"))
        plt.close()