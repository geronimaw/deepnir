import os
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[2]

cartella_input = ROOT / "src/deep_nir/VRAI/outputs_noPrep"

for dir in os.listdir(cartella_input):
    dir_path = os.path.join(cartella_input, dir)
    csv_files = [f for f in os.listdir(dir_path) if f.lower().endswith(".csv")]
    
    if csv_files:  # Only create the ExcelWriter if there are CSV files
        file_output = os.path.join(dir_path, f"{dir}_merged_results.xlsx")
        with pd.ExcelWriter(file_output, engine="openpyxl") as writer:
            for filename in csv_files:
                filepath = os.path.join(dir_path, filename)
                
                # Read the CSV
                df = pd.read_csv(filepath)
                
                # Transpose the DataFrame
                df_transposed = df.T
                
                # Sheet name = file name without extension, max 31 characters
                sheet_name = os.path.splitext(filename)[0][:31]
                
                # Write the transposed table to the Excel file
                df_transposed.to_excel(writer, sheet_name=sheet_name, header=False)

        print("File Excel generato:", file_output)

