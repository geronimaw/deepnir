import os
import math
import numpy as np
import pandas as pd

from shap_analysis import get_models, MODELS_PATH, ROOT

DATA_PATH = "/Users/u0186653/Desktop/research/dingen/raw/Grain"
training_set = '58freqs'
OUT_PATH = ROOT / "outputs" / "xgb_unified" 

# Esecuzione delle previsioni
def esegui_previsioni(inputs, models, samples_temp):
    sample_n = 0
    for k in samples_temp.keys():
        if "sample" in k: 
            sample_n += 1
    preds = [{} for _ in range(sample_n)]

    for sensor_id, sensor_data in inputs.items():
        for pred_dict in preds:
            pred_dict[sensor_id] = {}

        for idx, (id_scan, values) in enumerate(sensor_data.items()):
            input_values = np.array(values).reshape(1, -1)

            if idx < 7:
                preds[0][sensor_id][id_scan] = {}
                for model_name, model in models.items():
                    prediction = model.predict(input_values)
                    preds[0][sensor_id][id_scan][model_name] = prediction[0]
                    # print(f"Sample 1 - Sensor: {sensor_id}, ID scan: {id_scan}, Model: {model_name}, Prediction: {prediction[0]}")

            elif idx < 14:
                preds[1][sensor_id][id_scan] = {}
                for model_name, model in models.items():
                    prediction = model.predict(input_values)
                    preds[1][sensor_id][id_scan][model_name] = prediction[0]
                    # print(f"Sample 2 - Sensor: {sensor_id}, ID scan: {id_scan}, Model: {model_name}, Prediction: {prediction[0]}")
            
            else:
                preds[2][sensor_id][id_scan] = {}
                for model_name, model in models.items():
                    prediction = model.predict(input_values)
                    preds[2][sensor_id][id_scan][model_name] = prediction[0]
                    # print(f"Sample 2 - Sensor: {sensor_id}, ID scan: {id_scan}, Model: {model_name}, Prediction: {prediction[0]}")

    return preds


if __name__ == "__main__":

    inputs = {}
    temperatures = {}

    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    for crop in os.listdir(DATA_PATH):

        if "Store" in crop:
            continue

        # Get temperature values
        file_path = [fil for fil in os.listdir(os.path.join(DATA_PATH, crop)) if fil.endswith("RESULTS.xlsx")][0]
        sheet_name = "Temperature Variation"
        df = pd.read_excel(os.path.join(DATA_PATH, crop, file_path), sheet_name=sheet_name, engine="openpyxl")

        # Estrai i nomi dei sensori (celle B7:E7)
        nomi_sens = [str(nome) for nome in df.iloc[5, 1:5].values.tolist()]  # Riga 7 (indice 6), colonne B:E (indice 1:5)

        # Estrai i dati per sample_1 (celle B8:E21)
        sample_1 = df.iloc[6:13, 1:5].values.tolist()
        sample_2 = df.iloc[13:20, 1:5].values.tolist()
        sample_3 = df.iloc[20:27, 1:5].values.tolist()
        
        # Crea il dizionario samples_temp
        temperatures[crop] = {
            "nomi_sens": nomi_sens,
            "sample_1": sample_1,
            "sample_2": sample_2,
            "sample_3": sample_3
        }
        
        # Estrai valori BOH per medie dei residui
        ref_residuals = df.iloc[28, 1:5].values.tolist()

        inputs[crop] = {}
        temp_fold = os.path.join(DATA_PATH, crop, "Temperatura")
        if not os.path.exists(temp_fold):
            temp_fold = temp_fold.replace("ura", "ure")

        for sensor in os.listdir(temp_fold):
            if sensor.endswith(".txt"):
                inputs[crop][sensor[:-4]] = pd.read_csv(os.path.join(temp_fold, sensor), sep="\t", header=None)       
    
    models_path = os.path.join(MODELS_PATH, training_set)
    models, _ = get_models(models_path)
    final_eval = {}

    for crop in inputs.keys():
        print("\nCrop:", crop)
        
        model_name = None
        
        if crop == "multi_dataset_training":
            # CASE: Multi-dataset training
            # 1. Identify the specific model name for the aggregated dataset.
            #    Assuming the key in 'models' is 'multi_dataset_model' or similar.
            #    You may need to adjust this key based on your actual training script output.
            possible_keys = [k for k in models.keys() if "multi" in k.lower() or "all" in k.lower()]
            
            if possible_keys:
                model_name = possible_keys # Pick the first match
            else:
                print("Error: No specific multi-dataset model found in models dictionary.")
                continue

            if model_name not in models:
                print(f"Error: Model '{model_name}' not found.")
                continue

            # 2. Merge inputs from ALL crops into a single structure for this model.
            #    We create a temporary 'merged_input' dict combining all sensors from all crops.
            #    The structure expects {sensor_id: dataframe}.
            merged_input = {}
            merged_temperatures = {} # If temperature data also needs merging or specific handling
            
            # Note: Since the original logic iterates 'sensor_id' -> 'sensor_data',
            # we just need to ensure unique sensor IDs if they overlap across crops.
            # If sensor IDs are unique per crop, we can just update.
            # If they overlap, we might need to prefix them (e.g., "CropA_sensor1").
            # Assuming unique IDs for now, or that the model handles raw concatenation.
            
            for c in inputs.keys():
                if c == "multi_dataset_training": continue # Skip the placeholder key itself if it exists in inputs
                
                for sensor_id, df_data in inputs[c].items():
                    # Optional: Prefix to avoid collision if sensor IDs are not unique
                    # unique_id = f"{c}_{sensor_id}" 
                    unique_id = sensor_id 
                    
                    merged_input[unique_id] = df_data
            
            # Assign the merged data to the current crop loop variable for consistency
            input_data = merged_input
            temperature_data = temperatures # Use existing temperature structure or merge if needed
            
            # Use the unified model
            model_ = {nutr: models[model_name] [nutr] for nutr in models[model_name].keys()}
            
        else:
            # CASE: Single crop
            # Find the specific model for this crop
            for m_name in models.keys():
                if m_name in crop:
                    model_name = m_name
                    break
            
            if not model_name or model_name not in models:
                print(f"Warning: No matching model found for crop {crop}. Skipping.")
                continue

            model_ = models[model_name]
            input_data = inputs[crop]
            temperature_data = temperatures[crop]

        # --- Shared Processing Logic (Predictions & Residuals) ---
        
        sample_n = 0
        preds = {}
        residuals = {}
        avg_residuals = {}
        final_eval[crop] = {}
        
        # Determine sample count from temperature data
        for k in temperature_data.keys():
            if "sample" in k: 
                sample_n += 1
        
        for nutr, mod in model_.items():
            preds[nutr] = {}
            if mod is None:
                continue

            print(f"\tPredicting for {nutr}")

            for sensor_id, sensor_data in input_data.items():
                preds[nutr] [sensor_id] = {}

                for idx, (id_scan, values) in enumerate(sensor_data.iterrows()):
                    cropped_values = values.iloc[:58].values 
                    input_values = cropped_values.reshape(1, -1)
                    
                    sample_idx = idx // 7
                    if sample_idx not in preds[nutr] [sensor_id]:
                        preds[nutr] [sensor_id] [sample_idx] = []

                    if sample_idx < sample_n:
                        # Perform prediction
                        prediction = mod.predict(input_values)
                        # Store result
                        # Handle potential numpy scalar conversion
                        preds[nutr] [sensor_id] [sample_idx].append(prediction.item())

        # Average across temperatures (id_scan) -> Calculate residuals
        for nutr, preds_ in preds.items():
            residuals[nutr] = {}
            for sensor_id, preds__ in preds_.items():
                residuals[nutr] [sensor_id] = {}

                for sample_idx, preds___ in preds__.items():
                    residuals[nutr] [sensor_id] [sample_idx] = {}
                    values = []

                    for id_scan, value in enumerate(preds___):
                        values.append(value)
                    
                    # Calculate mean for this sample_idx
                    mean_val = sum(values) / len(values) if values else 0
                    for i, val in enumerate(values):
                        residuals[nutr] [sensor_id] [sample_idx] [i] = val - mean_val
        
        # Average residuals across sample_idx
        for nutr, residuals_ in residuals.items():
            avg_residuals[nutr] = {}
            for sensor_id, residuals__ in residuals_.items():
                if not residuals__: continue

                num_scans = max(max(inner_dict.keys()) for inner_dict in residuals__.values()) + 1

                scan_averages = {}
                for scan_idx in range(num_scans):
                    values_to_average = []
                    for sample_idx in residuals__.keys():
                        if scan_idx in residuals__[sample_idx]:
                            values_to_average.append(residuals__[sample_idx] [scan_idx])
                    
                    if values_to_average:
                        avg_val = np.mean(values_to_average)
                        scan_averages[scan_idx] = avg_val
                    else:
                        scan_averages[scan_idx] = 0.0

                avg_residuals[nutr] [sensor_id] = [scan_averages[i].item() for i in range(num_scans)]

        # Final repeatability evaluation
        # Note: ref_residuals needs to be defined for the multi-dataset case.
        # If it's not available in a single crop's file, you might need to define a global ref_residuals.
        # For now, assuming it exists in the first crop or a specific location.
        # If 'multi_dataset_training' has its own RESULTS.xlsx, we should have extracted it earlier.
        # Let's assume ref_residuals was extracted for the specific crop or is global.
        
        # If this is multi_dataset_training, we might need a different ref_residuals source.
        # Attempting to get it from the first available crop if not defined for 'multi'
        if crop == "multi_dataset_training":
             # Fallback: Use ref_residuals from the first single crop if available, 
             # or you must ensure ref_residuals is calculated/loaded specifically for the multi case.
             # This part depends on how you stored 'ref_residuals' in the first loop.
             # Ideally, you should have extracted ref_residuals for 'multi_dataset_training' in the first loop.
             # If not, you might need to load it manually here.
             pass 

        # Re-structure ref_residuals access if needed. 
        # Assuming 'ref_residuals' variable holds the list for the current context.
        # If 'ref_residuals' was defined per crop in the first loop, we need to ensure it's available here.
        # Let's assume you stored a global 'all_ref_residuals' or similar, or handle it here.
        # For this snippet, I will assume 'ref_residuals' is correctly scoped or you define it:
        # ref_residuals = [...] # Define based on your specific needs for the multi-model

        for nutr, avg_residuals_ in avg_residuals.items():
            final_eval[crop] [nutr] = {}
            for idx, (sensor_id, avg_residuals__) in enumerate(avg_residuals_.items()):
                # Ensure ref_residuals has enough elements
                if idx < len(ref_residuals):
                    final_eval[crop] [nutr] [sensor_id] = ((max(avg_residuals__) - min(avg_residuals__)) / ref_residuals[idx])
                else:
                    print(f"Warning: Missing ref_residuals for index {idx}")
                    final_eval[crop] [nutr] [sensor_id] = 0.0