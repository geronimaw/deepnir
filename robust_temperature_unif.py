import os
import math
import numpy as np
import pandas as pd

from shap_analysis import get_models, MODELS_PATH, ROOT

DATA_PATH = "/Users/u0186653/Desktop/research/dingen/raw/Grain"
model_name = "svm"
training_set = '58freqs'
OUT_PATH = ROOT / "outputs" / model_name

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

    # Check if the unified model exists
    unified_model_name = "multi_dataset_training"
    has_unified_model = unified_model_name in models

    for crop in inputs.keys():
        print(f"\nCrop: {crop}")
        
        # model_ = None
        # model_name_used = None

        # # 1. Try to find a crop-specific model first
        # for m_name in models.keys():
        #     # Skip the unified model name if we are looking for a specific one, 
        #     # or check if it matches the crop name pattern
        #     if m_name == unified_model_name:
        #         continue
        #     if m_name in crop:
        #         model_name_used = m_name
        #         break
        
        # # 2. If no specific model found, or if we explicitly want to test the unified model on this crop
        # #    (Assuming the requirement is: "If no specific model, use unified" OR "Always test unified if requested")
        # #    Based on your prompt, it seems you want to evaluate the unified model on these crops.
        
        # if model_name_used is None and has_unified_model:
        model_name_used = unified_model_name
        model_ = models[unified_model_name]
        print(f"\t-> Using UNIFIED model '{unified_model_name}'")
        # elif model_name_used and model_name_used in models:
        #     model_ = models[model_name_used]
        #     print(f"\t-> Using specific model '{model_name_used}'")
        # else:
        #     print(f"\t-> WARNING: No model found for {crop}. Skipping.")
        #     continue

        if model_ is None:
            continue

        # Data for this specific crop
        input_data = inputs[crop]
        temperature_data = temperatures[crop]
        
        # Get reference residuals for this specific crop (from the first loop logic)
        # Note: ref_residuals was defined inside the loop in the original code snippet.
        # We need to re-extract it here or ensure it's stored. 
        # Let's re-extract it from the file to be safe, assuming the file structure is the same.
        # (Repeating the extraction logic from your first loop for safety)
        file_path = [fil for fil in os.listdir(os.path.join(DATA_PATH, crop)) if fil.endswith("RESULTS.xlsx")]
        sheet_name = "Temperature Variation"
        df_crop = pd.read_excel(os.path.join(DATA_PATH, crop, file_path[0]), sheet_name=sheet_name, engine="openpyxl")
        ref_residuals = df_crop.iloc[28, 1:5].values.tolist() # Row 29 (index 28), Cols B:E

        sample_n = 0
        preds = {}
        residuals = {}
        avg_residuals = {}
        final_eval[crop] = {}
        
        # Count samples from temperature data
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
                        preds[nutr] [sensor_id] [sample_idx].append(prediction.item())

        # Calculate residuals (mean subtraction per sample)
        for nutr, preds_ in preds.items():
            residuals[nutr] = {}
            for sensor_id, preds__ in preds_.items():
                residuals[nutr] [sensor_id] = {}

                for sample_idx, preds___ in preds__.items():
                    residuals[nutr] [sensor_id] [sample_idx] = {}
                    values = preds___ # List of predictions for this sample
                    
                    if not values:
                        continue
                        
                    mean_val = sum(values) / len(values)
                    for i, val in enumerate(values):
                        residuals[nutr] [sensor_id] [sample_idx] [i] = val - mean_val

        # Average residuals across samples
        for nutr, residuals_ in residuals.items():
            avg_residuals[nutr] = {}
            for sensor_id, residuals__ in residuals_.items():
                if not residuals__:
                    continue

                # Determine max scan index
                scan_indices = []
                for sample_dict in residuals__.values():
                    scan_indices.extend(sample_dict.keys())
                
                if not scan_indices:
                    continue
                    
                num_scans = max(scan_indices) + 1

                scan_averages = {}
                for scan_idx in range(num_scans):
                    values_to_average = []
                    for sample_idx, scan_dict in residuals__.items():
                        if scan_idx in scan_dict:
                            values_to_average.append(scan_dict[scan_idx])
                    
                    if values_to_average:
                        scan_averages[scan_idx] = np.mean(values_to_average)
                    else:
                        scan_averages[scan_idx] = 0.0

                avg_residuals[nutr] [sensor_id] = [scan_averages[i].item() for i in range(num_scans)]

        # Final evaluation: (Max - Min) / ref_residuals
        for nutr, avg_residuals_ in avg_residuals.items():
            final_eval[crop] [nutr] = {}
            for idx, (sensor_id, avg_residuals__) in enumerate(avg_residuals_.items()):
                if not avg_residuals__:
                    final_eval[crop] [nutr] [sensor_id] = 0.0
                    continue
                
                if idx < len(ref_residuals) and ref_residuals[idx] != 0:
                    range_val = max(avg_residuals__) - min(avg_residuals__)
                    final_eval[crop] [nutr] [sensor_id] = range_val / ref_residuals[idx]
                else:
                    final_eval[crop] [nutr] [sensor_id] = 0.0

    # ... (Rest of the code: creating DataFrame and saving to Excel remains the same)
    rows = []
    for crop, nutrients in final_eval.items():
        for nutr, sensors in nutrients.items():
            for sensor_id, value in sensors.items():
                rows.append({
                    "Crop": crop,
                    "Nutriment": nutr,
                    "Sensor_ID": sensor_id,
                    "Value": value
                })

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["Crop", "Nutriment", "Sensor_ID"]).reset_index(drop=True)

    # Ensure OUT_PATH is defined
    df.to_excel(OUT_PATH / "rip_temperatura_unif.xlsx", index=False)
    print(df.head())