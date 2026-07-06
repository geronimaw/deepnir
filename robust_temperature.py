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
    
    # Get predictions
    final_eval = {}

    for crop in inputs.keys():
        print("\nCrop:", crop)
        if crop == "multi_dataset_training": # esiste sia in models.keys() che in input.keys(). TODO temperature
            continue
        
        for model_name in models.keys():
            if model_name in crop:
                crop_model_name = model_name
        model_ = models[crop_model_name]
        input = inputs[crop]
        temperature = temperatures[crop]

        sample_n = 0
        preds = {}

        for k in temperature.keys():
            if "sample" in k: 
                # preds[sample_n] = {}
                sample_n += 1
        residuals = {}
        avg_residuals = {}
        final_eval[crop] = {}
        
        for nutr, mod in model_.items():

            preds[nutr] = {}
            if mod is None:
                continue

            print(f"\tPredicting for {nutr}")

            for sensor_id, sensor_data in input.items():
                preds[nutr][sensor_id] = {}


                for idx, (id_scan, values) in enumerate(sensor_data.iterrows()):

                    cropped_values = values.iloc[:58].values 
                    input_values = cropped_values.reshape(1, -1)
                    
                    sample_idx = idx // 7
                    if sample_idx not in preds[nutr][sensor_id]:
                        preds[nutr][sensor_id][sample_idx] = []

                    if sample_idx < sample_n:

                        # Perform prediction
                        prediction = mod.predict(input_values)
                        
                        # Store result
                        preds[nutr][sensor_id][sample_idx].append(prediction[0].item())
            
        # Average across temperatures (id_scan)
        for nutr, preds_ in preds.items():
            
            residuals[nutr] = {}
            for sensor_id, preds__ in preds_.items():
                residuals[nutr][sensor_id] = {}

                for sample_idx, preds___ in preds__.items():
                    residuals[nutr][sensor_id][sample_idx] = {}
                    values = []

                    for id_scan, value in enumerate(preds___):
                        values.append(value)
                    
                    for i, val in enumerate(values):
                        residuals[nutr][sensor_id][sample_idx][i] = val - sum(values) / len(values)
        
        # print("\t\t", residuals)

        # Average residuals across sample_idx
        for nutr, residuals_ in residuals.items():
            
            avg_residuals[nutr] = {}
            for sensor_id, residuals__ in residuals_.items():
                # avg_residuals[nutr][sensor_id] = []
                
                num_scans = max(max(inner_dict.keys()) for inner_dict in residuals__.values()) + 1

                scan_averages = {}
                # Itera su ogni indice di scan
                for scan_idx in range(num_scans):
                    values_to_average = []
                    
                    # Raccogli i valori per questo scan_idx da tutti i sample (chiavi 0, 1, ...)
                    for sample_idx in residuals__.keys():
                        if scan_idx in residuals__[sample_idx]:
                            values_to_average.append(residuals__[sample_idx][scan_idx])
                    
                    if values_to_average:
                        # Calcola la media
                        avg_val = np.mean(values_to_average)
                        scan_averages[scan_idx] = avg_val
                    else:
                        scan_averages[scan_idx] = 0.0

                avg_residuals[nutr][sensor_id] = [scan_averages[i].item() for i in range(num_scans)]
        # print("\n\t\t", avg_residuals)
        
        # Final repeatability evaluation
        for nutr, avg_residuals_ in avg_residuals.items():
            
            final_eval[crop][nutr] = {}
            for idx, (sensor_id, avg_residuals__) in enumerate(avg_residuals_.items()):
                final_eval[crop][nutr][sensor_id] = ((max(avg_residuals__) - min(avg_residuals__)) / ref_residuals[idx])

    # print(final_eval)

    rows = []
    # Itera attraverso le nidificazioni
    for crop, nutrients in final_eval.items():
        for nutr, sensors in nutrients.items():
            for sensor_id, value in sensors.items():
                rows.append({
                    "Crop": crop,
                    "Nutriment": nutr,
                    "Sensor_ID": sensor_id,
                    "Value": value
                })

    # Crea il DataFrame
    df = pd.DataFrame(rows)

    # Ordinamento (opzionale ma consigliato per leggibilità)
    # Ordina prima per Crop, poi per Nutriment, poi per Sensor_ID
    df = df.sort_values(by=["Crop", "Nutriment", "Sensor_ID"]).reset_index(drop=True)

    # Salva in Excel
    output_file = "final_eval_results.xlsx"
    df.to_excel(OUT_PATH / "rip_temperatura.xlsx", index=False)

    print(f"File salvato con successo in: {output_file}")
    print(df.head())