import os
import sys
import numpy as np
import pandas as pd
from joblib import load
import matplotlib.pyplot as plt
sys.path.append("../")

# Caricamento dei dati di input (lunghezze d'onda)
def carica_dati_input(input_path):
    if not os.path.exists(input_path):
        input_path = input_path.replace("Temperature", "Temperatura")
    if not os.path.exists(input_path):
        print(f"ERRORE, NON ESISTE IL PERCORSO SPECIFICATO ({input_path})")
    inputs = {}
    for txt in os.listdir(input_path):
        if txt.endswith(".txt"):
            with open(os.path.join(input_path, txt), "r") as f:
                lines = f.readlines()
                sensor_id = txt.split("_")[1].split(".")[0]
                inputs[sensor_id] = {}
                for line in lines:
                    line_tab = line.strip().split("\t")
                    inputs[sensor_id][int(line_tab[0])] = [float(value) for value in line_tab[1:]]
    return inputs

# Caricamento dei modelli
def carica_modelli(model_path):
    models = {}
    for model_file in os.listdir(model_path):
        if model_file.endswith(".joblib"):
            model_name = model_file.split("_")[0]
            model_full_path = os.path.join(model_path, model_file)
            models[model_name] = load(model_full_path)
    return models

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

# Percorsi dei file
input_path = "../../../data/raw/Grain/"
model_path = "./outputs/xgb_trained/"
out_folder = "./outputs/xgb_robustness/"

# Esecuzione
for fold in os.listdir(input_path):
    out_fold = os.path.join(out_folder, fold, "Temperature Variation")
    os.makedirs(out_fold, exist_ok=True)

    # Carica il file Excel
    nome_file = [fil for fil in os.listdir(os.path.join(input_path, fold)) if fil.endswith("RESULTS.xlsx")][0]
    file_path = os.path.join(input_path, fold, nome_file)  # Sostituisci con il percorso del tuo file
    sheet_name = "Temperature Variation"

    # Leggi il foglio Excel
    df = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")

    # Estrai i nomi dei sensori (celle B7:E7)
    nomi_sens = [str(nome) for nome in df.iloc[5, 1:5].values.tolist()]  # Riga 7 (indice 6), colonne B:E (indice 1:5)

    # Estrai i dati per sample_1 (celle B8:E21)
    sample_1 = df.iloc[6:13, 1:5].values.tolist()
    sample_2 = df.iloc[13:20, 1:5].values.tolist()
    
    # Crea il dizionario samples_temp
    samples_temp = {
        "nomi_sens": nomi_sens,
        "sample_1": sample_1,
        "sample_2": sample_2
    }
    
    nome = "Temperatura"
    if fold != "08_CornGrain":
        sample_3 = df.iloc[20:27, 1:5].values.tolist()
        samples_temp["sample_3"] = sample_3
        nome = "Temperature"

    inputs = carica_dati_input(os.path.join(input_path, fold, nome))
    models = carica_modelli(model_path)
    preds = esegui_previsioni(inputs, models, samples_temp)

    sensor_list = list(preds[0].keys())
    sample_names = [f"Sample {i+1}" for i in range(len(preds))]

    x = np.arange(len(sensor_list))
    width = 0.8 / len(preds) 

    all_data = []
    for model in models.keys():
        # Inizializza le liste per medie e std per ogni sample
        means = [[] for _ in range(len(preds))]
        stds = [[] for _ in range(len(preds))]

        for sensor in sensor_list:
            for ii, pred_dict in enumerate(preds):
                model_preds = [pred_dict[sensor][id_scan][model] for id_scan in pred_dict[sensor].keys()]
                means[ii].append(np.mean(model_preds))
                stds[ii].append(np.std(model_preds))

                row = {
                    "Model": model,
                    "Sensor": sensor,
                    "Sample": f"Sample {ii+1}"
                }
                for j, pred in enumerate(model_preds):
                    row[f"Prediction @ T{j}"] = pred
                all_data.append(row)
        
        # Crea un DataFrame
        df = pd.DataFrame(all_data)

        # Salva come CSV
        df.to_csv(os.path.join(out_fold, "model_predictions.csv"), index=False)

        # Plotta i bar per ogni sample
        for ii in range(len(preds)):
            plt.bar(x + (ii - len(preds)/2 + 0.5) * width, means[ii], width,
                    yerr=stds[ii], label=sample_names[ii])

        plt.xlabel('Sensors')
        plt.ylabel('Predictions')
        plt.title(f'Predictions by Sensor for Model: {model}')
        plt.xticks(x, sensor_list)
        plt.legend()
        plt.savefig(os.path.join(out_fold, f"distrib_pred_{model}.png"))
        plt.close()
