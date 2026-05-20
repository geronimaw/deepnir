import os
import sys
import numpy as np
import pandas as pd
from joblib import load
import matplotlib.pyplot as plt
sys.path.append("../")

# Caricamento dei dati di input (lunghezze d'onda)
def carica_dati_input(input_path):
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
def esegui_previsioni(inputs, models, sample_n):
    preds = [{} for _ in range(sample_n)]

    for sensor_id, sensor_data in inputs.items():
        # for pred_dict in preds:
        #     pred_dict[sensor_id] = {}

        for idx, id_scan in enumerate(sorted(sensor_data.keys())):
            values = sensor_data[id_scan]
            input_values = np.array(values).reshape(1, -1)

            sample_idx = idx // 10

            if sensor_id not in preds[sample_idx]:
                preds[sample_idx][sensor_id] = []

            input_values = np.array(values).reshape(1, -1)

            # 🔥 SKIP righe zero
            if np.all(np.array(values) == 0):
                continue

            scan_result = {
                model_name: model.predict(input_values)[0]
                for model_name, model in models.items()
            }

            preds[sample_idx][sensor_id].append(scan_result)

    return preds

# Percorsi dei file
input_path = "../../../data/raw/Grain/"
model_path = "./outputs/xgb_trained/"
out_folder = "./outputs/xgb_robustness/"

# Esecuzione
for fold in os.listdir(input_path):
    print(f"\n\nEseguo per {fold}")
    out_fold = os.path.join(out_folder, fold, "Repeatability1")
    os.makedirs(out_fold, exist_ok=True)
    
    nome = "Ripetibilit…1_sameCUP"

    inputs = carica_dati_input(os.path.join(input_path, fold, nome))
    sensor_list = list(inputs.keys())
    n_exp = len(inputs[sensor_list[0]])
    n_samples = n_exp // 10
    
    models = carica_modelli(model_path)
    preds = esegui_previsioni(inputs, models, n_samples)

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
                # model_preds = [pred_dict[sensor][id_scan][model] for id_scan in pred_dict[sensor].keys()]
                model_preds = [scan[model] for scan in pred_dict[sensor]]
                means[ii].append(np.mean(model_preds))
                stds[ii].append(np.std(model_preds))

                row = {
                    "Model": model,
                    "Sensor": sensor,
                    "Sample": f"Sample {ii+1}"
                }
                for j, pred in enumerate(model_preds):
                    row[f"Prediction {j}"] = pred
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
