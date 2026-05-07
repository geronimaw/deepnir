import os
import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

model = 'xgb'
trained_path = f"./trained_{model}/"
perf_path = f"./outputs/{model}_posthoc/"

os.makedirs(perf_path, exist_ok=True)

selected_freqs = {}
for fil in os.listdir(trained_path):
    if fil.endswith(".csv"):
        with open(os.path.join(trained_path, fil), newline='') as csvfil:
            spamreader = csv.reader(csvfil, delimiter=',')
            for row in spamreader:
                selected_freqs[fil[:fil.find("_")]] = [int(freq) for freq in row]

# tutte le lunghezze uniche
all_wl = sorted(set(sum(selected_freqs.values(), [])))

# matrice
mat = pd.DataFrame(0, index=selected_freqs.keys(), columns=all_wl)

for task, wl in selected_freqs.items():
    mat.loc[task, wl] = 1

plt.figure(figsize=(10,6))
sns.heatmap(mat, cmap="viridis", cbar=False)
plt.xlabel("Wavelength index")
plt.ylabel("Task")
plt.savefig(os.path.join(perf_path, "selected_freqs.png"))
plt.close()

# Frequenza di selezion
from collections import Counter

counts = Counter(sum(selected_freqs.values(), []))
plt.bar(counts.keys(), counts.values())
plt.xlabel("Wavelength index")
plt.ylabel("Times selected")
plt.savefig(os.path.join(perf_path, "selected_freqs_freq.png"))


# Jaccard similarity index tra task
from itertools import combinations

def jaccard(a,b):
    a=set(a); b=set(b)
    return len(a&b)/len(a|b)

tasks=list(selected_freqs.keys())

rows = []

for t1, t2 in combinations(tasks, 2):
    score = jaccard(selected_freqs[t1], selected_freqs[t2])
    rows.append({
        "task1": t1,
        "task2": t2,
        "jaccard": score
    })

df = pd.DataFrame(rows)

# ordina decrescente
df = df.sort_values("jaccard", ascending=False)

# salva su csv
df.to_csv(os.path.join(perf_path, "jaccard_similarity_tasks.csv"), index=False)
print(df)