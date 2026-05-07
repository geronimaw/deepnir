import os
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from deep_nir.VRAI.utils.visual import pca_transform, tsne_transform, fpca_transform, mahalanobis_transform

HERE = Path(__file__).parent
ROOT = HERE.parents[2]

out_path = ROOT / "src/deep_nir/VRAI"
in_path = ROOT / "data/raw/Grain"

def principal_components(path, sheet_names, technique, n_components=10):
    # Load training and validation data from Excel sheets
    for sheet_name in sheet_names:
        if sheet_name == "DATASET":
            data = pd.read_excel(path, sheet_name=sheet_name)
            val_data = pd.read_excel(path, sheet_name="VALID")
            val_data.columns = val_data.columns.map(str)
        else:
            data = pd.read_excel(path, sheet_name=sheet_name, header=1)
            val_data = None
        data.columns = data.columns.map(str)

        # Get dtrain and dtest 
        train_data = data
        if val_data is None:
            train_data = data.sample(frac=0.7, random_state=42)
            val_data = data.drop(train_data.index)

        # Training set
        x_columns = train_data.loc[:, "1100":"1800"].columns.tolist()
        X_train = train_data[x_columns]

        # Validation set
        X_val = val_data[x_columns]

        # Standardize data before PCA/TSNE/FPCA
        X_train = (X_train - X_train.mean()) / X_train.std()
        X_val = (X_val - X_val.mean()) / X_val.std()

        # Merge train and val for PCA
        X_combined = pd.concat([X_train, X_val])
    
        if technique == "pca":
            # PCA transform to compare families
            return pca_transform(X_combined, n_components=n_components)
        elif technique == "tsne":
            # TSNE transform to compare families
            return tsne_transform(X_combined, n_components=n_components)
        elif technique == "mahalanobis":
            # Mahalanobis distance matrix
            return mahalanobis_transform(X_combined)
        else:
            # FPCA transform to compare families
            return fpca_transform(X_combined, n_components=n_components)
       
            

if __name__ == "__main__":

    techniques = ["pca", "tsne", "fpca", "mahalanobis"]

    for technique in techniques:

        if technique == "fpca":
            # not implemented yet, skip
            print("FPCA not implemented yet, skipping...")
            continue
        
        print(f"Using {technique.upper()} with 10 components.")
        n_components = 10 if technique in ["pca", "fpca"] else 2

        sheet_names = ["DATASET"]

        # Save everything related to this training to a csv file
        output_csv = os.path.join(out_path, "data_inspection")
        os.makedirs(output_csv, exist_ok=True)
        
        components, explained_variance = {}, {}
        
        for dir in os.listdir(in_path):
            for file in os.listdir(os.path.join(in_path, dir)):
                if file.endswith("DATASET.xlsx"):
                    dataset = dir.split("_", 1)[1]
                    print(f"Processing dataset: {dataset}, file: {file}")
                    output = principal_components(os.path.join(in_path, dir, file), sheet_names,
                                                technique, n_components)
                    if "pca" in technique:
                        components[dataset], explained_variance[dataset] = output[0], output[1]
                    else:
                        components[dataset] = output
        
        if technique != "mahalanobis":
            # Compare PCAs across datasets: plot all PC1 vs PC2 and color by dataset
            print(f"Plotting {technique.upper()} comparisons...")

            plt.figure(figsize=(10, 8))
            for dataset, comp_data in components.items():
                # debug print shape of pca_data
                print(f"{technique.upper()} data shape for {dataset}: {comp_data.shape}")
                plt.scatter(comp_data.iloc[:, 0], comp_data.iloc[:, 1], label=dataset, alpha=0.6)
            plt.xlabel(f"{technique.upper()} Component 1")
            plt.ylabel(f"{technique.upper()} Component 2")
            plt.title(f"{technique.upper()} Comparison Across Datasets")
            # Add text with explained variance for PCA
            if "pca" in technique:
                for dataset, var in explained_variance.items():
                    plt.text(0.05, 0.95 - 0.05 * list(explained_variance.keys()).index(dataset),
                            f"{dataset} Explained Variance: {var[0]:.2f}, {var[1]:.2f}",
                            transform=plt.gca().transAxes)
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(output_csv, f"{technique}_comparison.png"))
            plt.close()

            # Compute distance between PCA centroids of each dataset
            centroids = {}
            for dataset, comp_data in components.items():
                centroids[dataset] = comp_data.mean().values
            import numpy as np
            from scipy.spatial.distance import pdist, squareform
            centroid_matrix = np.array(list(centroids.values()))
            distances = squareform(pdist(centroid_matrix))
            distance_df = pd.DataFrame(distances, index=centroids.keys(), columns=centroids.keys())
            distance_df.to_csv(os.path.join(output_csv, f"{technique}_centroid_distances.csv"))
            print(f"{technique.upper()} centroid distances saved at {os.path.join(output_csv, f'{technique}_centroid_distances.csv')}.")
        else:
            # For Mahalanobis, save distance matrices for each dataset
            for dataset, dist_matrix in components.items():
                dist_df = pd.DataFrame(dist_matrix)
                dist_df.to_csv(os.path.join(output_csv, f"{technique}_{dataset}_distance_matrix.csv"))
                print(f"{technique.upper()} distance matrix for {dataset} saved at {os.path.join(output_csv, f'{technique}_{dataset}_distance_matrix.csv')}.")