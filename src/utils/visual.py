import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def pca_transform(X, n_components=10):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    # Save explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    return pd.DataFrame(X_pca, index=X.index), explained_variance


def tsne_transform(X, n_components=2, perplexity=30, random_state=42):
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    X_tsne = tsne.fit_transform(X)
    return pd.DataFrame(X_tsne, index=X.index)

# TODO: implement FPCA transform
def fpca_transform(X, n_components=10):
    # Placeholder implementation
    raise NotImplementedError("FPCA transform is not implemented yet.")


def mahalanobis_transform(X):
    from scipy.spatial.distance import mahalanobis
    from numpy.linalg import inv

    # X_train shape: (n_train, 71)
    X_values = X.values
    mu = np.mean(X_values, axis=0)              # media spettrale
    cov = np.cov(X_values, rowvar=False)       # matrice di covarianza
    cov_inv = inv(cov + 1e-6*np.eye(cov.shape[0]))  # regolarizzazione

    # distanza di Mahalanobis per ciascun campione
    dist_train = np.array([mahalanobis(x, mu, cov_inv) for x in X_values])
    print("Statistiche Mahalanobis (train):", np.mean(dist_train), np.std(dist_train))

    # dividi lo spettro in segmenti e calcola la distanza di Mahalanobis per ciascun segmento
    segment_size = 71 // 10  # 10 segmenti # TODO: parametrize

    return pd.DataFrame(dist_train, index=X.index, columns=["Mahalanobis_Distance"])


def plot_wavelengths(X, outpath, dataset, sheet_name):
    plt.figure(figsize=(12, 6))
    for i in range(min(10, X.shape[0])):  # plot first 10 samples
        plt.plot(X.columns.astype(float), X.iloc[i, :], alpha=0.7)
    plt.title(f'Scattered NIR Spectra - {dataset} - {sheet_name}')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Absorbance')
    plt.grid()
    plt.savefig(os.path.join(outpath, f"spectrum_{dataset}.png"))
    plt.close()


def plot_predictions(y_true, y_pred, y_col, model_type, dataset, outpath, 
                     num_feat_selected=None, beams_step=1):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(y_true)), y_true, color='blue', label='True Values', alpha=0.6)
    plt.plot(range(len(y_pred)), y_pred, color='red', label='Predicted Values', alpha=0.6)
    title = f'{model_type.capitalize()} ({dataset}) Predictions vs True Values for {y_col}'
    title += f' (Beams Step: {beams_step})' if beams_step > 1 else ''
    if num_feat_selected is not None:
        title += f' (Features Selected: {num_feat_selected})'
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel(y_col)
    plt.legend()
    plt.grid()
    file_name = f"{y_col}_{model_type.capitalize()}_{dataset}{'_each' + str(beams_step) if beams_step > 1 else ''}.png"
    plt.savefig(os.path.join(outpath, file_name))
    plt.close()
    print(f"Plot saved: {os.path.join(outpath, file_name)}")


def plot_cm(cm, model, classes, acc, outpath=''):
    # Plot confusion matrix with absolute numbers in each cell
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix ({model} - Acc: {acc:.2f})')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(outpath, f"{model}_Confusion_Matrix.png"), dpi=300)
    plt.close()