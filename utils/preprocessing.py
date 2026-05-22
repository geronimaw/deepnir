import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.base import TransformerMixin, BaseEstimator

# Group wavelenghts
def group_wavelengths(data, beams_step):
    """
    Groups wavelengths in the dataset based on the beams_step value.
    """
    if beams_step == 1:
        return data  # No grouping, use all wavelengths as-is
    
    data = data.select_dtypes(include="number")

    # Group columns by step
    grouped_data = pd.DataFrame()
    num_wavelengths = data.shape[1]
    for i in range(0, num_wavelengths, beams_step):
        group_cols = data.iloc[:, i:i + beams_step]
        grouped_data[f"group_{i // beams_step}"] = group_cols.mean(axis=1)
    return grouped_data

# Spectral Normalization Variate (SNV): normalizes each spectrum to mean=0, std=1
class SNV(BaseEstimator, TransformerMixin):
    """Standard Normal Variate per spettri: normalizza ogni spettro (riga) a mean=0, std=1."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X assumed shape (n_samples, n_wavelengths)
        Xc = X.copy().astype(np.float64)
        mean = Xc.to_numpy().mean(axis=1, keepdims=True) if isinstance(Xc, pd.DataFrame) else Xc.mean(axis=1, keepdims=True)
        std = Xc.to_numpy().std(axis=1, keepdims=True) if isinstance(Xc, pd.DataFrame) else Xc.std(axis=1, keepdims=True)
        return (Xc - mean) / (std + 1e-8)

# Savitzky-Golay Smoothing / Derivative: smooth data
class SavitzkyGolaySmooth(BaseEstimator, TransformerMixin):
    """Applica Savitzky-Golay smoothing / derivata su ogni spettro."""
    def __init__(self, window_length=7, polyorder=2, deriv=0):
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.array(X)  # Ensure input is a NumPy array
        if X.ndim != 2:
            raise ValueError("Input data must be a 2D array where each row is a spectrum.")
        
        # Check if window_length is valid for all spectra
        if X.shape[1] < self.window_length:
            raise ValueError(f"Each spectrum must have at least {self.window_length} data points.")

        # Apply Savitzky-Golay filter to each spectrum
        Xf = np.array([
            savgol_filter(spectrum, 
                          window_length=self.window_length, 
                          polyorder=self.polyorder, 
                          deriv=self.deriv,
                          axis=-1,
                          mode='interp')
            for spectrum in X
        ])
        return Xf

# (Optional) Multiplicative Scatter Correction (MSC): corrects scatter effects
class MSC(BaseEstimator, TransformerMixin):
    """Multiplicative Scatter Correction."""
    def fit(self, X, y=None):
        # calcola lo spettro di riferimento come media degli spettri
        self.ref_spectrum_ = np.mean(X, axis=0)
        return self

    def transform(self, X):
        # Ensure X is a numerical array
        if not np.issubdtype(X.dtype, np.number):
            raise ValueError("Input data X must be numerical. Found non-numerical values.")

        X_msc = X.copy().astype(np.float64)
        ref = self.ref_spectrum_
        ref_mean = ref.mean()
        ref_centered = ref - ref_mean
        out = np.zeros_like(X_msc)
        for i, spec in enumerate(X_msc):
            # Ensure spec is numerical
            if isinstance(spec, str):
                raise ValueError(f"Row {i} contains non-numerical data: {spec}")

            # Linear regression spec ~ ref_spectrum
            slope, intercept = np.polyfit(ref_centered, spec - spec.mean(), 1)
            # Correction
            out[i, :] = (spec - intercept) / (slope + 1e-8)
        return out

# Plot wavelengths and mean spectrum
def plot_wavelengths(X_train, out_path, dataset, sheet_name):
    # Load training and validation data from Excel sheets
    wavelengths = X_train.columns.astype(float)
    mean_spectrum = X_train.mean().values
    std_spectrum = X_train.std().values

    # Plot wavelengths
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, mean_spectrum, label="Mean Spectrum", color="blue")
    plt.fill_between(wavelengths, mean_spectrum - std_spectrum, mean_spectrum + std_spectrum, color="blue", alpha=0.2, label="±1 Std Dev")
    plt.title(f"Mean NIR Spectrum with Standard Deviation - {dataset.replace('_val', ' (val)')} {sheet_name}")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.legend()
    plt.grid()
    plot_path = os.path.join(out_path, "plots")
    os.makedirs(plot_path, exist_ok=True)
    plt.savefig(os.path.join(plot_path, f"{dataset}_{sheet_name}_mean_spectrum.png"))
    plt.close()