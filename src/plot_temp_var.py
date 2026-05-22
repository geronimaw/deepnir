import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

temp_df = pd.read_excel('/Users/u0186653/Desktop/research/dingen/raw/Grain/08_CornGrain/08-MG_Rev489_RESULTS_TEMPVAR.xlsx', sheet_name='Foglio1', usecols='B', skiprows=7, nrows=7)
temperatures = temp_df.iloc[:, 0].tolist()

# Example usage
fold = "/Users/u0186653/Desktop/research/dingen/raw/Grain/08_CornGrain/Temperatura/nstd"
sensor_files = os.listdir(fold)   

# spectra = pd.read_csv(os.path.join(fold, sensor_files[0]), header=None, sep="\t")
# print(type(spectra), spectra.shape)
# exit()

for sensor_file in sensor_files:
    # Read spectral data (assuming each row is a spectrum)
    spectra = pd.read_csv(os.path.join(fold, sensor_file), header=None, sep="\t")  # Shape: (N_spectra, wavelengths)
    
    plt.figure(figsize=(10, 6))
    for i in range(len(temperatures)):
        plt.plot(spectra.iloc[i, 1:58], label=f'{temperatures[i]}°C')
    plt.title(f'Spectra - {sensor_file}')
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True)
    plt.show()