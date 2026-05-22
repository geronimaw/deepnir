# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from glob import glob

# # Set up plotting style
# sns.set_style("whitegrid")
# metrics_dir = '/Users/u0186653/Desktop/research/dingen/deepnir/src/outputs/xgb'  # Root folder

# # Components to analyze
# components = [
#     'ADF', 'Ash', 'Crude Fat', 'Crude Fib.', 'DM', 'NDF', 'Protein', 'Starch'
# ]

# # Metrics to extract
# metric_labels = {
#     'val_R2': 'R²',
#     'val_SEP': 'SEP',
#     'val_std_SEP': 'stdSEP',
#     'val_Bias': 'Bias'
# }

# # Collect available components
# available_components = []

# # Prepare subplots
# fig, axes = plt.subplots(len(components), len(metric_labels), figsize=(18, 4 * len(components)))
# if len(components) == 1:
#     axes = axes.reshape(1, -1)

# for idx, comp in enumerate(components):
#     csv_pattern = f"{comp.replace(' ', '_').replace('.', '')}_regr_XGB_.csv"
#     all_data = []

#     # Walk through the directory tree
#     for freq_type in ['58freqs', 'all_freqs']:
#         # Single crops
#         for crop in ['Barley', 'Corn', 'Rapeseed', 'Soybean', 'Wheat']:
#             csv_file = os.path.join(metrics_dir, 'performance', freq_type, crop, csv_pattern)
#             if os.path.exists(csv_file):
#                 df = pd.read_csv(csv_file)
#                 df['Crop'] = crop
#                 df['Freq'] = freq_type
#                 all_data.append(df)

#         # Multi-dataset subfolders
#         multi_path = os.path.join(metrics_dir, 'performance', freq_type, 'multi_dataset_training')
#         if os.path.exists(multi_path):
#             for subfolder in os.listdir(multi_path):
#                 sub_path = os.path.join(multi_path, subfolder)
#                 if os.path.isdir(sub_path):
#                     csv_file = os.path.join(sub_path, csv_pattern)
#                     if os.path.exists(csv_file):
#                         df = pd.read_csv(csv_file)
#                         df['Crop'] = f"Multi_{subfolder}"
#                         df['Freq'] = freq_type
#                         all_data.append(df)

#     if not all_data:
#         # Hide axes if no data
#         for j in range(3):
#             axes[idx, j].set_visible(False)
#         continue

#     available_components.append(comp)
#     data = pd.concat(all_data, ignore_index=True)
#     suffix = comp.split()[-1]  # Handle 'Crude Fib.' → 'Fib'
#     data['val_SEP'] = data[f'val_SEP_{suffix}']
#     data['val_std_SEP'] = data[f'val_std_SEP_{suffix}']
#     data['val_Bias'] = data[f'val_Bias_{suffix}']
#     data['val_R2'] = data[f'val_R2_{suffix}']

#     for j, (metric, label) in enumerate(metric_labels.items()):
#         sns.boxplot(data=data, x='Crop', y=metric, hue='Freq', ax=axes[idx, j])
#         axes[idx, j].set_title(f'{comp} - {label}')
#         axes[idx, j].tick_params(axis='x', rotation=45)

# # Hide unused subplots and adjust layout
# if len(available_components) < len(components):
#     fig.suptitle(f"Regression Metrics ({len(available_components)}/{len(components)} components)", fontsize=16)
# else:
#     fig.suptitle("Regression Performance Metrics by Crop and Frequency Type", fontsize=16)

# plt.tight_layout()
# plt.show()



# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# sns.set_style("whitegrid")
# metrics_dir = '/Users/u0186653/Desktop/research/dingen/deepnir/src/outputs/xgb'
# components = ['ADF', 'Ash', 'Crude Fat', 'Crude Fib.', 'DM', 'NDF', 'Protein', 'Starch']

# for comp in components:
#     csv_pattern = f"{comp.replace(' ', '_').replace('.', '')}_regr_XGB_.csv"
#     all_data = []

#     for freq_type in ['58freqs', 'all_freqs']:
#         # Single crops
#         for crop in ['Barley', 'Corn', 'Rapeseed', 'Soybean', 'Wheat']:
#             csv_file = os.path.join(metrics_dir, 'performance', freq_type, crop, csv_pattern)
#             if os.path.exists(csv_file):
#                 df = pd.read_csv(csv_file)
#                 df['Crop'] = crop
#                 df['Freq'] = freq_type
#                 all_data.append(df)

#         # Multi-dataset subfolders
#         multi_path = os.path.join(metrics_dir, 'performance', freq_type, 'multi_dataset_training')
#         if os.path.exists(multi_path):
#             for subfolder in os.listdir(multi_path):
#                 sub_path = os.path.join(multi_path, subfolder)
#                 if os.path.isdir(sub_path):
#                     csv_file = os.path.join(sub_path, csv_pattern)
#                     if os.path.exists(csv_file):
#                         df = pd.read_csv(csv_file)
#                         df['Crop'] = f"Multi_{subfolder}"
#                         df['Freq'] = freq_type
#                         all_data.append(df)

#     if not all_data:
#         print(f"No data found for {comp}")
#         continue

#     data = pd.concat(all_data, ignore_index=True)
#     suffix = comp.split()[-1]  # e.g., 'ADF' from 'ADF', 'Fib.' from 'Crude Fib.'
#     suffix = suffix.rstrip('.')  # Remove trailing dot

#     # Extract metrics
#     data['R²'] = data[f'val_R2_{suffix}']
#     data['SEP'] = data[f'val_SEP_{suffix}']
#     data['std/SEP'] = data[f'val_std_SEP_{suffix}']
#     data['Bias'] = data[f'val_Bias_{suffix}']

#     # Melt for plotting
#     df_melted = data.melt(id_vars=['Crop', 'Freq'], value_vars=['R²', 'SEP', 'std/SEP', 'Bias'],
#                           var_name='Metric', value_name='Value')

#     # Plot
#     plt.figure(figsize=(10, 6))
#     sns.boxplot(data=df_melted, x='Crop', y='Value', hue='Freq', palette='Set2')
#     plt.title(f'{comp} - Performance Metrics by Crop and Frequency Type')
#     plt.ylabel(comp)
#     plt.xticks(rotation=45)
#     plt.legend(title='Frequency Type')
#     plt.tight_layout()
#     plt.show()   


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

metrics_dir = '/Users/u0186653/Desktop/research/dingen/deepnir/src/outputs/xgb'
components = ['ADF', 'Ash', 'Crude Fat', 'Crude Fib.', 'DM', 'NDF', 'Protein', 'Starch']

for comp in components:
    csv_pattern = f"{comp.replace(' ', '_').replace('.', '')}_regr_XGB_.csv"
    all_data = []

    for freq_type in ['58freqs', 'all_freqs']:
        for crop in ['Barley', 'Corn', 'Rapeseed', 'Soybean', 'Wheat']:
            csv_file = os.path.join(metrics_dir, 'performance', freq_type, crop, csv_pattern)
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                df['Crop'] = crop
                df['Freq'] = freq_type
                all_data.append(df)

        multi_path = os.path.join(metrics_dir, 'performance', freq_type, 'multi_dataset_training')
        if os.path.exists(multi_path):
            for subfolder in os.listdir(multi_path):
                sub_path = os.path.join(multi_path, subfolder)
                if os.path.isdir(sub_path):
                    csv_file = os.path.join(sub_path, csv_pattern)
                    if os.path.exists(csv_file):
                        df = pd.read_csv(csv_file)
                        df['Crop'] = f"Multi_{subfolder}"
                        df['Freq'] = freq_type
                        all_data.append(df)

    if not all_data:
        print(f"No data found for {comp}")
        continue

    data = pd.concat(all_data, ignore_index=True)
    suffix = comp.split()[-1].rstrip('.')
    r2_col = f'val_R2_{suffix}'
    sep_col = f'val_SEP_{suffix}'
    std_sep_col = f'val_std_SEP_{suffix}'
    bias_col = f'val_Bias_{suffix}'

    # Normalize: higher R² is good, lower SEP/Bias are good
    data['R²'] = data[r2_col]
    data['SEP'] = 1 - (data[sep_col] / data[sep_col].max())  # Invert
    # data['std/SEP'] = 1 - (data[std_sep_col] / data[std_sep_col].max())  # Invert
    data['Bias'] = 1 - (data[bias_col].abs() / data[bias_col].abs().max())  # Invert

    # # After computing std/SEP
    # threshold_raw = 3.0
    # # Cap values below threshold (or set to NaN to exclude)
    # data['std/SEP_raw'] = data[std_sep_col]
    # # # Invert after thresholding: higher is better
    # # data['std/SEP'] = np.where(
    # #     data['std/SEP_raw'] >= threshold_raw,
    # #     1 - (data[std_sep_col] / data[std_sep_col].max()),
    # #     0  # Below threshold → 0
    # # )

    # Group by Crop and Freq
    # grouped = data.groupby(['Crop', 'Freq'])[['R²', 'SEP', 'std/SEP_raw', 'std/SEP', 'Bias']].mean().reset_index()
    grouped = data.groupby(['Crop', 'Freq'])[['R²', 'SEP', 'Bias']].mean().reset_index()
    categories = ['R²', 'SEP', 'Bias']

    # Plot
    for (crop, freq), group in grouped.groupby(['Crop', 'Freq']):
        values = group[categories].values.flatten()
        values = np.append(values, values[0])  # Close the circle

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.fill(angles, values, color='red', alpha=0.25)
        ax.plot(angles, values, color='red', linewidth=2)

        # Add value labels
        values_flat = [group['R²'].iloc[0], group['SEP'].iloc[0], group['Bias'].iloc[0]]
        for angle, value, label in zip(angles[:-1], values_flat, categories):
            if label == "std/SEP":
                ax.text(angle, 0.9, f'{value:.3f}', ha='center', va='bottom', fontsize=10, color='red')
            else:
                ax.text(angle, value + 0.05, f'{value:.3f}', ha='center', va='bottom', fontsize=10, color='red')
        
        # threshold_norm = 1 - (threshold_raw / data[std_sep_col].max())
        # angles_thresh = [angles[2]] * 2  # Assuming 'std/SEP' is 3rd metric
        # values_thresh = [0, threshold_norm]
        # ax.plot(angles_thresh, values_thresh, color='gray', linestyle='--', linewidth=2)
        # ax.text(angles[2], threshold_norm + 0.05, f'Thresh: {threshold_norm:.2f}', color='gray', ha='center')

        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        plt.title(f"{comp} - {crop} ({freq})")
        plt.tight_layout()
        plt.show()   