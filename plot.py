import pandas as pd
import matplotlib.pyplot as plt
from params import opts


# Data extracted from the document for each dataset and ACC@1, filling missing data with 30
data = {
    'BreastMNIST': {
        'Model': ['VGG19', 'Res.50', 'Dense.121', 'Eff.V2M', 'OpenCLIP', 'UNI', 'MedCLIP', 'BioMedCLIP'],
        '28x28': [79.48, 79.48, 80.12, 78.20, 87.18, 78.21, 76.28, 81.41],
        '64x64': [82.05, 80.76, 82.05, 75.64, 85.90, 80.77, 75.64, 83.33],
        '128x128': [80.76, 83.97, 87.82, 78.84, 84.62, 83.97, 82.69, 87.18],
        '224x224': [86.53, 83.33, 84.61, 81.41, 85.90, 82.05, 77.56, 86.54]
    },
    'PneumoniaMNIST': {
        'Model': ['VGG19', 'Res.50', 'Dense.121', 'Eff.V2M', 'OpenCLIP', 'UNI', 'MedCLIP', 'BioMedCLIP'],
        '28x28': [83.17, 82.85, 83.49, 77.08, 82.53, 80.93, 80.61, 85.74],
        '64x64': [82.37, 86.21, 84.77, 75.00, 84.94, 84.29, 84.78, 88.46],
        '128x128': [81.25, 84.29, 84.29, 78.04, 84.13, 87.02, 92.63, 88.94],
        '224x224': [82.37, 87.01, 87.5, 81.08, 86.06, 89.90, 93.59, 89.74]
    },
    'RetinaMNIST': {
        'Model': ['VGG19', 'Res.50', 'Dense.121', 'Eff.V2M', 'OpenCLIP', 'UNI', 'MedCLIP', 'BioMedCLIP'],
        '28x28': [46.25, 42.25, 45.25, 42.75, 45.25, 43.00, 33.00, 46.75],
        '64x64': [44.25, 48.5, 44.25, 45.00, 46.00, 46.75, 40.00, 49.50],
        '128x128': [47.75, 46.75, 50.00, 44.25, 49.00, 48.25, 42.50, 50.75],
        '224x224': [46.00, 48.00, 49.75, 48.75, 51.50, 47.75, 43.50, 54.00]
    },
    'DermaMNIST': {
        'Model': ['VGG19', 'Res.50', 'Dense.121', 'Eff.V2M', 'OpenCLIP', 'UNI', 'MedCLIP', 'BioMedCLIP'],
        '28x28': [67.73, 67.88, 67.83, 64.13, 72.17, 68.93, 63.79, 71.12],
        '64x64': [70.72, 72.16, 69.72, 64.93, 79.60, 78.65, 67.68, 75.46],
        '128x128': [71.82, 76.40, 73.11, 69.57, 81.30, 79.30, 68.23, 77.36],
        '224x224': [74.71, 77.75, 74.36, 70.92, 81.55, 80.15, 70.57, 77.41]
    },
    'BloodMNIST': {
        'Model': ['VGG19', 'Res.50', 'Dense.121', 'Eff.V2M', 'OpenCLIP', 'UNI', 'MedCLIP', 'BioMedCLIP'],
        '28x28': [73.07, 64.98, 69.95, 56.47, 77.87, 79.48, 55.74, 77.81],
        '64x64': [71.03, 66.93, 73.16, 63.98, 89.71, 95.09, 71.91, 89.27],
        '128x128': [74.86, 79.50, 78.74, 76.90, 89.51, 96.11, 81.29, 90.88],
        '224x224': [81.58, 87.69, 84.41, 81.29, 90.44, 96.78, 86.20, 91.49]
    },
    'PathMNIST': {
        'Model': ['VGG19', 'Res.50', 'Dense.121', 'Eff.V2M', 'OpenCLIP', 'UNI', 'MedCLIP', 'BioMedCLIP'],
        '28x28': [68.67, 69.74, 71.43, 64.24, 81.25, 90.86, 67.56, 81.69],
        '64x64': [76.40, 79.83, 81.89, 73.21, 89.76, 96.20, 75.72, 90.50],
        '128x128': [80.52, 84.40, 84.19, 79.33, 89.65, 96.36, 75.06, 92.34],
        '224x224': [81.10, 85.18, 85.01, 79.61, 86.78, 96.04, 73.75, 92.14]
    },
    'AdrenalMNIST3D': {
        'Model': ['VGG19', 'Res.50', 'Dense.121', 'Eff.V2M', 'OpenCLIP', 'UNI', 'MedCLIP', 'BioMedCLIP'],
        '28x28': [75.17, 71.48, 71.81, 71.47, 71.14, 68.79, 69.13, 69.13],
        '64x64': [72.48, 69.80, 72.82, 62.75, 70.13, 63.67, 68.46, 64.09],
        '128x128': [30, 30, 30, 30, 30, 30, 30, 30],
        '224x224': [30, 30, 30, 30, 30, 30, 30, 30]
    },
    'SynapseMNIST3D': {
        'Model': ['VGG19', 'Res.50', 'Dense.121', 'Eff.V2M', 'OpenCLIP', 'UNI', 'MedCLIP', 'BioMedCLIP'],
        '28x28': [65.62, 67.05, 65.06, 69.32, 65.62, 61.65, 61.93, 65.62],
        '64x64': [64.49, 67.05, 73.01, 62.21, 67.90, 71.31, 68.18, 62.78],
        '128x128': [30, 30, 30, 30, 30, 30, 30, 30],
        '224x224': [30, 30, 30, 30, 30, 30, 30, 30]
    }
}

# Convert to DataFrame
dfs = {key: pd.DataFrame(value).set_index('Model') for key, value in data.items()}

# Plotting the data and saving the plots
for key, df in dfs.items():
    plt.figure(figsize=(12, 8))
    ax = df.plot(kind='bar', ylim=(30, 100))
    # plt.title(f'ACC@1 for {key} using Different Models and Image Sizes')
    plt.xlabel('Model')
    plt.ylabel('ACC@1')
    plt.xticks(rotation=0, fontsize=6)  # Make x-axis font size smaller
    plt.grid(axis='y')

    # Place legend outside the plot
    plt.legend(title='Image Size', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Save the plot as a .png file
    plt.tight_layout()
    plt.savefig(opts['save_figures'] + f'{key}_ACC@1.png', dpi=300)
    plt.show()