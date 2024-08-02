import pandas as pd
import matplotlib.pyplot as plt
from params import opts


# Data extracted from the document for each dataset and ACC@1, filling missing data with 30
data = {
    'BreastMNIST': {
        'Model': ['VGG19', 'Res.50', 'Dense.121', 'Eff.V2M', 'MedCLIP', 'BioMedCLIP', 'OpenCLIP', 'CONCH', 'UNI'],
        '28x28': [79.48, 79.48, 80.12, 78.20, 76.28, 81.41, 87.18, 79.49, 78.21],
        '64x64': [82.05, 80.76, 82.05, 75.64, 75.64, 83.33, 85.90, 75.64, 80.77],
        '128x128': [80.76, 83.97, 87.82, 78.84, 82.69, 87.18, 84.62, 83.97, 83.97],
        '224x224': [86.53, 83.33, 84.61, 81.41, 77.56, 86.54, 85.90, 87.18, 82.05]
    },
    'PneumoniaMNIST': {
        'Model': ['VGG19', 'Res.50', 'Dense.121', 'Eff.V2M', 'MedCLIP', 'BioMedCLIP', 'OpenCLIP', 'CONCH', 'UNI'],
        '28x28': [83.17, 82.85, 83.49, 77.08, 80.61, 85.74, 82.53, 83.01, 80.93],
        '64x64': [82.37, 86.21, 84.77, 75.00, 84.78, 88.46, 84.94, 83.97, 84.29],
        '128x128': [81.25, 84.29, 84.29, 78.04, 92.63, 88.94, 84.13, 86.70, 87.02],
        '224x224': [82.37, 87.01, 87.5, 81.08, 93.59, 89.74, 86.06, 84.46, 89.90]
    },
    'RetinaMNIST': {
        'Model': ['VGG19', 'Res.50', 'Dense.121', 'Eff.V2M', 'MedCLIP', 'BioMedCLIP', 'OpenCLIP', 'CONCH', 'UNI'],
        '28x28': [46.25, 42.25, 45.25, 42.75, 33.00, 46.75, 45.25, 44.50, 43.00],
        '64x64': [44.25, 48.5, 44.25, 45.00, 40.00, 49.50, 46.00, 47.00, 46.75],
        '128x128': [47.75, 46.75, 50.00, 44.25, 42.50, 50.75, 49.00, 46.25, 48.25],
        '224x224': [46.00, 48.00, 49.75, 48.75, 43.50, 54.00, 51.50, 44.75, 47.75]
    },
    'DermaMNIST': {
        'Model': ['VGG19', 'Res.50', 'Dense.121', 'Eff.V2M', 'MedCLIP', 'BioMedCLIP', 'OpenCLIP', 'CONCH', 'UNI'],
        '28x28': [67.73, 67.88, 67.83, 64.13, 63.79, 71.12, 72.17, 71.82, 68.93],
        '64x64': [70.72, 72.16, 69.72, 64.93, 67.68, 75.46, 79.60, 75.76, 78.65],
        '128x128': [71.82, 76.40, 73.11, 69.57, 68.23, 77.36, 81.30, 76.46, 79.30],
        '224x224': [74.71, 77.75, 74.36, 70.92, 70.57, 77.41, 81.55, 77.16, 80.15]
    },
    'BloodMNIST': {
        'Model': ['VGG19', 'Res.50', 'Dense.121', 'Eff.V2M', 'MedCLIP', 'BioMedCLIP', 'OpenCLIP', 'CONCH', 'UNI'],
        '28x28': [73.07, 64.98, 69.95, 56.47, 55.74, 77.81, 77.87, 85.56, 79.48],
        '64x64': [71.03, 66.93, 73.16, 63.98, 71.91, 89.27, 89.71, 93.57, 95.09],
        '128x128': [74.86, 79.50, 78.74, 76.90, 81.29, 90.88, 89.51, 94.36, 96.11],
        '224x224': [81.58, 87.69, 84.41, 81.29, 86.20, 91.49, 90.44, 93.92, 96.78]
    },
    'PathMNIST': {
        'Model': ['VGG19', 'Res.50', 'Dense.121', 'Eff.V2M', 'MedCLIP', 'BioMedCLIP', 'OpenCLIP', 'CONCH', 'UNI'],
        '28x28': [68.67, 69.74, 71.43, 64.24, 67.56, 81.69, 81.25, 84.72, 90.86],
        '64x64': [76.40, 79.83, 81.89, 73.21, 75.72, 90.50, 89.76, 92.48, 96.20],
        '128x128': [80.52, 84.40, 84.19, 79.33, 75.06, 92.34, 89.65, 94.96, 96.36],
        '224x224': [81.10, 85.18, 85.01, 79.61, 73.75, 92.14, 86.78, 94.96, 96.04]
    },
    'AdrenalMNIST3D': {
        'Model': ['VGG19', 'Res.50', 'Dense.121', 'Eff.V2M', 'MedCLIP', 'BioMedCLIP', 'OpenCLIP', 'CONCH', 'UNI'],
        '28x28': [75.17, 71.48, 71.81, 71.47, 69.13, 69.13, 71.14, 72.48, 68.79],
        '64x64': [72.48, 69.80, 72.82, 62.75, 68.46, 64.09, 70.13, 72.48, 63.67],
        '128x128': [30, 30, 30, 30, 30, 30, 30, 30, 30],
        '224x224': [30, 30, 30, 30, 30, 30, 30, 30, 30]
    },
    'SynapseMNIST3D': {
        'Model': ['VGG19', 'Res.50', 'Dense.121', 'Eff.V2M', 'MedCLIP', 'BioMedCLIP', 'OpenCLIP', 'CONCH', 'UNI'],
        '28x28': [65.62, 67.05, 65.06, 69.32, 61.93, 65.62, 65.62, 70.17, 61.65],
        '64x64': [64.49, 67.05, 73.01, 62.21, 68.18, 62.78, 67.90, 75.57, 71.31],
        '128x128': [30, 30, 30, 30, 30, 30, 30, 30, 30],
        '224x224': [30, 30, 30, 30, 30, 30, 30, 30, 30]
    }
}
# Convert to DataFrame
dfs = {key: pd.DataFrame(value).set_index('Model') for key, value in data.items()}

# Plotting the data and saving the plots
for key, df in dfs.items():
    plt.figure(figsize=(30, 10))
    ax = df.plot(kind='bar', ylim=(30, 100))
    # plt.title(f'ACC@1 for {key} using Different Models and Image Sizes')
    plt.title(key, fontsize=40)
    plt.xlabel('', fontsize=40)
    plt.ylabel('ACC@1', fontsize=40)
    plt.xticks(rotation=0, fontsize=30)
    plt.yticks(fontsize=30)  # Increase y-axis font size
    plt.grid(axis='y')

    # Customize legend for specific datasets
    if key in ['AdrenalMNIST3D', 'SynapseMNIST3D']:
        handles, labels = ax.get_legend_handles_labels()
        new_handles = [handles[i] for i in [0, 1]]  # Keeping only '28x28' and '64x64'
        new_labels = [labels[i] for i in [0, 1]]
        plt.legend(new_handles, new_labels, title='Image Size', title_fontsize=35, bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=35)
    else:
        plt.legend(title='Image Size', title_fontsize=35, bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=35)

    plt.gcf().set_size_inches(30, 10)
    # Save the plot as a .png file
    plt.tight_layout()
    #plt.savefig(opts['save_figures'] + f'{key}_ACC@1.png', dpi=300, bbox_inches='tight')
    plt.savefig(opts['save_figures'] + f'{key}_ACC@1.png')
    #plt.show()