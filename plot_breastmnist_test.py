import pandas as pd
import matplotlib.pyplot as plt
from params import opts


# Data extracted from the document for each dataset and ACC@1, filling missing data with 50
data = {
    'BreastMNIST': {
        'Model': ['VGG19', 'ResNet50', 'DenseNet', 'Eff.Net', 'CCL', 'SAM', 'MEDSAM', 'Biomedclip'],
        '28x28': [79.48, 79.48, 80.12, 78.20, 50, 50, 50, 50],
        '64x64': [82.05, 80.76, 82.05, 75.64, 50, 50, 50, 50],
        '128x128': [80.76, 83.97, 87.82, 78.84, 50, 50, 50, 50],
        '224x224': [86.53, 83.33, 84.61, 81.41, 50, 50, 50, 50]
    },
    'PneumoniaMNIST': {
        'Model': ['VGG19', 'ResNet50', 'DenseNet', 'Eff.Net', 'CCL', 'SAM', 'MEDSAM', 'Biomedclip'],
        '28x28': [83.17, 82.85, 83.49, 77.08, 50, 50, 50, 50],
        '64x64': [82.37, 86.21, 84.77, 75.00, 50, 50, 50, 50],
        '128x128': [81.25, 84.29, 84.29, 78.04, 50, 50, 50, 50],
        '224x224': [82.37, 87.01, 87.5, 81.08, 50, 50, 50, 50]
    },
    'RetinaMNIST': {
        'Model': ['VGG19', 'ResNet50', 'DenseNet', 'Eff.Net', 'CCL', 'SAM', 'MEDSAM', 'Biomedclip'],
        '28x28': [46.25, 42.25, 45.25, 42.75, 50, 50, 50, 50],
        '64x64': [44.25, 48.5, 44.25, 45.00, 50, 50, 50, 50],
        '128x128': [47.75, 46.75, 50.00, 44.25, 50, 50, 50, 50],
        '224x224': [46.00, 48.00, 49.75, 48.75, 50, 50, 50, 50]
    },
    'DermaMNIST': {
        'Model': ['VGG19', 'ResNet50', 'DenseNet', 'Eff.Net', 'CCL', 'SAM', 'MEDSAM', 'Biomedclip'],
        '28x28': [67.73, 67.88, 67.83, 64.13, 50, 50, 50, 50],
        '64x64': [70.72, 72.16, 69.72, 64.93, 50, 50, 50, 50],
        '128x128': [71.82, 76.40, 73.11, 69.57, 50, 50, 50, 50],
        '224x224': [74.71, 77.75, 74.36, 70.92, 50, 50, 50, 50]
    },
    'BloodMNIST': {
        'Model': ['VGG19', 'ResNet50', 'DenseNet', 'Eff.Net', 'CCL', 'SAM', 'MEDSAM', 'Biomedclip'],
        '28x28': [73.07, 64.98, 69.95, 56.47, 50, 50, 50, 50],
        '64x64': [71.03, 66.93, 73.16, 63.98, 50, 50, 50, 50],
        '128x128': [74.86, 79.50, 78.74, 76.90, 50, 50, 50, 50],
        '224x224': [81.58, 87.69, 84.41, 81.29, 50, 50, 50, 50]
    },
    'PathMNIST': {
        'Model': ['VGG19', 'ResNet50', 'DenseNet', 'Eff.Net', 'CCL', 'SAM', 'MEDSAM', 'Biomedclip'],
        '28x28': [68.67, 69.74, 71.43, 64.24, 50, 50, 50, 50],
        '64x64': [76.40, 79.83, 81.89, 73.21, 50, 50, 50, 50],
        '128x128': [80.52, 50.00, 50.00, 50.00, 50, 50, 50, 50],
        '224x224': [81.10, 85.18, 85.01, 79.61, 50, 50, 50, 50]
    }
}

# Convert to DataFrame
dfs = {key: pd.DataFrame(value).set_index('Model') for key, value in data.items()}

# Plotting the data and saving the plots
for key, df in dfs.items():
    plt.figure(figsize=(12, 8))
    ax = df.plot(kind='bar', ylim=(40, 90))
    plt.title(f'ACC@1 for {key} using Different Models and Image Sizes')
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


