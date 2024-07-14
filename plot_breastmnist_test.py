import pandas as pd
import matplotlib.pyplot as plt
from params import opts


# Data extracted from the document for each dataset and ACC@1, filling missing data with 50
data = {
    'BreastMNIST': {
        'Model': ['VGG19', 'ResNet50', 'DenseNet', 'Eff.Net', 'CCL', 'SAM', 'MEDSAM', 'Biomedclip'],
        '28x28': [79.48, 79.48, 80.12, 78.20, 40, 40, 40, 40],
        '64x64': [82.05, 80.76, 82.05, 75.64, 40, 40, 40, 40],
        '128x128': [80.76, 83.97, 87.82, 78.84, 40, 40, 40, 40],
        '224x224': [86.53, 83.33, 84.61, 81.41, 40, 40, 40, 40]
    },
    'PneumoniaMNIST': {
        'Model': ['VGG19', 'ResNet50', 'DenseNet', 'Eff.Net', 'CCL', 'SAM', 'MEDSAM', 'Biomedclip'],
        '28x28': [83.17, 82.85, 83.49, 77.08, 40, 40, 40, 40],
        '64x64': [82.37, 86.21, 84.77, 75.00, 40, 40, 40, 40],
        '128x128': [81.25, 84.29, 84.29, 78.04, 40, 40, 40, 40],
        '224x224': [82.37, 87.01, 87.5, 81.08, 40, 40, 40, 40]
    },
    'RetinaMNIST': {
        'Model': ['VGG19', 'ResNet50', 'DenseNet', 'Eff.Net', 'CCL', 'SAM', 'MEDSAM', 'Biomedclip'],
        '28x28': [46.25, 42.25, 45.25, 42.75, 40, 40, 40, 40],
        '64x64': [44.25, 48.5, 44.25, 45.00, 40, 40, 40, 40],
        '128x128': [47.75, 46.75, 50.00, 44.25, 40, 40, 40, 40],
        '224x224': [46.00, 48.00, 49.75, 48.75, 40, 40, 40, 40]
    },
    'DermaMNIST': {
        'Model': ['VGG19', 'ResNet50', 'DenseNet', 'Eff.Net', 'CCL', 'SAM', 'MEDSAM', 'Biomedclip'],
        '28x28': [67.73, 67.88, 67.83, 64.13, 40, 40, 40, 40],
        '64x64': [70.72, 72.16, 69.72, 64.93, 40, 40, 40, 40],
        '128x128': [71.82, 76.40, 73.11, 69.57, 40, 40, 40, 40],
        '224x224': [74.71, 77.75, 74.36, 70.92, 40, 40, 40, 40]
    },
    'BloodMNIST': {
        'Model': ['VGG19', 'ResNet50', 'DenseNet', 'Eff.Net', 'CCL', 'SAM', 'MEDSAM', 'Biomedclip'],
        '28x28': [73.07, 64.98, 69.95, 56.47, 40, 40, 40, 40],
        '64x64': [71.03, 66.93, 73.16, 63.98, 40, 40, 40, 40],
        '128x128': [74.86, 79.50, 78.74, 76.90, 40, 40, 40, 40],
        '224x224': [81.58, 87.69, 84.41, 81.29, 40, 40, 40, 40]
    },
    'PathMNIST': {
        'Model': ['VGG19', 'ResNet50', 'DenseNet', 'Eff.Net', 'CCL', 'SAM', 'MEDSAM', 'Biomedclip'],
        '28x28': [68.67, 69.74, 71.43, 64.24, 40, 40, 40, 40],
        '64x64': [76.40, 79.83, 81.89, 73.21, 40, 40, 40, 40],
        '128x128': [80.52, 84.40, 84.19, 79.33, 40, 40, 40, 40],
        '224x224': [81.10, 85.18, 85.01, 79.61, 40, 40, 40, 40]
    },
    'AdrenalMNIST3D': {
        'Model': ['VGG19', 'ResNet50', 'DenseNet', 'Eff.Net', 'CCL', 'SAM', 'MEDSAM', 'Biomedclip'],
        '28x28': [72.48, 71.47, 74.16, 71.47, 40, 40, 40, 40],
        '64x64': [72.14, 67.11, 68.79, 62.75, 40, 40, 40, 40],
        '128x128': [40, 40, 40, 40, 40, 40, 40, 40],
        '224x224': [40, 40, 40, 40, 40, 40, 40, 40]
    },
    'SynapseMNIST3D': {
        'Model': ['VGG19', 'ResNet50', 'DenseNet', 'Eff.Net', 'CCL', 'SAM', 'MEDSAM', 'Biomedclip'],
        '28x28': [1, 1, 1, 1, 40, 40, 40, 40],
        '64x64': [67.89, 66.47, 63.92, 62.21, 40, 40, 40, 40],
        '128x128': [40, 40, 40, 40, 40, 40, 40, 40],
        '224x224': [40, 40, 40, 40, 40, 40, 40, 40]
    }
}

# Convert to DataFrame
dfs = {key: pd.DataFrame(value).set_index('Model') for key, value in data.items()}

# Plotting the data and saving the plots
