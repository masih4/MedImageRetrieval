import numpy as np
from params import opts
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, average_precision_score
import matplotlib.pyplot as plt
import cv2
from metric import *
from tqdm import tqdm
import gc
import glob
import os
from natsort import natsorted
import time
from scipy.ndimage import zoom
from sklearn.decomposition import PCA
from PIL import Image
from sklearn.manifold import TSNE
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from sklearn.manifold import TSNE
import umap
from matplotlib.patches import Patch
from utils import *


size = opts['resize']
top_n = opts['top_k']
data = np.load(opts['data_path'])
file_pattern = '*.npy'

#####################################################
# run only if you do not have saved images on storage
# train_images = data['train_images']
# test_images = data['test_images']
# for train_idx in tqdm(range(len(train_images))):
#     img_train = train_images[train_idx]
#     np.save(opts['save_train_hard'] + str(train_idx) + '.npy', img_train)
# #
# for test_idx in tqdm(range(len(test_images))):
#     img_test = test_images[test_idx]
#     np.save(opts['save_test_hard'] + str(test_idx) + '.npy', img_test)
##############################################################################
train_labels = data['train_labels']
test_labels = data['test_labels']




train_files = glob.glob(os.path.join(opts['save_train_hard'], file_pattern))
test_files = glob.glob(os.path.join(opts['save_test_hard'], file_pattern))


train_files = natsorted(train_files)
test_files = natsorted(test_files)



start_time_train = time.time()
train_features = load_and_preprocess_images(train_files, size, opts)
end_time_train = time.time()

start_time_test = time.time()
test_features = load_and_preprocess_images(test_files, size, opts)



metric_cal(test_features, train_features, test_labels,train_labels, top_n, type='without feature reduction')

end_time_test = time.time()
runtime_seconds_train = end_time_train - start_time_train
runtime_minutes_train = runtime_seconds_train / 60

runtime_seconds_test = end_time_test - start_time_test
runtime_minutes_test = runtime_seconds_test / 60
print('###########################################################################')
print(f"Runtime Train: {runtime_minutes_train:.2f} minutes \n"
      f"Runtime Test: {runtime_minutes_test:.2f} minutes \n"
      )

train_features = np.array(train_features)
test_features = np.array(test_features)
print('###########################################################################')
# PCA
pca = PCA(n_components=256)
train_features_pca = pca.fit_transform(train_features)
test_features_pca = pca.fit_transform(test_features)
metric_cal(test_features_pca, train_features_pca, test_labels, train_labels, top_n, type='PCA')
print('###########################################################################')
# Autoencoder
# Define the autoencoder
input_dim = np.array(train_features).shape[1]
encoding_dim = 256
input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation="relu")(input_layer)
decoder = Dense(input_dim, activation="sigmoid")(encoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
# Train the autoencoder
autoencoder.fit(np.array(train_features), train_features, epochs=50, batch_size=256, shuffle=True, verbose=0)
encoder_model = Model(inputs=input_layer, outputs=encoder)
train_features_autoencoder = encoder_model.predict(train_features)
autoencoder.fit(test_features, test_features, epochs=200, batch_size=256, shuffle=True, verbose=0)
encoder_model = Model(inputs=input_layer, outputs=encoder)
test_features_autoencoder = encoder_model.predict(test_features)
metric_cal(test_features_autoencoder, train_features_autoencoder, test_labels, train_labels, top_n, type='Auto Encoder')
print('###########################################################################')
# TSNE
# TSNE is more commonly used for 2D or 3D visualization, but it can be used for higher dimensions
tsne = TSNE(n_components=2)
train_features_TSNE = tsne.fit_transform(train_features)
test_features_TSNE = tsne.fit_transform(test_features)
metric_cal(test_features_TSNE, train_features_TSNE, test_labels, train_labels, top_n, type='TSNE')
print('###########################################################################')
#########################################################
# UMAP
umap_reducer = umap.UMAP(n_components=2)
train_features_UMAP = umap_reducer.fit_transform(train_features)
test_features_UMAP = umap_reducer.fit_transform(test_features)
metric_cal(test_features_UMAP, train_features_UMAP, test_labels, train_labels, top_n, type='UMAP')
print('###########################################################################')


if opts['tsne']:
    train_features = np.array(train_features)
    # Reshape test_labels to be a 1D array for indexing purposes
    train_labels = train_labels.flatten()

    # Fit t-SNE model
    tsne = TSNE(n_components=2, random_state=42, n_iter=1500)
    train_features_red = tsne.fit_transform(train_features)


    # Plotting t-SNE results with legend
    plt.figure(figsize=(10, 8))

    # Get unique labels
    unique_labels = np.unique(train_labels)
    colors = plt.cm.get_cmap('plasma', len(unique_labels))
    distinct_colors = ['red', 'blue', 'green', 'purple', 'orange', 'black','cyan', 'magenta', 'yellow']

    # Create scatter plot with a legend
    for i, label in enumerate(unique_labels):
        indices = train_labels == label
        plt.scatter(train_features_red[indices, 0], train_features_red[indices, 1],
                    color=distinct_colors[i], label=f'Class {label}', alpha=1, s=10)
    # plt.legend(title='Classes', title_fontsize=25, fontsize=25, loc='upper right', framealpha=1)
    handles = [Patch(color=distinct_colors[i]) for i in range(len(unique_labels))]
    plt.legend(handles=handles, title='', title_fontsize=20, loc='upper right', fontsize=20,
               framealpha=1, handleheight=0.5, handlelength=1)  # Create legend with colored rectangles

    plt.title(opts['tsne_title'], fontsize=40)
    plt.xlabel('t-SNE dimension 1', fontsize=40)
    plt.ylabel('t-SNE dimension 2', fontsize=40)
    # Remove x-axis and y-axis numbers
    plt.xticks([])
    plt.yticks([])


    # Save the figure with high resolution
    output_path = opts['save_figures_tsen'] + opts['tsne_title'] + '.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    print(f"t-SNE plot saved as {output_path}")