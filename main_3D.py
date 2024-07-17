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
import umap
from sklearn.manifold import TSNE
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from utils import *
if opts['framework'] == 'pytorch':
    from open_clip import create_model_from_pretrained, get_tokenizer  # works on open-clip-torch>=2.23.0, timm>=0.9.8
    import torch


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