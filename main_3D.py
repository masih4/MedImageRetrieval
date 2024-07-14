import numpy as np
import tensorflow as tf
from params import opts
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.models import Model
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


size = opts['resize']
top_n = opts['top_k']
data = np.load(opts['data_path'])
file_pattern = '*.npy'

def convert_to_rgb(images):
    return np.stack([images, images, images], axis=-1)

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


if opts['pretrained_network_name'] == 'EfficientNetV2M':
    from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2M, preprocess_input
    model = EfficientNetV2M(weights='imagenet', include_top=False, input_shape=(size, size, 3), pooling='avg')

elif opts['pretrained_network_name'] == 'VGG19':
    from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
    model = VGG19(weights='imagenet', include_top=False, input_shape=(size, size, 3), pooling='avg')

elif opts['pretrained_network_name'] == 'DenseNet121':
    from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
    model = DenseNet121(weights='imagenet', include_top=False, input_shape=(size, size, 3), pooling='avg')

elif opts['pretrained_network_name'] == 'ResNet50':
    from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(size, size, 3), pooling='avg')


train_files = glob.glob(os.path.join(opts['save_train_hard'], file_pattern))
test_files = glob.glob(os.path.join(opts['save_test_hard'], file_pattern))

# this is NOT sorting correctly!
# train_files.sort()
# test_files.sort()

train_files = natsorted(train_files)
test_files = natsorted(test_files)


train_features, test_features = [], []

start_time_train = time.time()
for i_train in tqdm(range(len(train_files))):
    img_3d = np.load(train_files[i_train])

    #resize
    depth_factor = size / img_3d.shape[0]
    resized_depth_image_3d = zoom(img_3d, (depth_factor, 1, 1), order=3)
    resized_image_3d = np.zeros((size, size, size))

    for i in range(size):
        resized_image_3d[i] = cv2.resize(resized_depth_image_3d[i], (size, size),
                                         interpolation=cv2.INTER_CUBIC)

    train_images_resized = resized_image_3d
    train_images_resized = preprocess_input(train_images_resized )

    feature_whole_imgX = []
    for x_slice in range(len(train_images_resized[0])):
        slice = train_images_resized[x_slice, :, :]
        slice_rgb = convert_to_rgb(slice)
        slice_rgb_expand = np.expand_dims( slice_rgb, axis=0)
        slice_rgb_feature = model.predict(slice_rgb_expand, batch_size=1, verbose=0)
        feature_whole_imgX.append(slice_rgb_feature)
    feature_whole_imgX_concat = np.concatenate(feature_whole_imgX, axis=1)
    feature_whole_imgX_concat = np.squeeze(feature_whole_imgX_concat)
    train_features.append(feature_whole_imgX_concat)
end_time_train = time.time()




start_time_test = time.time()
for i_test in tqdm(range(len(test_files))):
    img_3d = np.load(test_files[i_test])

    #resize
    depth_factor = size / img_3d.shape[0]
    resized_depth_image_3d = zoom(img_3d, (depth_factor, 1, 1), order=3)
    resized_image_3d = np.zeros((size, size, size))

    for i in range(size):
        resized_image_3d[i] = cv2.resize(resized_depth_image_3d[i], (size, size),
                                         interpolation=cv2.INTER_CUBIC)

    test_images_resized = resized_image_3d
    test_images_resized = preprocess_input(test_images_resized )

    feature_whole_imgX = []
    for x_slice in range(len(test_images_resized[0])):
        slice = test_images_resized[x_slice, :, :]
        slice_rgb = convert_to_rgb(slice)
        slice_rgb_expand = np.expand_dims( slice_rgb, axis=0)
        slice_rgb_feature = model.predict(slice_rgb_expand, batch_size=1, verbose=0)
        feature_whole_imgX.append(slice_rgb_feature)
    feature_whole_imgX_concat = np.concatenate(feature_whole_imgX, axis=1)
    feature_whole_imgX_concat = np.squeeze(feature_whole_imgX_concat)
    test_features.append(feature_whole_imgX_concat)




def metric_cal(test_features, train_features, type = 'None'):
    ap_k_list, hit_rate_k_list, mmv_k_list, acc_1_list, acc_3_list, acc_5_list = [], [], [], [], [], []
    for i in tqdm(range(len(test_features))):
        query_features = test_features[i]
        label_true = test_labels[i]
        retrieved = []
        for idx in range(len(train_features)):
            distance = np.linalg.norm(query_features - train_features[idx])
            retrieved.append((distance, idx))
        results = sorted(retrieved)[0:top_n]


        labels_ret =  [train_labels[r[1]] for r in results]

        ap_k_idx = ap_k([label_true], labels_ret, k=top_n)
        hit_rate_k_idx = hit_rate_k([label_true], labels_ret, k=top_n)
        acc_1_idx = acc_k([label_true], labels_ret, acc_topk=1)
        acc_3_idx = acc_k([label_true], labels_ret, acc_topk=3)
        acc_5_idx = acc_k([label_true], labels_ret, acc_topk=5)

        mmv_k_idx = mMV_k([label_true], labels_ret, k=top_n)
        ap_k_list.append(ap_k_idx)
        hit_rate_k_list.append(hit_rate_k_idx)
        acc_1_list.append(acc_1_idx)
        acc_3_list.append(acc_3_idx)
        acc_5_list.append(acc_5_idx)
        mmv_k_list.append(mmv_k_idx)

    mean_ap_k_list = np.mean(ap_k_list)
    mean_hit_rate_k_list = np.mean(hit_rate_k_list)
    mean_mmv_k_list = np.mean(mmv_k_list)
    mean_acc_1_list = np.mean(acc_1_list)
    mean_acc_3_list = np.mean(acc_3_list)
    mean_acc_5_list = np.mean(acc_5_list)

    print("Results for " + type + "\n",
          f"mean_ap_k_list: {mean_ap_k_list:.2f} \n"
          f"mean_hit_rate_k_list: {mean_hit_rate_k_list:.2f} \n"
          f" mean_mmv_k_list: {mean_mmv_k_list:.2f} \n"
          f" mean ACC@1: {mean_acc_1_list:.2f} \n"
          f" mean ACC@3: {mean_acc_3_list:.2f} \n"
          f" mean ACC@5: {mean_acc_5_list:.2f} \n"
          )

metric_cal(test_features, train_features, type='without feature reduction')

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
from sklearn.decomposition import PCA
pca = PCA(n_components=256)
train_features_pca = pca.fit_transform(train_features)
test_features_pca = pca.fit_transform(test_features)
metric_cal(test_features_pca, train_features_pca, type='PCA')
print('###########################################################################')

# autoencoder
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
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
metric_cal(test_features_autoencoder, train_features_autoencoder, type='Auto Encoder')
print('###########################################################################')

#TSNE
from sklearn.manifold import TSNE
# TSNE is more commonly used for 2D or 3D visualization, but it can be used for higher dimensions
tsne = TSNE(n_components=2)
train_features_TSNE = tsne.fit_transform(train_features)
test_features_TSNE = tsne.fit_transform(test_features)
metric_cal(test_features_TSNE , train_features_TSNE , type='TSNE')
print('###########################################################################')

#########################################################
#UMAP
import umap
umap_reducer = umap.UMAP(n_components=2)
train_features_UMAP = umap_reducer.fit_transform(train_features)
test_features_UMAP = umap_reducer.fit_transform(test_features)


metric_cal(test_features_UMAP, train_features_UMAP, type='UMAP')
print('###########################################################################')



