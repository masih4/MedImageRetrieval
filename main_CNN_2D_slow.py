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

size = opts['resize']
top_n = opts['top_k']
data = np.load(opts['data_path'])
file_pattern = '*.npy'


def convert_to_rgb(images):
    return np.stack([images, images, images], axis=-1)


##############################################################################
# run only if you do not have saved images on storage
# train_images = data['train_images']
# test_images = data['test_images']
# for train_idx in tqdm(range(len(train_images))):
#     img_train = train_images[train_idx]
#     np.save(opts['save_train_hard'] + str(train_idx) + '.npy', img_train)
#
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
    img = np.load(train_files[i_train])
    train_images_resized = cv2.resize(img, (size, size))
    if len(train_images_resized.shape) == 2:
        train_images_rgb = convert_to_rgb(train_images_resized)
    else:
        train_images_rgb = train_images_resized
    train_images_rgb = preprocess_input(train_images_rgb)
    train_images_rgb_expand = np.expand_dims(train_images_rgb, axis=0)
    train_features_img = model.predict(train_images_rgb_expand, batch_size=1, verbose=0)
    train_features.append(train_features_img)
end_time_train = time.time()

start_time_test = time.time()
for i_test in tqdm(range(len(test_files))):
    img = np.load(test_files[i_test])
    test_images_resized = cv2.resize(img, (size, size))
    if len(test_images_resized.shape) == 2:
        test_images_rgb = convert_to_rgb(test_images_resized)
    else:
        test_images_rgb = test_images_resized
    test_images_rgb = preprocess_input(test_images_rgb)
    test_images_rgb_expand = np.expand_dims(test_images_rgb, axis=0)
    test_features_img = model.predict(test_images_rgb_expand, batch_size=1, verbose=0)
    test_features.append(test_features_img)

ap_k_list, hit_rate_k_list, mmv_k_list, acc_1_list, acc_3_list, acc_5_list = [], [], [], [], [], []
for i in tqdm(range(len(test_features))):
    query_features = test_features[i]
    label_true = test_labels[i]
    retrieved = []
    for idx in range(len(train_features)):
        distance = np.linalg.norm(query_features - train_features[idx])
        retrieved.append((distance, idx))
    results = sorted(retrieved)[0:top_n]

    labels_ret = [train_labels[r[1]] for r in results]

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

end_time_test = time.time()

runtime_seconds_train = end_time_train - start_time_train
runtime_minutes_train = runtime_seconds_train / 60

runtime_seconds_test = end_time_test - start_time_test
runtime_minutes_test = runtime_seconds_test / 60

print(f"mean_ap_k_list: {mean_ap_k_list:.4f} \n"
      f"mean_hit_rate_k_list: {mean_hit_rate_k_list:.4f} \n"
      f" mean_mmv_k_list: {mean_mmv_k_list:.4f} \n"
      f" mean ACC@1: {mean_acc_1_list:.4f} \n"
      f" mean ACC@3: {mean_acc_3_list:.4f} \n"
      f" mean ACC@5: {mean_acc_5_list:.4f} \n"
      f"Runtime Train: {runtime_minutes_train:.2f} minutes \n"
      f"Runtime Test: {runtime_minutes_test:.2f} minutes \n"
      )
