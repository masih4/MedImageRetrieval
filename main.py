import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, average_precision_score
import matplotlib.pyplot as plt
import cv2
from metric import *
from tqdm import tqdm

size = 128

# Load the dataset
data = np.load('../dataset/128/breastmnist_128.npz')

train_images = data['train_images']
train_labels = data['train_labels']
test_images = data['test_images']
test_labels = data['test_labels']


# Resize images
train_images_resized = np.array([cv2.resize(img, (size, size)) for img in train_images])
test_images_resized = np.array([cv2.resize(img, (size, size)) for img in test_images])

# Normalize images
train_images_resized = preprocess_input(train_images_resized)
test_images_resized = preprocess_input(test_images_resized)

def convert_to_rgb(images):
    return np.stack([images, images, images], axis=-1)

if len(train_images_resized.shape) == 3:
    train_images_rgb = convert_to_rgb(train_images_resized)
    test_images_rgb = convert_to_rgb(test_images_resized)


print('number of classes:', len(np.unique(train_labels)))

model = VGG19(weights='imagenet', include_top=False, input_shape=(size, size, 3), pooling='avg')


# Extract features
train_features = model.predict(train_images_rgb, batch_size=8)
test_features = model.predict(test_images_rgb, batch_size=8)

# Build the retrieval model
def retrieve_similar_images(query_features, databank_features, top_n=5):
    similarities = cosine_similarity(query_features, databank_features)
    indices = np.argsort(similarities, axis=1)[:, -top_n:][:, ::-1]
    return indices

# Retrieve the top 5 similar images for each test image
top_n = 5
retrieved_indices = retrieve_similar_images(test_features, train_features, top_n)



#evaluate emad code
list_res = []
for i in range(len(test_labels)):
    res = map_k(test_labels[i], train_labels[retrieved_indices[i]], k=5)
    list_res.append(res)

np.mean(list_res)






ap_k_list, hit_rate_k_list, mmv_k_list = [], [], []
for i in tqdm(range(len(test_features))):
    query_features = test_features[i]
    label_true = test_labels[i]
    retrieved = []
    for idx in range(len(train_features)):
        distance = np.linalg.norm(query_features - train_features[idx])
        retrieved.append((distance, idx))
    results = sorted(retrieved)[0:top_n]
    #results = np.array(results)

    #labels_ret = [labels_train[r[1]] for r in results]
    #label_true = dataset_test[i].split("/")[3]  ##############################

    labels_ret =  [train_labels[r[1]] for r in results]

    ap_k_idx = ap_k([label_true], labels_ret, k=top_n)
    hit_rate_k_idx = hit_rate_k([label_true], labels_ret, k=top_n)
    mmv_k_idx = mMV_k([label_true], labels_ret, k=top_n)
    ap_k_list.append(ap_k_idx)
    hit_rate_k_list.append(hit_rate_k_idx)
    mmv_k_list.append(mmv_k_idx)

mean_ap_k_list = np.mean(ap_k_list)
mean_hit_rate_k_list = np.mean(hit_rate_k_list)
mean_mmv_k_list = np.mean(mmv_k_list)
print(
    f"pretrained_mean_ap_k_list: {mean_ap_k_list} pretrained_mean_hit_rate_k_list: {mean_hit_rate_k_list}"
    f" pretrained_mean_mmv_k_list: {mean_mmv_k_list}")










# # Evaluate (ineternet)
# def evaluate_retrieval(retrieved_indices, test_labels, train_labels):
#     precisions = []
#     aps = []
#     for i, indices in enumerate(retrieved_indices):
#         retrieved_labels = train_labels[indices]
#         true_label = test_labels[i]
#         precision = precision_score([true_label] * top_n, retrieved_labels, average='macro')
#         ap = average_precision_score([true_label] * top_n, retrieved_labels)
#         precisions.append(precision)
#         aps.append(ap)
#     return np.mean(precisions), np.mean(aps)
#
# mean_precision, mean_ap = evaluate_retrieval(retrieved_indices, test_labels, train_labels)
#
# print(f'Mean Precision: {mean_precision}')
# print(f'Mean Average Precision: {mean_ap}')
