import numpy as np
import tensorflow as tf
#from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
#from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
#from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2M, preprocess_input
from params import opts



from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, average_precision_score
import matplotlib.pyplot as plt
import cv2
from metric import *
from tqdm import tqdm

size = opts['resize']
top_n = opts['top_k']
data = np.load(opts['data_path'])

train_images = data['train_images']
train_labels = data['train_labels']
test_images = data['test_images']
test_labels = data['test_labels']

#just test for github update


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

model = EfficientNetV2M(weights='imagenet', include_top=False, input_shape=(size, size, 3), pooling='avg')


# Extract features
train_features = model.predict(train_images_rgb, batch_size=8)
test_features = model.predict(test_images_rgb, batch_size=8)


ap_k_list, hit_rate_k_list, mmv_k_list = [], [], []
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
    mmv_k_idx = mMV_k([label_true], labels_ret, k=top_n)
    ap_k_list.append(ap_k_idx)
    hit_rate_k_list.append(hit_rate_k_idx)
    mmv_k_list.append(mmv_k_idx)

mean_ap_k_list = np.mean(ap_k_list)
mean_hit_rate_k_list = np.mean(hit_rate_k_list)
mean_mmv_k_list = np.mean(mmv_k_list)
print(f"mean_ap_k_list: {mean_ap_k_list} \n"
      f"mean_hit_rate_k_list: {mean_hit_rate_k_list} \n"
      f" mean_mmv_k_list: {mean_mmv_k_list} \n")




