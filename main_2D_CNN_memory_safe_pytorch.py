import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
import glob
import os
import time
from tqdm import tqdm
from natsort import natsorted
from params import opts
from metric import *

def convert_to_rgb(images):
    return np.stack([images, images, images], axis=-1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

size = opts['resize']
top_n = opts['top_k']
data = np.load(opts['data_path'])
file_pattern = '*.npy'

train_labels = data['train_labels']
test_labels = data['test_labels']
val_labels = data['val_labels']

# Selecting pretrained model
if opts['pretrained_network_name'] == 'EfficientNetV2M':
    model = models.efficientnet_v2_m(weights='IMAGENET1K_V1')
elif opts['pretrained_network_name'] == 'VGG19':
    model = models.vgg19(weights='IMAGENET1K_V1')
elif opts['pretrained_network_name'] == 'DenseNet121':
    model = models.densenet121(weights='IMAGENET1K_V1')
elif opts['pretrained_network_name'] == 'ResNet50':
    model = models.resnet50(weights='IMAGENET1K_V1')
else:
    raise ValueError("Unsupported model")

# Remove fully connected layers
model = nn.Sequential(*list(model.children())[:-1])
model = model.to(device)
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((size, size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_files = natsorted(glob.glob(os.path.join(opts['save_train_hard'], file_pattern)))
test_files = natsorted(glob.glob(os.path.join(opts['save_test_hard'], file_pattern)))

train_features, test_features = [], []

start_time_train = time.time()
for file in tqdm(train_files):
    img = np.load(file)
    img = cv2.resize(img, (size, size))
    if len(img.shape) == 2:
        img = convert_to_rgb(img)
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(img).cpu().numpy().flatten()
    train_features.append(features)
end_time_train = time.time()

start_time_test = time.time()
for file in tqdm(test_files):
    img = np.load(file)
    img = cv2.resize(img, (size, size))
    if len(img.shape) == 2:
        img = convert_to_rgb(img)
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(img).cpu().numpy().flatten()
    test_features.append(features)
end_time_test = time.time()

ap_k_list, hit_rate_k_list, mmv_k_list, acc_1_list, acc_3_list, acc_5_list = [], [], [], [], [], []
for i in tqdm(range(len(test_features))):
    query_features = test_features[i]
    label_true = test_labels[i]
    retrieved = [(np.linalg.norm(query_features - train_features[idx]), idx) for idx in range(len(train_features))]
    results = sorted(retrieved)[:top_n]
    labels_ret = [train_labels[r[1]] for r in results]

    ap_k_list.append(ap_k([label_true], labels_ret, k=top_n))
    hit_rate_k_list.append(hit_rate_k([label_true], labels_ret, k=top_n))
    acc_1_list.append(acc_k([label_true], labels_ret, acc_topk=1))
    acc_3_list.append(acc_k([label_true], labels_ret, acc_topk=3))
    acc_5_list.append(acc_k([label_true], labels_ret, acc_topk=5))
    mmv_k_list.append(mMV_k([label_true], labels_ret, k=top_n))

# Compute mean metrics
mean_ap_k = np.mean(ap_k_list)
mean_hit_rate_k = np.mean(hit_rate_k_list)
mean_mmv_k = np.mean(mmv_k_list)
mean_acc_1 = np.mean(acc_1_list)
mean_acc_3 = np.mean(acc_3_list)
mean_acc_5 = np.mean(acc_5_list)

runtime_train = (end_time_train - start_time_train) / 60
runtime_test = (end_time_test - start_time_test) / 60

print(f"mean_ap_k: {mean_ap_k:.4f}\n"
      f"mean_hit_rate_k: {mean_hit_rate_k:.4f}\n"
      f"mean_mmv_k: {mean_mmv_k:.4f}\n"
      f"mean ACC@1: {mean_acc_1:.4f}\n"
      f"mean ACC@3: {mean_acc_3:.4f}\n"
      f"mean ACC@5: {mean_acc_5:.4f}\n"
      f"Runtime Train: {runtime_train:.2f} minutes\n"
      f"Runtime Test: {runtime_test:.2f} minutes\n")