import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm
import cv2
from scipy.ndimage import zoom
from PIL import Image
import os
from metric import *



def convert_to_rgb(images):
    return np.stack([images, images, images], axis=-1)



def metric_cal(test_features, train_features, test_labels, train_labels, top_n, type='None'):
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

    print("Results for " + type + "\n",
          f"mean_ap_k_list: {mean_ap_k_list:.4f} \n"
          f"mean_hit_rate_k_list: {mean_hit_rate_k_list:.4f} \n"
          f" mean_mmv_k_list: {mean_mmv_k_list:.4f} \n"
          f" mean ACC@1: {mean_acc_1_list:.4f} \n"
          f" mean ACC@3: {mean_acc_3_list:.4f} \n"
          f" mean ACC@5: {mean_acc_5_list:.4f} \n"
          )





def load_and_preprocess_images(files, size, opts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model based on the option
    if opts['pretrained_network_name'] == 'EfficientNetV2M':
        model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1)
    elif opts['pretrained_network_name'] == 'VGG19':
        model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    elif opts['pretrained_network_name'] == 'DenseNet121':
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    elif opts['pretrained_network_name'] == 'ResNet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    else:
        raise ValueError("Unsupported network name")

    model = model.to(device).eval()

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    features = []
    for file in tqdm(files):
        img_3d = np.load(file)
        depth_factor = size / img_3d.shape[0]
        resized_depth_image_3d = zoom(img_3d, (depth_factor, 1, 1), order=3)
        resized_image_3d = np.zeros((size, size, size))

        for i in range(size):
            resized_image_3d[i] = cv2.resize(resized_depth_image_3d[i], (size, size), interpolation=cv2.INTER_CUBIC)

        images_resized = resized_image_3d
        feature_whole_imgX = []

        for x_slice in range(len(images_resized[0])):
            slice = images_resized[x_slice, :, :]
            slice_rgb = convert_to_rgb(slice).astype(np.uint8)
            slice_rgb = preprocess(slice_rgb).unsqueeze(0).to(device)

            if opts['CNN']:
                with torch.no_grad():
                    slice_rgb_feature = model(slice_rgb)
                feature_whole_imgX.append(slice_rgb_feature.cpu().numpy())

        if opts['CNN']:
            feature_whole_imgX_concat = np.concatenate(feature_whole_imgX, axis=1).squeeze()
            features.append(feature_whole_imgX_concat)

    return features
