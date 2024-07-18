import numpy as np
import cv2
from params import opts
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, average_precision_score
import matplotlib.pyplot as plt
from metric import *
from tqdm import tqdm
import time
from urllib.request import urlopen
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer  # works on open-clip-torch>=2.23.0, timm>=0.9.8
import torch
from natsort import natsorted
import glob
import os
from utils import *

size = opts['resize']
top_n = opts['top_k']
data = np.load(opts['data_path'])
file_pattern = '*.npy'

# Use GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

train_labels = data['train_labels']
test_labels = data['test_labels']

# Load the pretrained model and tokenizer


train_files = natsorted(glob.glob(os.path.join(opts['save_train_hard'], file_pattern)))
test_files = natsorted(glob.glob(os.path.join(opts['save_test_hard'], file_pattern)))


def process_and_extract_features(files, labels, opts, batch_size=100):
    if opts['pretrained_network_name'] == 'biomedclip':
        model, preprocess = create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        labels = ['dummy text']
        model.to(device)
        model.eval()
        context_length = 256
        texts = tokenizer([l for l in labels], context_length=context_length).to(device)
    elif opts['pretrained_network_name'] == 'medclip':
        from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor

        processor = MedCLIPProcessor()
        model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
        model.from_pretrained(input_dir='../pretrained_weights/medclip-vit/')
        model.cuda()




    features = []
    for i in tqdm(range(0, len(files), batch_size)):
        batch_files = files[i:i + batch_size]
        batch_images = []
        for file in batch_files:
            img = np.load(file)
            img_resized = cv2.resize(img, (size, size))
            if len(img_resized.shape) == 2:
                img_rgb = convert_to_rgb(img_resized)
            else:
                img_rgb = img_resized
            batch_images.append(img_rgb)

        if opts['pretrained_network_name'] == 'biomedclip':
            batch_pil = [Image.fromarray(img) for img in batch_images]
            batch_preprocessed = torch.stack([preprocess(img) for img in batch_pil]).to(device)
            with torch.no_grad():
                image_features, _, _ = model(batch_preprocessed, texts)
                features.extend(image_features.cpu().numpy())
        elif opts['pretrained_network_name'] == 'medclip':
            inputs = processor(
                text=["dummy"] * len(batch_images),
                images=[Image.fromarray(img) for img in batch_images],
                return_tensors="pt",
                padding=True
            ).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                features.extend(outputs['img_embeds'].cpu().numpy())

    return features


# Process and extract features for train and test sets



# Function to evaluate metrics
def evaluate_metrics(test_features, test_labels, train_features, train_labels, top_n):
    ap_k_list, hit_rate_k_list, mmv_k_list, acc_1_list, acc_3_list, acc_5_list = [], [], [], [], [], []
    for i in tqdm(range(len(test_features))):
        query_features = test_features[i]
        label_true = test_labels[i]
        distances = np.linalg.norm(train_features - query_features, axis=1)
        results = np.argsort(distances)[:top_n]
        labels_ret = [train_labels[r] for r in results]

        ap_k_list.append(ap_k([label_true], labels_ret, k=top_n))
        hit_rate_k_list.append(hit_rate_k([label_true], labels_ret, k=top_n))
        acc_1_list.append(acc_k([label_true], labels_ret, acc_topk=1))
        acc_3_list.append(acc_k([label_true], labels_ret, acc_topk=3))
        acc_5_list.append(acc_k([label_true], labels_ret, acc_topk=5))
        mmv_k_list.append(mMV_k([label_true], labels_ret, k=top_n))

    return {
        "mean_ap_k": np.mean(ap_k_list),
        "mean_hit_rate_k": np.mean(hit_rate_k_list),
        "mean_mmv_k": np.mean(mmv_k_list),
        "mean_acc_1": np.mean(acc_1_list),
        "mean_acc_3": np.mean(acc_3_list),
        "mean_acc_5": np.mean(acc_5_list)
    }


# Extract features
start_time_train = time.time()
train_features = process_and_extract_features(train_files, train_labels, opts, batch_size=opts['bath_size'])
end_time_train = time.time()

start_time_test = time.time()
test_features = process_and_extract_features(test_files, test_labels, opts, batch_size=opts['bath_size'])

# Evaluate metrics
metrics = evaluate_metrics(test_features, test_labels, train_features, train_labels, top_n)
end_time_test = time.time()

# Print results
runtime_minutes_train = (end_time_train - start_time_train) / 60
runtime_minutes_test = (end_time_test - start_time_test) / 60

print(f"mean_ap_k_list: {metrics['mean_ap_k']:.4f} \n"
      f"mean_hit_rate_k_list: {metrics['mean_hit_rate_k']:.4f} \n"
      f"mean_mmv_k_list: {metrics['mean_mmv_k']:.4f} \n"
      f"mean ACC@1: {metrics['mean_acc_1']:.4f} \n"
      f"mean ACC@3: {metrics['mean_acc_3']:.4f} \n"
      f"mean ACC@5: {metrics['mean_acc_5']:.4f} \n"
      f"Runtime Train: {runtime_minutes_train:.2f} minutes \n"
      f"Runtime Test: {runtime_minutes_test:.2f} minutes \n")

