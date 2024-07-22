import numpy as np
from params import opts
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, average_precision_score
import matplotlib.pyplot as plt
import cv2
from metric import *
from tqdm import tqdm
import time
from urllib.request import urlopen
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
from torchvision import transforms
from utils import *

# Set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load data
size = opts['resize']
top_n = opts['top_k']
data = np.load(opts['data_path'])

train_images = data['train_images']
train_labels = data['train_labels']
test_images = data['test_images']
test_labels = data['test_labels']

print('number of classes:', len(np.unique(train_labels)))

# Load model and preprocess function based on the specified network
if opts['pretrained_network_name'] == 'biomedclip':
    model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    labels = ['dummy text']
    model.to(device)
    model.eval()
    context_length = 256
    texts = tokenizer([l for l in labels], context_length=context_length).to(device)
elif opts['pretrained_network_name'] == 'medclip':
    from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPVisionModel
    from medclip import MedCLIPProcessor
    processor = MedCLIPProcessor()
    model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    model.from_pretrained(input_dir='../pretrained_weights/medclip-vit/')
    ### Note: for resenet backbone set the batchsize to 1. otherwise the output is not consistent
    # model = MedCLIPModel(vision_cls=MedCLIPVisionModel) # for Resnet backbone
    # model.from_pretrained(input_dir='../pretrained_weights/medclip-resnet/')
    model.cuda()
elif opts['pretrained_network_name'] == 'UNI':
    import timm
    # from huggingface_hub import login, hf_hub_download
    #login()  # login with your User Access Token, found at https://huggingface.co/settings/tokens
    local_dir = "../pretrained_weights/UNI/assets/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/"
    # os.makedirs(local_dir, exist_ok=True)  # create directory if it does not exist
    # hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
    model = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
    )
    model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    model.eval()
    model.to(device)


# Resize and convert images
def preprocess_images(images):
    resized_images = np.array([cv2.resize(img, (size, size)) for img in images])
    if len(resized_images.shape) == 3:
        resized_images = convert_to_rgb(resized_images)
    return resized_images

train_images_rgb = preprocess_images(train_images)
test_images_rgb = preprocess_images(test_images)

# Function to extract features in batches
def extract_features(images_rgb, batch_size=1):
    features = []
    dataloader = DataLoader(TensorDataset(torch.tensor(images_rgb)), batch_size=batch_size)
    with torch.no_grad():
        for batch in tqdm(dataloader):
            image_batch = batch[0].numpy()
            if opts['pretrained_network_name'] == 'biomedclip':
                preprocessed_batch = torch.stack([preprocess(Image.fromarray(img)) for img in image_batch]).to(device)
                image_features, _, _ = model(preprocessed_batch, texts)
                features.append(image_features.cpu().numpy())
            elif opts['pretrained_network_name'] == 'medclip':
                inputs = processor(
                    text=["dummy2"] * len(image_batch),
                    images=[Image.fromarray(img) for img in image_batch],
                    return_tensors="pt",
                    padding=True
                ).to(device)
                outputs = model(**inputs)
                features.append(outputs['img_embeds'].cpu().numpy())
            elif opts['pretrained_network_name'] == 'UNI':
                preprocessed_batch = torch.stack([transform(Image.fromarray(img)) for img in image_batch]).to(device)
                with torch.inference_mode():
                    feature_emb = model(preprocessed_batch)  # Extracted features (torch.Tensor) with shape [1,1024]
                    features.append(feature_emb.cpu().numpy())

    return np.vstack(features)




# Extract features
start_time_train = time.time()
train_features = extract_features(train_images_rgb, batch_size=opts['bath_size'])
end_time_train = time.time()

start_time_test = time.time()
test_features = extract_features(test_images_rgb, batch_size=opts['bath_size'])

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

# Evaluate metrics
metrics = evaluate_metrics(test_features, test_labels, train_features, train_labels, top_n)
end_time_test = time.time()

# Print results
runtime_minutes_train = (end_time_train - start_time_train) / 60
runtime_minutes_test = (end_time_test - start_time_test) / 60

print(f"mean_ap_k_list: {metrics['mean_ap_k']:.4f} \n"
      f"mean_hit_rate_k_list: {metrics['mean_hit_rate_k']:.4f} \n"
      f" mean_mmv_k_list: {metrics['mean_mmv_k']:.4f} \n"
      f" mean ACC@1: {metrics['mean_acc_1']:.4f} \n"
      f" mean ACC@3: {metrics['mean_acc_3']:.4f} \n"
      f" mean ACC@5: {metrics['mean_acc_5']:.4f} \n"
      f"Runtime Train: {runtime_minutes_train:.2f} minutes \n"
      f"Runtime Test: {runtime_minutes_test:.2f} minutes \n"
      )
