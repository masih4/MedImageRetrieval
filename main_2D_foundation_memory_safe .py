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
import torch
from natsort import natsorted
import glob
import os
from sklearn.manifold import TSNE
import umap
from matplotlib.patches import Patch
from utils import *

size = opts['resize']
top_n = opts['top_k']
data = np.load(opts['data_path'])
file_pattern = '*.npy'

# Use GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('GPU vailablity:', torch.cuda.is_available())

train_labels = data['train_labels']
test_labels = data['test_labels']

# Load the pretrained model and tokenizer


train_files = natsorted(glob.glob(os.path.join(opts['save_train_hard'], file_pattern)))
test_files = natsorted(glob.glob(os.path.join(opts['save_test_hard'], file_pattern)))


def process_and_extract_features(files, labels, opts, batch_size=100):
    if opts['pretrained_network_name'] == 'biomedclip':
        from open_clip import create_model_from_pretrained, get_tokenizer  # works on open-clip-torch>=2.23.0, timm>=0.9.8
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
    elif opts['pretrained_network_name'] == 'UNI':
        import timm
        from torchvision import transforms
        # from huggingface_hub import login, hf_hub_download
        # login()  # login with your User Access Token, found at https://huggingface.co/settings/tokens
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
    elif opts['pretrained_network_name'] == 'openclip':
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s34b_b88k')  # 80.1%
        model.eval()
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        model.to(device)
    elif opts['pretrained_network_name'] == 'conch':
        from conch.open_clip_custom import create_model_from_pretrained
        model, preprocess = create_model_from_pretrained('conch_ViT-B-16',"../pretrained_weights/CONCH/pytorch_model.bin")
        model.eval()
        model.to(device)
    if opts['pretrained_network_name'] == 'virchow':
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        from timm.layers import SwiGLUPacked
        model = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
        model = model.eval()
        model.to(device)
        transforms = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))


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
        elif opts['pretrained_network_name'] == 'UNI':
            preprocessed_batch = torch.stack([transform(Image.fromarray(img)) for img in batch_images]).to(device)
            with torch.inference_mode():
                feature_emb = model(preprocessed_batch)  # Extracted features (torch.Tensor) with shape [1,1024]
                features.extend(feature_emb.cpu().numpy())
        elif opts['pretrained_network_name'] == 'openclip':
            preprocessed_batch = torch.stack([preprocess(Image.fromarray(img)) for img in batch_images]).to(device)
            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = model.encode_image(preprocessed_batch)
                # image_features /= image_features.norm(dim=-1, keepdim=True)
                features.extend(image_features.cpu().numpy())
        elif opts['pretrained_network_name'] == 'conch':
            preprocessed_batch = torch.stack([preprocess(Image.fromarray(img)) for img in batch_images]).to(device)
            with torch.inference_mode():
                image_features = model.encode_image(preprocessed_batch, proj_contrast=False, normalize=True)
                features.extend(image_features.cpu().numpy())
        if opts['pretrained_network_name'] == 'virchow':
            preprocessed_batch = torch.stack([transforms(Image.fromarray(img)) for img in batch_images]).to(device)
            with torch.inference_mode():
                output = model(preprocessed_batch)  # size: 1 x 257 x 1280
            class_token = output[:, 0]  # size: 1 x 1280
            patch_tokens = output[:, 1:]  # size: 1 x 256 x 1280
            embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
            features.extend(embedding.cpu().numpy())


    return features


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
