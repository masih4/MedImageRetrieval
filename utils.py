import numpy as np
from tqdm import tqdm
from metric import *
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from params import opts
from scipy.ndimage import zoom
import cv2
import os
from torchvision import transforms
from PIL import Image
if opts['framework'] == 'pytorch':
    from open_clip import create_model_from_pretrained, get_tokenizer  # works on open-clip-torch>=2.23.0, timm>=0.9.8
    import torch



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

    elif opts['pretrained_network_name'] == 'biomedclip':
        model, preprocess = create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        labels = ['dummy text']
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)
        model.eval()
        context_length = 256
        texts = tokenizer([l for l in labels], context_length=context_length).to(device)

    elif opts['pretrained_network_name'] == 'medclip':
        from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        processor = MedCLIPProcessor()
        model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
        model.from_pretrained(input_dir='../pretrained_weights/medclip-vit/')
        model.cuda()
    elif opts['pretrained_network_name'] == 'UNI':
        import timm
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        local_dir = "../pretrained_weights/UNI/assets/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/"
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


    features = []
    for file in tqdm(files):
        img_3d = np.load(file)
        depth_factor = size / img_3d.shape[0]
        resized_depth_image_3d = zoom(img_3d, (depth_factor, 1, 1), order=3)
        resized_image_3d = np.zeros((size, size, size))
        for i in range(size):
            resized_image_3d[i] = cv2.resize(resized_depth_image_3d[i], (size, size), interpolation=cv2.INTER_CUBIC)
        images_resized = resized_image_3d
        ###########################3
        feature_whole_imgX = []
        for x_slice in range(len(images_resized[0])):
            slice = images_resized[x_slice, :, :]
            slice_rgb = convert_to_rgb(slice)

            if opts['CNN']:
                slice_rgb = preprocess_input(slice_rgb)
                slice_rgb_expand = np.expand_dims(slice_rgb, axis=0)
                slice_rgb_feature = model.predict(slice_rgb_expand, batch_size=1, verbose=0)
                feature_whole_imgX.append(slice_rgb_feature)
            elif opts['pretrained_network_name'] == 'biomedclip':
                slice_rgb = slice_rgb.astype(np.uint8)
                image_pil = Image.fromarray(slice_rgb)
                image_pil_preprocess = torch.stack([preprocess(image_pil)]).to(device)
                with torch.no_grad():
                    image_features, _, _ = model(image_pil_preprocess, texts)
                    feature_whole_imgX.append(image_features.squeeze().cpu().numpy())
            elif opts['pretrained_network_name'] == 'medclip':
                slice_rgb = slice_rgb.astype(np.uint8)
                inputs = processor(
                    text=["dummy"] ,
                    images= Image.fromarray(slice_rgb),
                    return_tensors="pt",
                    padding=True
                ).to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    feature_whole_imgX.append(outputs['img_embeds'].squeeze().cpu().numpy())
            elif opts['pretrained_network_name'] == 'UNI':
                slice_rgb = slice_rgb.astype(np.uint8)
                image_pil = Image.fromarray(slice_rgb)
                image_pil_preprocess = torch.stack([transform(image_pil)]).to(device)
                with torch.inference_mode():
                    image_features = model(image_pil_preprocess)  # Extracted features (torch.Tensor) with shape [1,1024]
                    feature_whole_imgX.append(image_features.squeeze().cpu().numpy())

        if opts['CNN']:
            feature_whole_imgX_concat = np.concatenate(feature_whole_imgX, axis=1).squeeze()
        elif opts['pretrained_network_name'] == 'biomedclip' or 'medclip' or 'UNI':
            feature_whole_imgX_concat = np.concatenate(feature_whole_imgX, axis=0)
        # elif opts['pretrained_network_name'] == 'medclip':
        #     feature_whole_imgX_concat = np.concatenate(feature_whole_imgX, axis=0)

        features.append(feature_whole_imgX_concat)

    return features