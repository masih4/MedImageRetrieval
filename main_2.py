import os
import cv2
import numpy as np
from tqdm import tqdm
import model_metrics
from keras.applications.nasnet import NASNetLarge, preprocess_input  # 4048
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.xception import Xception, preprocess_input  # 2048
from keras.applications.densenet import DenseNet121, preprocess_input
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.applications.efficientnet_v2 import EfficientNetV2M, preprocess_input

N_RANK = 5
PLAN_FEATURE = 'pretrain_keras'
TYPE_SCENARIO = '../brea_128_scenario'
class_dir = label_subclass = ['zero', 'one']
IMAGE_SIZE = (256, 256)

class EmbeddingPretrainManager:
    def __init__(self):
        self.model_pretrain_dict = {}
        self.model_pretrain_dict['VGG19'] = VGG19(weights="../models/pretrained/vgg19-no-top.h5",
                                                  include_top=False,
                                                  input_shape=(256, 256, 3),
                                                  pooling='max')
    @staticmethod
    def pretrained_model(encoder, labels_train, images_train, images_test, dataset_test, max_results):
        train_index = []
        for i in tqdm(images_train):
            encoded_image = preprocess_input(np.array([i]))
            encoded_image = encoder.predict(encoded_image)
            train_index.append(encoded_image)
        test_index = []
        for i in tqdm(images_test):
            encoded_image = preprocess_input(np.array([i]))
            encoded_image = encoder.predict(encoded_image)
            test_index.append(encoded_image)
        query_indexes = list(range(len(images_test)))

        ap_k_list, hit_rate_k_list, mmv_k_list = [], [], []
        for i in tqdm(query_indexes):
            query_features = test_index[i]
            retrieved = []
            for idx in range(len(train_index)):
                distance = np.linalg.norm(query_features - train_index[idx])
                retrieved.append((distance, idx))
            results = sorted(retrieved)[1:max_results+1]
            labels_ret = [labels_train[r[1]] for r in results]
            label_true = dataset_test[i].split("/")[3]   ##############################

            ap_k = model_metrics.ap_k([label_true], labels_ret, k=max_results)
            hit_rate_k = model_metrics.hit_rate_k([label_true], labels_ret, k=max_results)
            mmv_k = model_metrics.mMV_k([label_true], labels_ret, k=max_results)
            ap_k_list.append(ap_k)
            hit_rate_k_list.append(hit_rate_k)
            mmv_k_list.append(mmv_k)

        mean_ap_k_list = np.mean(ap_k_list)
        mean_hit_rate_k_list = np.mean(hit_rate_k_list)
        mean_mmv_k_list = np.mean(mmv_k_list)
        print(
            f"pretrained_mean_ap_k_list: {mean_ap_k_list} pretrained_mean_hit_rate_k_list: {mean_hit_rate_k_list}"
            f" pretrained_mean_mmv_k_list: {mean_mmv_k_list}")
        return None
    @staticmethod
    def prepare_data(class_dir, TYPE_SCENARIO):
        paths_train = []
        for class_item in tqdm(class_dir):
            cur_dir = os.path.join(TYPE_SCENARIO, 'train', class_item)
            paths_train.extend(os.path.join(cur_dir, file) for file in os.listdir(cur_dir))
        paths_test = []
        for class_item in tqdm(class_dir):
            cur_dir = os.path.join(TYPE_SCENARIO, 'test', class_item)
            paths_test.extend(os.path.join(cur_dir, file) for file in os.listdir(cur_dir))

        images_train = []
        for image_path in tqdm(paths_train):
            if ".png" in image_path or ".tif" in image_path:
                image = cv2.imread(image_path)
                image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                images_train.append(image)
        images_test = []
        for image_path in tqdm(paths_test):
            if ".png" in image_path or ".tif" in image_path:
                image = cv2.imread(image_path)
                image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                images_test.append(image)

        labels_train = [path.split("/")[3] for path in paths_train]   #################
        labels_test = [path.split("/")[3] for path in paths_test]
        return paths_test, images_train, images_test, labels_train

EPM = EmbeddingPretrainManager()
paths_test, images_train, images_test, labels_train = EPM.prepare_data(class_dir, TYPE_SCENARIO)
EPM.pretrained_model(EPM.model_pretrain_dict['VGG19'], labels_train, images_train, images_test, paths_test, N_RANK)