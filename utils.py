import numpy as np
from tqdm import tqdm
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
