# Evaluating Pre-trained Convolutional Neural Networks and Foundation Models as Feature Extractors for Content-based Medical Image Retrieval

## Dataset: a subset of MedMNIST V2 
![Project Image](https://github.com/masih4/MedImageRetrieval/blob/master/Untitled.png)
## Feature extractors:
- VGG19
- ResNet50
- DenseNet121
- EfficientNetV2N
- CCL
- SAM
- MedSAM
- BioMedClip

## Similarity:
Cosine similarity index

## Evaluation:
- mAP@5
- HitRate@5
- MMV@5
- ACC@1
- ACC@3
- ACC@5 ( = HitRate@5)

## How to run:
- For pre-trained CNNs for 2D datasets run `main_2D_CNN.py`
- For pre-trained CNNs for 2D datasets (if your memory is limited to load the entire data) run `main_2D_CNN_slow.py`
- For pre-trained BioMedClip for 2D datasets run `main_2D_biomedclip.py`
- For pre-trained BioMedCLip for 2D datasets (if your memory is limited to load the entire data) run `main_2D_Biomedclip_slow.py`
- For pre-trained models (CNN or BioMedClip) for 3D datasets run `main_3D.py`
- To create bar charts run `plot.py`

## Citation:
Will be updated
```
@article{,
title = "Evaluating Pre-trained Convolutional Neural Networks and Foundation Models as Feature Extractors for Content-based Medical Image Retrieval",
journal = "",
volume = "",
pages = "",
year = "",
doi = "",
author = "Amirreza Mahbod and Nematollah Saeidi and Ramona Woitek"
}
```


## Contact:
Amirreza Mahbod
amirreza.mahbod@dp-uni.ac.at





