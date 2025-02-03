[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

# Evaluating Pre-trained Convolutional Neural Networks and Foundation Models as Feature Extractors for Content-based Medical Image Retrieval

## Preprint:
https://www.arxiv.org/abs/2409.09430 

## Dataset: a subset of MedMNIST V2 
![Project Image](https://github.com/masih4/MedImageRetrieval/blob/master/Untitled.png)
## Feature extractors:
- VGG19
- ResNet50
- DenseNet121
- EfficientNetV2M
- MedCLIP
- BioMedClip
- OpenCLIP
- CONCH
- UNI 

## Similarity:
Cosine similarity index

## Evaluation:
- mAP@5
- MMV@5
- ACC@1
- ACC@3
- ACC@5

## Results:
- Main results in the manuscript
- More detailed results are available in ["Supplementary Materials"](https://github.com/masih4/MedImageRetrieval/blob/master/Supplementary%20Materials.pdf)
- Selective results based on ACC@1:
##### ACC@1 for different datasets based on the models and input image size
<div style="display: flex; justify-content: space-around;">
  <img src="figures/BreastMNIST_ACC@1.png" alt="" width="400"/>
  <img src="figures/PneumoniaMNIST_ACC@1.png" alt="" width="400"/>
</div>

<div style="display: flex; justify-content: space-around;">
  <img src="figures/RetinaMNIST_ACC@1.png" alt="" width="400"/>
  <img src="figures/DermaMNIST_ACC@1.png" alt="" width="400"/>
</div>

<div style="display: flex; justify-content: space-around;">
  <img src="figures/BloodMNIST_ACC@1.png" alt="" width="400"/>
  <img src="figures/PathMNIST_ACC@1.png" alt="" width="400"/>
</div>

<div style="display: flex; justify-content: space-around;">
  <img src="figures/AdrenalMNIST3D_ACC@1.png" alt="" width="400"/>
  <img src="figures/SynapseMNIST3D_ACC@1.png" alt="" width="400"/>
</div>


## How to run:
- For Tensorflow implementation of pre-trained CNNs for 2D datasets run `main_2D_CNN.py`
- For Tensorflow implementation of pre-trained CNNs for 2D datasets (if your memory is limited to load the entire data) run `main_2D_CNN_memory_safe`
- For pre-trained foundation models for 2D datasets run `main_2D_foundation.py`
- For pre-trained foundation models for 2D datasets (if your memory is limited to load the entire data) run `main_2D_foundation_memory_safe `
- For pre-trained models (CNN or foundation) for 3D datasets run `main_3D.py` (The CNN models in this script are based on Tensorflow implementation)
- To create bar charts run `plot.py`
- For the PyTorch implementation of pre-trained CNNs for 2D datasets, run `main_2D_CNN_memory_safe_pytorch.py`. For the PyTorch implementation of pre-trained CNNs for 3D datasets, run `main_3D_CNN_pytorch.py`.Please note that the reported results for CNN models in the paper are based on the TensorFlow implementation.

## Used repositories/sources:
- For CNNs: https://www.tensorflow.org/api_docs/python/tf/keras/applications
- For MedCLIP: https://github.com/RyanWangZf/MedCLIP
- For BioMedCLIP: https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/blob/main/biomed_clip_example.ipynb
- For OpenCLIP: https://github.com/mlfoundations/open_clip
- For CONCH: https://github.com/mahmoodlab/CONCH
- For UNI: https://github.com/mahmoodlab/UNI

## Citation:
```
@article{mahbod2024evaluating,
  title={Evaluating Pre-trained Convolutional Neural Networks and Foundation Models as Feature Extractors for Content-based Medical Image Retrieval},
  author={Mahbod, Amirreza and Saeidi, Nematollah and Hatamikia, Sepideh and Woitek, Ramona},
  journal={arXiv preprint arXiv:2409.09430},
  year={2024}
}
```


## Contact:
Amirreza Mahbod
amirreza.mahbod@dp-uni.ac.at

## Main References:
- VGG19: Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).
- ResNet50: He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
- DenseNet121: Huang, Gao, et al. "Densely connected convolutional networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
- EfficientNetV2M: Tan, Mingxing, and Quoc Le. "Efficientnetv2: Smaller models and faster training." International conference on machine learning. PMLR, 2021.
- MedCLIP: Wang, Zifeng, et al. "Medclip: Contrastive learning from unpaired medical images and text." arXiv preprint arXiv:2210.10163 (2022).
- BioMedClip: Zhang, Sheng, et al. "BiomedCLIP: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs." arXiv preprint arXiv:2303.00915 (2023).
- OpenCLIP: Cherti, Mehdi, et al. "Reproducible scaling laws for contrastive language-image learning." Proceedings of the IEEE/CVF Conference on - Computer Vision and Pattern Recognition. 2023.
- CONCH: Lu, Ming Y., et al. "Visual language pretrained multiple instance zero-shot transfer for histopathology images." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2023.
- UNI: Chen, Richard J., et al. "Towards a general-purpose foundation model for computational pathology." Nature Medicine 30.3 (2024): 850-862.






