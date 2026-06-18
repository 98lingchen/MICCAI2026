# Precision Recall Controllable Radiology Report Generation via Hybrid Natural Language and Clinical Reward Learning

This repository contains the implementation for our method that enables continuous control of clinical precision and recall via a tunable parameter λ using group-relative reinforcement learning.


### Requirements

torch==1.7.1
torchvision==0.8.2
opencv-python==4.4.0.42
f1chexbert



## Datasets

For MIMIC-CXR, you can download the dataset from [here](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) and then put the files in `data/mimic_cxr`.

### Train

Run `bash run_train.sh` to train a model on the MIMIC-CXR data.



### Test

Run `bash run_test.sh` to test a model on the MIMIC-CXR data.

## Pretrained Model The pretrained model and related files can be downloaded from [[Google Drive]](https://drive.google.com/drive/folders/1vqyvFI3SB4hDb--3-PMI_hV2g75NdNz2?usp=sharing).
Please place the downloaded files in the corresponding checkpoint directory before testing. The modified version of f1chexbert is also included here.



## Acknowledgments

Thanks for the open source code in  "Generating Radiology Reports via Memory-driven Transformer", "Reinforced Cross-modal Alignment for Radiology Report Generation", and "LHR-RFL: Linear Hybrid-Reward-Based Reinforced Focal Learning for Automatic Radiology Report Generation".
