# Precision Recall Controllable Radiology Report Generation via Hybrid Natural Language and Clinical Reward Learning

This repository contains the implementation for our method that enables continuous control of clinical precision and recall via a tunable parameter λ using group-relative reinforcement learning.


### Requirements

torch==1.7.1
torchvision==0.8.2
opencv-python==4.4.0.42
f1chexbert



## Datasets

For MIMIC-CXR, you can download the dataset from [here]([https://physionet.org/content/mimic-cxr/](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)) and then put the files in `data/mimic_cxr`.

### Train

Run `bash train.sh` to train a model on the MIMIC-CXR data.



### Test

Run `bash test.sh` to train a model on the MIMIC-CXR data.



## Acknowledgments

Thanks for the open source code in  "Generating Radiology Reports via Memory-driven Transformer", and "LHR-RFL: Linear Hybrid-Reward-Based Reinforced Focal Learning for Automatic Radiology Report Generation".
