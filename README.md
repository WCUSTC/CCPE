# CCPE
Pytorch implementation of the paper "Wildfire Smoke Detection with Cross Contrast Patch Embedding"

## Introduction
- **Cross Contrast Patch Embedding (CCPE) module**

    We propose the Cross Contrast Patch Embedding (CCPE) module based on the Swin Transformer. This module leverages multi-scale spatial contrast information in both vertical and horizontal directions to enhance the network's discrimination of underlying details. By combining Cross Contrast with Transformer, we exploit the advantages of Transformer in global receptive field and context modeling while compensating for its inability to capture very low-level details, resulting in a more powerful backbone network tailored for smoke recognition tasks.

- **Separable Negative Sampling Mechanism (SNSM)**
 
    This mechanism alleviates supervision signal confusion during network training by employing different negative instance sampling strategies on positive and negative images.

- **SKLFS-WildFire Test dataset**
 
     The largest real wildfire test set to date, containing 50,735 images from 3,649 video clips, to evaluate our method and facilitate future research.



## How to Install
The installation is exactly the same as [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection). This project is constructed based on the [mmdetection-V2.25.0](https://github.com/open-mmlab/mmdetection/tree/2.x)

After installing mmdetection, users can copy the files and folders of this project to the mmdetection directory and replace files with the same name. 

Our changes to native mmdetection are as follows:
```
mmdetection-master
├── mmdet
│   ├── datasets
│   │   ├── __init__.py
│   │   ├── fire_smoke.py
│   ├── models
│   │   ├── utils
│   │   │   ├── __init__.py
│   │   │   ├── transformer.py
│   │   ├── backbones
│   │   │   ├── __init__.py
│   │   │   ├── swin.py
│   │   ├── dense_heads
│   │   │   ├── __init__.py
│   │   │   ├── yolox_head.py
├── ConfigFIgLib
├── ConfigSKLFS
```

## Data Preparation
- **FIgLib**

  Organize the dataset in the format of VOC 2007. Each subset contains a JPEGImages folder for storing images, Annotations for storing XML annotation files, and a txt format sample list
```
FIgLib
├── train
│   ├── Annotations
│   ├── JPEGImages
│   ├── FIgLib_train.txt
├── val
│   ├── Annotations
│   ├── JPEGImages
│   ├── FIgLib_val.txt
├── test
│   ├── Annotations
│   ├── JPEGImages
│   ├── FIgLib_test.txt
```
- **SKLFS-WildFire**
  
  Same as FIgLib. However, SKLFS-WildFire does not set a validation set, and users can self split from the training set
```
SKLFS-WildFire
├── train
│   ├── Annotations
│   ├── JPEGImages
│   ├── SKLFS-WildFire_train.txt
├── test
│   ├── Annotations
│   ├── JPEGImages
│   ├── SKLFS-WildFire_test.txt
```

## Getting Started
- **Single Frame FIgLib Training**

  For training, run
  ```Shell
  python tools/train.py [path_to_your_config] 
  ```
  For example, run
  ```Shell
  python tools/train.py ConfigFIgLib/yolox_swintContrast_SepInd10#OHEM190_ftFIgLib1024.py
  ```

- **Single Frame FIgLib testing**

  For testing, run
    ```Shell
    python tools/test.py [path_to_your_config]  [path_to_your_checkpoint] --eval bbox
    ```


  For example, run
  ```Shell
  python tools/test.py ConfigFIgLib/yolox_swintContrast_SepInd10#OHEM190_ftFIgLib1024.py work_dirs_wildfire/yolox_swintContrast_SepInd10#OHEM190_ftFIgLib1024/epoch_80.pth   --eval bbox
  ```

- **Mulit-Frame FIgLib Training**

    ```Shell
    python tools/train.py ConfigFIgLib/yolox_swintContrast_SepInd10#OHEM190_ftFIgLib2m.py
    ```

- **Mulit-Frame FIgLib testing**
  For example, run
  ```Shell
  python tools/test.py ConfigFIgLib/yolox_swintContrast_SepInd10#OHEM190_ftFIgLib2m.py work_dirs_wildfire/yolox_swintContrast_SepInd10#OHEM190_ftFIgLib2m/epoch_30.pth   --eval bbox
  ```


## Finetuned Models
We will provide the Finetuned Model Links after the review phase

| Model                 | Download     |
|-----------------------|--------------|
| Single Frame FIgLib   | [download]() |
| Mulit-Frame FIgLib    |[download]()  |
| SKLFS-WildFire    |[download]()  |





## Citation (Not Available)

If you use this code or dataset in your research, please cite this project.

```
@article{mmdetection,
  title   = {Wildfire Smoke Detection with Cross Contrast Patch Embedding},
  author  = {},
  journal= {},
  year={}
}
```

