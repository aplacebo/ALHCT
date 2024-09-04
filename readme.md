


# ALHCT
This repository is the official implementation of [''Dual Attention Transformers:  Adaptive Linear and Hybrid Cross Attention for Remote Sensing Classification''].

## Abstract
In remote sensing image classification, Convolutional Neural Networks (CNNs) are widely used but are limited by their local receptive fields, making it challenging to extract multi-scale features from images with varying scales and object sizes. Vision Transformer (ViT)-based methods, while effective at capturing global information, often struggle to effectively identify critical objects. To address this issue, we propose an Adaptive Linear Hybrid Cross Attention Transformer (ALHCT). It integrates Adaptive Linear (AL) attention and Hybrid Cross (HC) attention to learn local and global representations simultaneously. AL reduces the computational complexity of attention from exponential to linear scale, and HC fuses and enhances local and global features, improving the network’s global perception and discriminative power. The ALHCT architecture also integrates two adaptive linear Swin Transformers (ALST) for different resolutions, achieving multi-scale feature representation and capturing both high-level semantics and fine details. Experiments on three remote sensing datasets show that ALHCT significantly improves classification accuracy in intricate remote sensing scenes, outperforming several state-of-the-art methods.


## Installation

To install requirements:

```setup
pip install -r requirements.txt
```
With conda:

```
conda create -n pytorch python=3.6.13
conda activate pytorch
conda install pytorch=1.7.1 torchvision  cudatoolkit=11.0 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Data preparation
Download three remote sensing datasets for image classification :  
AID: https://captain-whu.github.io/AID/  
UC-Merced: http://weegee.vision.ucmerced.edu/datasets/landuse.html  
NWPU-RESISC45: http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html  

The three datasets are divided according to a common training-test ratio.

```
/path/to/datasets/
AID/UC-Merced/NWPU-RESISC45
├── train
│   ├── class1
│   │   ├── img1.jpeg
│   │   ├── img2.jpeg
│   │   └── ...
│   ├── class2
│   │   ├── img3.jpeg
│   │   └── ...
│   └── ...
└── val
    ├── class1
    │   ├── img4.jpeg
    │   ├── img5.jpeg
    │   └── ...
    ├── class2
    │   ├── img6.jpeg
    │   └── ...
    └── ...
```


## Pretrained model  
We provide several pretrained models trained on ImageNet1K (http://image-net.org/).
And you can load pretrained models from the folder `pretrained`.  


## Train model  
To train `flatten_swin_b_w7_224` `flatten_swin_b_w7_384` on three remote sensing datasets with 1 gpu for 100 epochs run, respectively.  

```shell 
python3 train.py --batch_size 56 --epochs 100 --data_use ucm  --data_ratio 0.5 --pthpath ./pth/ucm0_8/  
python3 train.py --batch_size 56 --epochs 100 --data_use ucm  --data_ratio 0.5 --pthpath ./pth/ucm0_5/  
python3 train.py --batch_size 56 --epochs 100 --data_use aid  --data_ratio 0.2 --pthpath ./pth/aid0_2/
python3 train.py --batch_size 56 --epochs 100 --data_use aid  --data_ratio 0.5 --pthpath ./pth/aid0_5/
python3 train.py --batch_size 56 --epochs 100 --data_use nwpu  --data_ratio 0.2 --pthpath ./pth/nwpu0_2/
python3 train.py --batch_size 56 --epochs 100 --data_use nwpu  --data_ratio 0.5 --pthpath ./pth/nwpu0_5/

```
In addition, we replaced the above two models with models `swin_b_w7_224` `swin_b_w12_384`  for the same experiments.

## Contact

If you have any questions, please feel free to contact the authors.  
Yake Zhang: [yake_1023@163.com](mailto:yake_1023@163.com)

Yufan Zhao:  [2067058843@qq.com](mailto:2067058843@qq.com)

