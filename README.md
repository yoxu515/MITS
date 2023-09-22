# MITS

## Introduction
Integrating Boxes and Masks: A Multi-Object Framework for Unified Visual Tracking and Segmentation. [[Arxiv]](https://arxiv.org/abs/2308.13266)

Tracking any given object(s) spatially and temporally is a common purpose in Visual Object Tracking (VOT) and Video Object Segmentation (VOS). We proposed a Multi-object Mask-box Integrated framework for unified Tracking and Segmentation, dubbed MITS. 
- Firstly, the unified identification module is proposed to support both box and mask reference for initialization, where detailed object information is inferred from boxes or directly retained from masks. 
- Additionally, a novel pinpoint box predictor is proposed for accurate multi-object box prediction, facilitating target-oriented representation learning. 
- All target objects are processed simultaneously from encoding to propagation and decoding, which enables our framework to handle complex scenes with multiple objects efficiently. 

## Requirements
* Python3
* pytorch >= 1.7.0 and torchvision
* opencv-python
* Pillow
* Pytorch Correlation. Recommend to install from [source](https://github.com/ClementPinard/Pytorch-Correlation-extension) instead of using `pip`:
    ```bash
    git clone https://github.com/ClementPinard/Pytorch-Correlation-extension.git
    cd Pytorch-Correlation-extension
    python setup.py install
    cd -
    ```
    
## Getting Started

### Training
#### Data Preparation
- VOS datasets: YouTube-VOS 2019 train, DAVIS 2017 train
- VOT datasets: LaSOT train, GOT-10K train

Download datasets and re-organize the folders as the following structure:
```
datasets
└───YTB
    └───2019
        └───train
            └───JPEGImages
            └───Annotations
└───DAVIS
    └───JPEGImages
    └───Annotations
    └───ImageSets
└───LaSOT
    └───JPEGImages
    └───Annotations
    └───BoxAnnotations
└───GOT10K
    └───JPEGImages
    └───Annotations
    └───BoxAnnotations
```

- **JPEGImages** and **Annotations** includes subfolders for video sequences containing images and masks. **BoxAnnotations** contains txt annotation files for every video sequence.
- The standard training of MITS keeps the same training data as a prior work [RTS](https://arxiv.org/abs/2203.11191). The download link for pseudo-masks for LaSOT and GOT10k can be found [here](https://github.com/visionml/pytracking/blob/master/ltr/README.md#RTS). The corresponding folders are **Annotations** of LaSOT and GOT10K.
- For LaSOT training set, training frames are sampled every 5 frames from the original sequences, according to the pseudo-masks provided by RTS. This may require to sample frame images and box annotations from the original LaSOT data source.

**Note**: Although the pseudo masks are used for training by default, MITS can also be trained with mixed annotations without pseudo masks due to its strong compatibility.
#### Pretrain Weights
MITS is initialized with pretrained [DeAOT](https://github.com/yoxu515/aot-benchmark/blob/main/MODEL_ZOO.md). Download the [R50_DeAOTL_PRE.pth](https://drive.google.com/file/d/1sTRQ1g0WCpqVCdavv7uJiZNkXunBt3-R) and put it to pretrain_models folder.

#### Train Models
Run ```train.sh``` to launch training. Configs for different training settings are in folder **configs**. See model zoo for details.


### Evaluation
#### Data Preparation
- VOS datasets: YouTube-VOS 2019 valid, DAVIS 2017 valid
- VOT datasets: LaSOT test, TrackingNet test, GOT-10K test

We follow the original file structure from each dataset. First frame mask for YouTube-VOS/DAVIS and first frame box for LaSOT/TrackingNet/GOT-10K are required for evaluation.

```
datasets
└───YTB
    └───2019
        └───valid
            └───JPEGImages
            └───Annotations
└───DAVIS
    └───JPEGImages
    └───Annotations
    └───ImageSets
└───LaSOTTest
    └───airplane-1
        └───img
        └───groundtruth.txt
└───TrackingNetTest
    └───JPEGImages
        └───__WaG8fRMto_0
    └───BoxAnnotations
        └───__WaG8fRMto_0.txt
└───GOT10KTest
    └───GOT-10k_Test_000001
        └───00000001.jpg
        ...
        └───groundtruth.txt
```

#### Evaluate Models
Run ```eval_vos.sh``` to evaluate on YouTube-VOS or DAVIS, ```eval_vot.sh``` to evaluate on LaSOT, TrackingNet or GOT10K.

The outputs include predicted masks from mask head, bounding boxes from masks (bbox), predicted boxes from box head (boxh). By default, masks are for VOS benchmarks and boxh boxes are for VOT benchmarks.

## Model Zoo
#### Model Download
|   Model  | Training Data | File                                                                                         |
|:--------:|:-------------:|----------------------------------------------------------------------------------------------|
| MITS     |      full     | [gdrive](https://drive.google.com/file/d/1Db9DxXc-gyRkxhKs0AMXJ2RH6DyHWCfq/view?usp=sharing) |
| MITS_box |  no VOT masks | [gdrive](https://drive.google.com/file/d/1388k-L55W3Q7mzbOgViMn0jxdttcOx2h/view?usp=sharing) |
| MITS_got | only GOT10k   | [gdrive](https://drive.google.com/file/d/18bSHZ22XbTgZAcG2D303pl7ki9ycVu6e/view?usp=sharing) |


#### Evaluation Results
|              Model             |                                             LaSOT Test<br /> AUC/PN/P                                             |                                          TrackingNet Test<br /> AUC/PN/P                                          |                                         GOT10k Test<br /> AO/SR0.5/SR0.75                                         |                                        YouTube-VOS 19 val<br /> G                                       |                                           DAVIS 17 val<br /> G                                          |
|:------------------------------:|:-----------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------:|
| MITS<br /> Prediction file     | 72.1/80.1/78.6<br /> [gdrive](https://drive.google.com/file/d/1WGo4pQa19XGSm2gquXPEmtxwJQZXGw6i/view?usp=sharing) | 83.5/88.7/84.5<br /> [gdrive](https://drive.google.com/file/d/1cR84l5_BJqoqAkhHljFcLMsBH8Mb8jpu/view?usp=sharing) | 78.5/87.5/73.7<br /> [gdrive](https://drive.google.com/file/d/1Fp-O7hogARiKLhVdVtjgXS1Tm_l5w-sb/view?usp=sharing) | 85.9<br /> [gdrive](https://drive.google.com/file/d/1yHBk5FsvVHD9wEKjCWtEOFDX-Zz2x-gY/view?usp=sharing) | 84.9<br /> [gdrive](https://drive.google.com/file/d/1spYFdzeFzS6mN_vg5AhRf3-w2fEkewe_/view?usp=sharing) |
| MITS_box<br /> Prediction file | 70.7/78.1/75.8<br /> [gdrive](https://drive.google.com/file/d/1pEfLw-9-w8nlQD5owTeK7uOqO5EVRo65/view?usp=sharing) | 83.0/87.8/83.1<br /> [gdrive](https://drive.google.com/file/d/1TaGVmaUODGfCZ00u2EvK2BAU8kpV6vtH/view?usp=sharing) | 78.0/86.4/71.7<br /> [gdrive](https://drive.google.com/file/d/1glC55GgzRLMq1VTGE9hkJx3WMl3l7nuj/view?usp=sharing) | 85.7<br /> [gdrive](https://drive.google.com/file/d/1GUiaLZaIFPbgjVi7pq21f11H6WBScvO9/view?usp=sharing) | 84.3<br /> [gdrive](https://drive.google.com/file/d/1M--rlbxlXunWpimtSpGaZ0gAc7lh9g8L/view?usp=sharing) |
| MITS_got<br /> Prediction file |                                                         -                                                         |                                                         -                                                         | 80.4/89.7/75.9<br /> [gdrive](https://drive.google.com/file/d/1gsLfcTIGu6ozVSgJ4r-NQ2t-eJ9unmxr/view?usp=sharing) |                                                    -                                                    |                                                    -                                                    |


By default, we use box prediction for VOT benchmarks and mask prediction for VOS benchmarks. There might be 0.1 performance difference with those reported in the paper due to the code update.


## Acknowledgement
The implementation is heavily based on prior VOS work [AOT/DeAOT](https://github.com/yoxu515/aot-benchmark).

Pseudo-masks for LaSOT and GOT10K for training are taken from [RTS](https://github.com/visionml/pytracking/blob/master/ltr/README.md#RTS).

## Citing
```
@article{xu2023integrating,
  title={Integrating Boxes and Masks: A Multi-Object Framework for Unified Visual Tracking and Segmentation},
  author={Xu, Yuanyou and Yang, Zongxin and Yang, Yi},
  journal={arXiv preprint arXiv:2308.13266},
  year={2023}
}
@inproceedings{yang2022deaot,
  title={Decoupling Features in Hierarchical Propagation for Video Object Segmentation},
  author={Yang, Zongxin and Yang, Yi},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2022}
}
@article{yang2021aost,
  title={Scalable Video Object Segmentation with Identification Mechanism},
  author={Yang, Zongxin and Wang, Xiaohan and Miao, Jiaxu and Wei, Yunchao and Wang, Wenguan and Yang, Yi},
  journal={arXiv preprint arXiv:2203.11442},
  year={2023}
}
@inproceedings{yang2021aot,
  title={Associating Objects with Transformers for Video Object Segmentation},
  author={Yang, Zongxin and Wei, Yunchao and Yang, Yi},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2021}
}
```




