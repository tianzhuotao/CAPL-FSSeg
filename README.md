# CAPL-FSSeg
This repo provides the implementation of CAPL in FS-Seg.

# Get Started

### Environment
+ Python 3.7.9
+ Torch 1.5.1
+ cv2 4.4.0
+ numpy 1.21.0
+ CUDA 10.1

### Datasets and Data Preparation
Please prepare the datasets (COCO-20i and PASCAL-5i) by following the instructions of [**PFENet**](https://github.com/dvlab-research/PFENet). 

### Run Demo / Test with Pretrained Models
+ Execute `mkdir initmodel` at the root directory.
+ Download the ImageNet pretrained [**backbones**](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155122171_link_cuhk_edu_hk/EQEY0JxITwVHisdVzusEqNUBNsf1CT8MsALdahUhaHrhlw?e=4%3a2o3XTL&at=9) and put them into the `initmodel` directory.
+ Please download the pretrained models.
+ We provide **16 pre-trained**  [**models**](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155122171_link_cuhk_edu_hk/Ek9AzBjMQ8xBn1MOx2fUqSoBXmLdlGmwveD2ilj32_J1qA?e=DKNZHK): 
8 for 1/5 shot results on PASCAL-5i and 8 for COCO.
+ Update the config files by speficifying the target **split**, **weights** and **val_shot** for loading different checkpoints.


### Train / Evaluate
+ For training, please set the option **only_evaluate** to **False** in the configuration file. Then execute this command at the root directory: 

    sh train.sh {*dataset*} {*model_config*}
    
+ For evaluation only, please set the option **only_evaluate** to **True** in the corresponding configuration file. 

    
Example: Train / Evaluate CAPL with 1-shot on the split 0 of PASCAL-5i: 

    sh train.sh pascal split0_1shot   
    
    
# Acknowledgement

We gratefully thank the authors of [**PANet**](https://github.com/kaixin96/PANet) and [**PPNet**](https://github.com/Xiangyi1996/PPNet-PyTorch) that inspire our implementation.

# Citation

If you find this project useful, please consider citing:
```
@InProceedings{tian2022gfsseg,
    title={Generalized Few-shot Semantic Segmentation},
    author={Zhuotao Tian and Xin Lai and Li Jiang and Shu Liu and Michelle Shu and Hengshuang Zhao and Jiaya Jia},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2022}
}
```
