编辑中
# 基于视觉先验自适应挖掘的遥感影像语义分割

## 本代码主要在以下环境中测试：

- pytorch = 2.8.0
- torchvision
- CUDA = 2.8
- 此外，本项目使用[DINOv3](https://github.com/facebookresearch/dinov3)，请根据实际运行环境配置 DINOv3 相关依赖和预训练权重。
## Dataset

本文主要在两个高分辨率遥感语义分割公开数据集上进行实验。
[第三方下载源](https://notes.smallbamboo.cn/ai-splitting-method-of-vaihingen-potsdam-datasets.html):[https://notes.smallbamboo.cn/ai-splitting-method-of-vaihingen-potsdam-datasets.html](https://notes.smallbamboo.cn/ai-splitting-method-of-vaihingen-potsdam-datasets.html)
### ISPRS Vaihingen Dataset

- 数据类型：高分辨率航空真正射影像
- 使用波段：近红外、红、绿
- 语义类别：不透水面、建筑物、低植被、树木、汽车、背景/杂乱区域

### ISPRS Potsdam Dataset

- 数据类型：高分辨率航空真正射影像
- 使用波段：红、绿、蓝
- 语义类别：不透水面、建筑物、低植被、树木、汽车、背景/杂乱区域

## Training and Evaluating
The pipeline for training with CLFDA is the following:

1. "option.py"
- `args.dataset = 'gid15'`
- `args.arch = 'CLFDA'`

2. Train and Evaluate the model. For example, to run an experiment for the GID-15 and FUSU dataset,  run:

- `python main.py`


## Acknowledgment
This code is heavily borrowed from [RS-Mamba(rsm-ss)](https://github.com/walking-shadow/Official_Remote_Sensing_Mamba), [RS3Mamba](https://github.com/sstary/SSRS/tree/main/RS3Mamba) and [DLTH](https://github.com/yueb17/DLTH)


## Citation
If you find our work useful in your research, please consider citing our paper:

```

```
## Contact
Please contact houdongyang1986@163.com if you have any question on the codes.
