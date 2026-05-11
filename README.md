# 基于视觉先验自适应挖掘的遥感影像语义分割

## 本代码主要在以下环境中测试：

- pytorch = 2.8.0
- torchvision
- CUDA = 12.8
- 此外，本项目使用[DINOv3](https://github.com/facebookresearch/dinov3)，请根据实际运行环境配置 DINOv3 相关依赖和预训练权重。
## 数据集

本文主要在两个高分辨率遥感语义分割公开数据集上进行实验。
[第三方下载源](https://notes.smallbamboo.cn/ai-splitting-method-of-vaihingen-potsdam-datasets.html):[https://notes.smallbamboo.cn/ai-splitting-method-of-vaihingen-potsdam-datasets.html](https://notes.smallbamboo.cn/ai-splitting-method-of-vaihingen-potsdam-datasets.html)
### ISPRS Vaihingen

- 数据类型：高分辨率航空真正射影像
- 使用波段：近红外、红、绿
- 语义类别：不透水面、建筑物、低植被、树木、汽车、背景/杂乱区域

### ISPRS Potsdam

- 数据类型：高分辨率航空真正射影像
- 使用波段：红、绿、蓝
- 语义类别：不透水面、建筑物、低植被、树木、汽车、背景/杂乱区域

## 训练与测试
VPAMNet 模型训练流程如下:

1. 下载DINOv3预训练权重dinov3_vits16_pretrain_lvd1689m

2. "option.py"
- `args.dataset = 'Vaihingen__'`
- `args.arch = 'Potsdam_'`

3. 训练与评估模型。 例如，在 GID-15 和 FUSU 数据集上运行实验:

- `python main.py`


## 致谢
本代码参考和借鉴了以下开源项目，在此表示感谢： [DINOv3](https://github.com/facebookresearch/dinov3), [DinoUNet](https://github.com/yifangao112/DinoUNet) and [DLTH](https://github.com/yueb17/DLTH)


## 引用
如果本工作对您的研究有帮助，请考虑引用我们的论文:

```

```
## 联系方式
如对代码有任何疑问，请联系 ***@163.com
