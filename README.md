# RGB-T-Glass-Segmentation

Code for this paper [Glass Segmentation with RGB-Thermal Image Pairs](https://arxiv.org)

Dong Huo, Jian Wang, Yiming Qian, Yee-Hong Yang

## Overview

This paper proposes a new glass segmentation method utilizing paired RGB and thermal images. Due to the large difference between the transmission property of visible light and that of the thermal energy through the glass where most glass is transparent to the visible light but opaque to thermal energy, glass regions of a scene are made more distinguishable with a pair of RGB and thermal images than solely with an RGB image. To exploit such a unique property, we propose a neural network architecture that effectively combines an RGB-thermal image pair with a new multi-modal fusion module based on attention. As well, we have collected a new dataset containing 5551 RGB-thermal image pairs with ground-truth segmentation annotations. The qualitative and quantitative evaluations demonstrate the effectiveness of the proposed approach on fusing RGB and thermal data for glass segmentation.

## Architecture

<p align="center">
  <img width="1000" src="./images/architecture.png">
</p>


## Datasets

The datasets for training can be downloaded via the links below:
- [Our RGB-T dataset](https://drive.google.com/file/d/1ysG04qGmnZv7UaybZUuyybaJYJLUkNHX/view?usp=sharing)
- [GDD](https://mhaiyang.github.io/CVPR2020_GDNet/index)

## Prerequisites
- Python 3.8 
- PyTorch 1.9.0
- Requirements: opencv-python
- Platforms: Ubuntu 20.04, RTX A6000, cuda-11.1

## Training

```python main.py```

Modify the arguments in parse_args()


## Testing

``` python main.py --resume checkpoints_path --eval```
Download the well-trained [models](-) 

Baidu netdisk users can also download the model from [link](-)  with password: -
