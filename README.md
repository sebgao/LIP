# LIP: Local Importance-based Pooling

PyTorch implementations of LIP (ICCV 2019).

[[arxiv link]](https://arxiv.org/abs/1908.04156)

This codebase is now complete and it contains:

- [x] the implementation of LIP based on PyTorch primitives,
- [x] LIP-ResNet,
- [x] LIP-DenseNet,
- [x] ImageNet training and testing code,
- [x] CUDA implementation of LIP.

## NEWS

[2021] A case of LIP when `G(I)=I`, SoftPool, is accepted to ICCV. Check [SoftPool](https://github.com/alexandrosstergiou/SoftPool).

## Dependencies
1. Python 3.6
2. PyTorch 1.0+
3. tensorboard and tensorboardX

## Pretrained Models
You can download ImageNet pretrained models [here](https://drive.google.com/drive/folders/1KCt22JTob1hHiPmpLOlgZo3fvTRc11SJ).

## ImageNet
Please refer to [imagenet/README.md](./imagenet/).


## CUDA LIP
Please refer to [cuda-lip/README.md](./cuda-lip/).

## Misc
If you find our research helpful, please consider citing our paper.

```
@InProceedings{LIP_2019_ICCV,
author = {Gao, Ziteng and Wang, Limin and Wu, Gangshan},
title = {LIP: Local Importance-Based Pooling},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {October},
year = {2019}
}
```
