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

[8.13] We uploaded the code of LIP-ResNets and their ImageNet pretrained models (LIP-ResNet-50 & 101).

[8.17] Fixed the missing `init_lr` key and the possible in_place `mul_` operation problem (reported in PyTorch 1.1).

[9.5] CUDA LIP is now available.

[11.6] The LIP-DensNet model is available (sorry for my procrastination). Change the name of `ProjectionLIP` to `SimplifiedLIP` for clarification.

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