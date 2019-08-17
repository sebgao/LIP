# LIP: Local Importance-based Pooling

PyTorch implementations of LIP (ICCV 2019).

[[arxiv link]](https://arxiv.org/abs/1908.04156)

This codebase is __under construction__ and will contain:

- [x] the implementation of LIP based on PyTorch primitives,
- [x] LIP-ResNet,
- [ ] LIP-DenseNet,
- [x] ImageNet training and testing code,
- [ ] CUDA implementation of LIP.

## NEWS

[8.13] We uploaded the code of LIP-ResNets and their ImageNet pretrained models (LIP-ResNet-50 & 101).

[8.17] Fixed the missing `init_lr` key and the possible in_place `mul_` operation problem (reported in PyTorch 1.1).

## Dependencies
1. Python 3.6
2. PyTorch 1.0+
3. tensorboard and tensorboardX

## Pretrained Models
You can download ImageNet pretrained models [here](https://drive.google.com/drive/folders/1KCt22JTob1hHiPmpLOlgZo3fvTRc11SJ).

## ImageNet
Please refer to [imagenet/README.md](./imagenet/).
