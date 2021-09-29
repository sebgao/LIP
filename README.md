# LIP: Local Importance-based Pooling

PyTorch implementations of LIP (ICCV 2019).

[[arxiv link]](https://arxiv.org/abs/1908.04156)

This codebase is now complete and it contains:

- [x] the implementation of LIP based on PyTorch primitives,
- [x] LIP-ResNet,
- [x] LIP-DenseNet,
- [x] ImageNet training and testing code,
- [x] CUDA implementation of LIP.

## News

[2021] A case of LIP when `G(I)=I`, SoftPool, is accepted to ICCV 2021. Check [SoftPool](https://github.com/alexandrosstergiou/SoftPool).

## A Simple Step to Customize LIP
LIP as the learnable generic pooling, its code is simply (in PyTorch):
```
def lip2d(x, logit, kernel=3, stride=2, padding=1):
    weight = logit.exp()
    return F.avg_pool2d(x*weight, kernel, stride, padding)/F.avg_pool2d(weight, kernel, stride, padding)
```

You need a sub fully convolutional network (FCN) as the logit module (whose output is of the same shape as the input) to produce the logit. You can customize the logit module like
```
logit_module_a = nn.Identity()
lip2d(x, logit_module_a(x)) // it gives SoftPool

logit_module_b = lambda x: x.mul(20)
lip2d(x, logit_module_b(x)) // it approximates max pooling

logit_module_c = lambda x: x.mul(0)
lip2d(x, logit_module_c(x)) // it is average pooling

logit_module_d = MyLogitModule() // Your customized logit module (a FCN) begins here
lip2d(x, logit_module_d(x))

```

## Dependencies
1. Python 3.6
2. PyTorch 1.0
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
