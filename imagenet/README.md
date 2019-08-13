## Getting Started
Our training and testing code `main.py` is modified from [PyTorch official example one](https://github.com/pytorch/examples/tree/master/imagenet). You can refer to [this](https://github.com/pytorch/examples/tree/master/imagenet) for preparing ImageNet dataset and dependencies.

## Training
The code `main.py` is specialized in single node, multiple GPUs training for faster speed. You can configure settings in `train-lip_resnet50` in `Makefile` and then train LIP-ResNet-50 by simply
```
make train-lip_resnet50
```

## Evaluating
Alike, you can evaluate the models by
```
make val-lip_resnet50
```

You can place our pretrained models in this directory and evaluate them. The results should be
```
LIP-ResNet-50
Epoch [0] * Acc@1 78.186 Acc@5 93.964

LIP-ResNet-101
Epoch [0] * Acc@1 79.330 Acc@5 94.602
```