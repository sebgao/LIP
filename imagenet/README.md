## Summary

| Model            | Acc@1   | Acc@5   | #Params | FLOPs    | Inf Time* |
| ---------------- | ------- | ------- | ------- | -------- | --------- |
| ResNet-50        | 76.40   | 93.15   | 25.6M   | 4.12G    | 1.41ms    |
| LIP-ResNet-50    | 78.19   | 93.96   | 23.9M   | 5.33G    | 1.88ms    |
|                  | _+1.79_ | _+0.81_ | _-6.6%_ | _+29.4%_ | _+33%_    |
| ResNet-101       | 77.98   | 93.98   | 44.5M   | 7.85G    | 2.29ms    |
| LIP-ResNet-101   | 79.33   | 94.60   | 42.9M   | 9.06G    | 2.77ms    |
|                  | _+1.46_ | _+0.62_ | _-3.6%_ | _+15.4%_ | _+21%_    |
| DenseNet-121     | 75.62   | 92.56   | 8.0M    | 2.88G    | 1.49ms    |
| LIP-DenseNet-121 | 76.64   | 93.16   | 8.7M    | 4.13G    | 1.80ms    |
|                  | _+1.02_ | _+0.60_ | _+8.8%_ | _+43.4%_ | _+21%_    |


\* average inference time of a single image. The results are calculated by repeated inference with batch size 32 on a single Titan XP card.

\** LIP models here denote the full LIP architectures with Bottleneck-128 logit modules.

## Getting Started
Our training and testing code `main.py` is modified from [PyTorch official example one](https://github.com/pytorch/examples/tree/master/imagenet). You can refer to [this](https://github.com/pytorch/examples/tree/master/imagenet) for preparing ImageNet dataset and dependencies.

## Training
The code `main.py` is specialized in single node, multiple GPUs training for the faster speed. You can configure settings in `train-lip_resnet50` in `Makefile` and then train LIP-ResNet-50 by simply
```
make train-lip_resnet50
```

You can resort to [PyTorch official example one](https://github.com/pytorch/examples/tree/master/imagenet) if the command above fails. For that, you need to modify the learning rate schedule to be consistent with the paper, i.e., lr decays 10x at the 30, 60, 80-th epoch.

## Evaluating
Alike, you can evaluate the models by
```
make val-lip_resnet50
make val-lip_resnet101
make val-lip_densenet121
```

You can place [our pretrained models](https://drive.google.com/drive/folders/1KCt22JTob1hHiPmpLOlgZo3fvTRc11SJ) in this directory and evaluate them. The results should be
```
LIP-ResNet-50
Epoch [0] * Acc@1 78.186 Acc@5 93.964

LIP-ResNet-101
Epoch [0] * Acc@1 79.330 Acc@5 94.602

LIP-DenseNet-121
Epoch [0] * Acc@1 76.636 Acc@5 93.156
```