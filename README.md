# ReadMe

## Getting started

Let's start by installing all the dependencies.

```shell
pip3 install -r requirement.txt
```

## Training

We employ `cifar10_RT_ER.py` and `cifar10_JTDMoE.py` to adversarially train an MoE model and a dual-model on the CIFAR-10 dataset using Resnet18. Similarly, we use `tinyimagenet_RT_ER.py` and `tinyimagenet_JTDMoE.py` for adversarial training of an MoE model and a dual-model on the TinyImageNet dataset using ViT-small. Below, we outline the key arguments and their usage.

- `--net` Specifies the model architecture used for training.
- `--resume` Defines the path to the checkpoint for evaluation or resuming training.
- `--size`Sets the size of the input image.
- `--beta`Controls the proportion of expert contributions in the model.
- `--alpha`Adjusts the contribution of the robust model within the dual-model.

## Evaluation with AutoAttack

Please use the file `AutoAttack.py` to evaluate the model using AutoAttack.

## Commands

To train an MoE on CIFAR-10 dataset using RT-ER method:

```shell
python3 cifar10_RT_ER.py --net res18_moe
```

To joint-training a dual-model on CIFAR-10 dataset using JTDMoE method:

```shell
python3 cifar10_JTDMoE.py --net dual_model_res18
```

To train an MoE on TinyImageNet dataset using RT-ER method:

```shell
python3 tinyimagenet_RT_ER.py --net vit_moe
```

To joint-training a dual-model on TinyImageNet dataset using JTDMoE method:

```shell
python3 tinyimagenet_JTDMoE.py --net dual_model_vit
```

## Dataset Preparation

### TinyImageNet

To obtain the original TinyImageNet dataset, please run the following scripts:

```shell
cd data/tiny_imagenet
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip -qq 'tiny-imagenet-200.zip'
rm tiny-imagenet-200.zip
```