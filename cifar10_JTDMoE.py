from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import pandas as pd
import csv
import time
import torch as ch
import torchvision
import torchvision.transforms as transforms
from typing import List
from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage, ImageMixup, LabelMixup
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
from sympy.abc import epsilon
from models import *
from models.moe import *
from models.utils import progress_bar
from models.randomaug import RandAugment
from models.vit import ViT
from models.convmixer import ConvMixer
from attack.PGD import *


def main():
    # parsers
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--opt', default="adam")
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--noaug', action='store_false', help='disable use randomaug')
    parser.add_argument('--noamp', action='store_true',
                        help='disable mixed precision training. for older pytorch versions')
    parser.add_argument('--nowandb', action='store_true', help='disable wandb')
    parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
    parser.add_argument('--net', default='vit')
    parser.add_argument('--dp', action='store_true', help='use data parallel')
    parser.add_argument('--bs', default='512')
    parser.add_argument('--size', default="32")
    parser.add_argument('--n_epochs', type=int, default='200')
    parser.add_argument('--patch', default='4', type=int, help="patch for ViT")
    parser.add_argument('--dimhead', default="512", type=int)
    parser.add_argument('--convkernel', default='8', type=int, help="parameter for convmixer")
    parser.add_argument('--beta', default='6', type=int, help="parameter for the proportion of experts specification")
    parser.add_argument('--alpha', default='0.7', type=float, help="the parameter controlling the contribution of the robust model to the dual-model")

    args = parser.parse_args()

    # take in args
    usewandb = ~args.nowandb
    if usewandb:
        import wandb

        watermark = "{}_lr{}".format(args.net, args.lr)
        wandb.init(project="CVPR2025",
                   name=watermark)
        wandb.config.update(args)

    bs = int(args.bs)
    imsize = int(args.size)

    use_amp = not args.noamp
    aug = args.noaug

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    if args.net == "vit_timm":
        size = 384
    else:
        size = imsize

    # Prepare dataset
    datasets = {
        'train': torchvision.datasets.CIFAR10(root='./data', train=True, download=True),
        'test': torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    }

    for (name, ds) in datasets.items():
        writer = DatasetWriter(f'./data/cifar_{name}.beton', {
            'image': RGBImageField(),
            'label': IntField()
        })
        writer.from_indexed_dataset(ds)

    # Note that statistics are wrt to uin8 range, [0,255].
    CIFAR_MEAN = [125.307, 122.961, 113.8575]
    CIFAR_STD = [51.5865, 50.847, 51.255]

    BATCH_SIZE = bs
    loaders = {}

    a = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for name in ['train', 'test']:
        label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice(a), Squeeze()]
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

        # Add image transforms and normalization
        if name == 'train':
            image_pipeline.extend([
                RandomHorizontalFlip(),
                RandomTranslate(padding=2),
                Cutout(8, tuple(map(int, CIFAR_MEAN))),  # Note Cutout is done before normalization.
            ])

        image_pipeline.extend([
            ToTensor(),
            ToDevice(a, non_blocking=True),
            ToTorchImage(),
            Convert(ch.float16),
            torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

        # Create loaders
        loaders[name] = Loader(f'./data/cifar_{name}.beton',
                               batch_size=BATCH_SIZE,
                               num_workers=8,
                               order=OrderOption.RANDOM,
                               drop_last=(name == 'train'),
                               pipelines={'image': image_pipeline,
                                          'label': label_pipeline})

    # Model factory..
    print('==> Building model..')
    # net = VGG('VGG19')
    if args.net == 'res18':
        net = ResNet18()
    elif args.net == 'vgg':
        net = VGG('VGG19')
    elif args.net == 'res34':
        net = ResNet34()
    elif args.net == 'res50':
        net = ResNet50()
    elif args.net == 'res101':
        net = ResNet101()
    elif args.net == "convmixer":
        # from paper, accuracy >96%. you can tune the depth and dim to scale accuracy and speed.
        net = ConvMixer(256, 16, kernel_size=args.convkernel, patch_size=1, n_classes=10)
    elif args.net == "mlpmixer":
        from models.mlpmixer import MLPMixer

        net = MLPMixer(
            image_size=32,
            channels=3,
            patch_size=args.patch,
            dim=512,
            depth=6,
            num_classes=10
        )
    elif args.net == "vit_small":
        from models.vit_small import ViT

        net = ViT(
            image_size=size,
            patch_size=args.patch,
            num_classes=10,
            dim=int(args.dimhead),
            depth=6,
            heads=8,
            mlp_dim=512,
            dropout=0.1,
            emb_dropout=0.1
        )
    elif args.net == "vit_tiny":
        from models.vit_small import ViT

        net = ViT(
            image_size=size,
            patch_size=args.patch,
            num_classes=10,
            dim=int(args.dimhead),
            depth=4,
            heads=6,
            mlp_dim=256,
            dropout=0.1,
            emb_dropout=0.1
        )
    elif args.net == "simplevit":
        from models.simplevit import SimpleViT

        net = SimpleViT(
            image_size=size,
            patch_size=args.patch,
            num_classes=10,
            dim=int(args.dimhead),
            depth=6,
            heads=8,
            mlp_dim=512
        )
    elif args.net == "vit":
        # ViT for cifar10
        net = ViT(
            image_size=size,
            patch_size=args.patch,
            num_classes=10,
            dim=int(args.dimhead),
            depth=6,
            heads=8,
            mlp_dim=512,
            dropout=0.1,
            emb_dropout=0.1
        )
    elif args.net == "vit_timm":
        import timm

        net = timm.create_model("vit_base_patch16_384", pretrained=True)
        net.head = nn.Linear(net.head.in_features, 10)
    elif args.net == "cait":
        from models.cait import CaiT

        net = CaiT(
            image_size=size,
            patch_size=args.patch,
            num_classes=10,
            dim=int(args.dimhead),
            depth=6,  # depth of transformer for patch to patch attention only
            cls_depth=2,  # depth of cross attention of CLS tokens to patch
            heads=8,
            mlp_dim=512,
            dropout=0.1,
            emb_dropout=0.1,
            layer_dropout=0.05
        )
    elif args.net == "cait_small":
        from models.cait import CaiT

        net = CaiT(
            image_size=size,
            patch_size=args.patch,
            num_classes=10,
            dim=int(args.dimhead),
            depth=6,  # depth of transformer for patch to patch attention only
            cls_depth=2,  # depth of cross attention of CLS tokens to patch
            heads=6,
            mlp_dim=256,
            dropout=0.1,
            emb_dropout=0.1,
            layer_dropout=0.05
        )
    elif args.net == "swin":
        from models.swin import swin_t

        net = swin_t(window_size=args.patch,
                     num_classes=10,
                     downscaling_factors=(2, 2, 2, 1))
    elif args.net == "dual_model_res18":
        net = Dual_Model_resnet18(alpha=args.alpha, size=size)

    # For Multi-GPU
    if 'cuda' in device:
        print(device)
        if args.dp:
            print("using data parallel")
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/' + args.net + '-JTDMoE.t7')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    # Loss is CE
    criterion = nn.CrossEntropyLoss()
    criterion_kl = nn.KLDivLoss(size_average=False)

    if args.opt == "adam":
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    elif args.opt == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=args.lr)

    # use cosine scheduling
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.00005, step_size_up=500, max_lr=args.lr)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # PGD Attacker
    epsilon = 8 / 255
    step_size = 2 / 255
    num_step_train = 10
    num_step_val = 50
    beta = args.beta

    # FFCV [0,255]
    pgd_train = PGD(eps=epsilon, sigma=step_size, nb_iter=num_step_train, DEVICE=device,
                    mean=torch.tensor(np.array(CIFAR_MEAN).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]),
                    std=torch.tensor(np.array(CIFAR_STD).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]))
    pgd_val = PGD(eps=epsilon, sigma=step_size, nb_iter=num_step_val, DEVICE=device,
                  mean=torch.tensor(np.array(CIFAR_MEAN).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]),
                  std=torch.tensor(np.array(CIFAR_STD).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]))

    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        robust_correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(loaders['train']):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.contiguous()  # Solve misaligned address

            with torch.cuda.amp.autocast(enabled=use_amp):
                # Loss = (f_r, f_r)
                robust_inputs_adv = pgd_train.attack(net=net.Rmoe, inp=inputs, label=targets)
                robust_inputs_adv.to(device)
                robust_outputs_adv = net.Rmoe(robust_inputs_adv)
                loss_rmoe = criterion(robust_outputs_adv, targets)

                for i in range(4):
                    expert = net.Rmoe.experts[i]
                    expert_inputs_adv = pgd_train.attack(net=expert, inp=inputs, label=targets)
                    expert_inputs_adv.to(device)
                    expert_outputs = expert(inputs)
                    expert_outputs_adv = expert(expert_inputs_adv)
                    loss_rmoe += beta * (1.0 / BATCH_SIZE) * criterion_kl(F.log_softmax(expert_outputs_adv, dim=1),
                                                F.softmax(expert_outputs, dim=1))

            for param in net.Smoe.parameters():
                param.requires_grad = False

            scaler.scale(loss_rmoe).backward(retain_graph=True)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            for param in net.Smoe.parameters():
                param.requires_grad = True

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = net(inputs)
                loss = criterion(outputs, targets)

            for param in net.Rmoe.parameters():
                param.requires_grad = False

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            for param in net.Rmoe.parameters():
                param.requires_grad = True

            scheduler.step()

            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = net(inputs)
                inputs_adv = pgd_train.attack(net=net, inp=inputs, label=targets)
                inputs_adv.to(device)
                outputs_adv = net(inputs_adv)
                loss = criterion(outputs_adv, targets)

            train_loss += loss.item()

            _, predicted = outputs.max(1)
            _, predicted_adv = outputs_adv.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            robust_correct += predicted_adv.eq(targets).sum().item()

            progress_bar(batch_idx, len(loaders['train']), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        return train_loss / (batch_idx + 1), 100. * correct / total, 100. * robust_correct / total

    def test(epoch):
        net.eval()
        test_loss = 0
        correct = 0
        robust_correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(loaders['test']):
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs.contiguous()  # Solve misaligned address
                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs = net(inputs)
                    with torch.enable_grad():
                        inputs_adv = pgd_val.attack(net=net, inp=inputs, label=targets)
                    inputs_adv.to(device)
                    outputs_adv = net(inputs_adv)
                    loss = criterion(outputs_adv, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                _, predicted_adv = outputs_adv.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                robust_correct += predicted_adv.eq(targets).sum().item()

                progress_bar(batch_idx, len(loaders['test']), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        # Save checkpoint.
        acc = 100. * correct / total
        ra_val = 100. * robust_correct / total
        print('Saving..')
        state = {"net": net.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "scaler": scaler.state_dict(),
                 "acc": acc,
                 "epoch": epoch}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/' + args.net + '-JTDMoE.t7')
        return test_loss, acc, ra_val

    if usewandb:
        wandb.watch(rmoe)

    net.cuda()

    for epoch in range(start_epoch, args.n_epochs):
        start = time.time()
        trainloss, acc_train, ra_train = train(epoch)
        val_loss, acc, ra_val = test(epoch)

        # Log training..
        if usewandb:
            wandb.log({'epoch': epoch, 'train_loss': trainloss,
                       "sa": acc_train, 'ra': ra_train,
                       'val_loss': val_loss,
                       'val_sa': acc, 'val_ra': ra_val,
                       "lr": optimizer1.param_groups[0]["lr"],
                       "epoch_time": time.time() - start})

    # writeout wandb
    if usewandb:
        wandb.save("wandb_{}.h5".format(args.net))


if __name__ == '__main__':
    main()