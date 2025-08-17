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
from ffcv.transforms import RandomHorizontalFlip, Cutout, RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage, ImageMixup, LabelMixup, RandomResizedCrop
from ffcv.fields.decoders import RandomResizedCropRGBImageDecoder
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
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
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
    parser.add_argument('--size', default="224")
    parser.add_argument('--n_epochs', type=int, default='200')
    parser.add_argument('--patch', default='4', type=int, help="patch for ViT")
    parser.add_argument('--dimhead', default="512", type=int)
    parser.add_argument('--convkernel', default='8', type=int, help="parameter for convmixer")
    parser.add_argument('--beta', default='6', type=int, help="parameter for the proportion of experts specification")

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
    size = 224

    # Prepare dataset
    imagenet_path = './data/tiny_imagenet/tiny-imagenet-200'
    train_dataset = torchvision.datasets.ImageFolder(root=f'{imagenet_path}/train')
    val_dataset = torchvision.datasets.ImageFolder(root=f'{imagenet_path}/val')

    train_beton_path = './data/tiny_imagenet/imagenet_train.beton'
    val_beton_path = './data/tiny_imagenet/imagenet_test.beton'

    train_writer = DatasetWriter(train_beton_path, {
        'image': RGBImageField(max_resolution=256, jpeg_quality=90),
        'label': IntField()
    })
    val_writer = DatasetWriter(val_beton_path, {
        'image': RGBImageField(max_resolution=256, jpeg_quality=90),
        'label': IntField()
    })

    train_writer.from_indexed_dataset(train_dataset)
    val_writer.from_indexed_dataset(val_dataset)

    # Note that statistics are wrt to uin8 range, [0,255].
    CIFAR_MEAN = [0.4802 * 255, 0.4481 * 255, 0.3975 * 255]
    CIFAR_STD = [0.2770 * 255, 0.2691 * 255,  0.2821 * 255]

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

            # RandomResizedCrop(scale=(1.0, 1.0), ratio=(1.0, 1.0), size=224),
            ToTensor(),
            ToDevice(a, non_blocking=True),
            ToTorchImage(),
            Convert(ch.float16),
            torchvision.transforms.Resize(size),
            torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

        # Create loaders
        loaders[name] = Loader(f'./data/tiny_imagenet/imagenet_{name}.beton',
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

        # net = timm.create_model("vit_base_patch16_384", pretrained=True)
        net = timm.create_model('vit_small_patch16_224', pretrained=True)
        net.head = nn.Linear(net.head.in_features, 200)
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
    elif args.net == "vit_moe":
        net = MOE_ViT(num_experts=4, size=size)

    # For Multi-GPU
    if 'cuda' in device:
        print(device)
        if args.dp:
            print("using data parallel")
            net = torch.nn.DataParallel(net)  # make parallel
            cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/' + args.net + '-RT_ER-{}.t7'.format(args.beta))
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
    epsilon = 2 / 255
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
                outputs = net(inputs)
                inputs_adv = pgd_train.attack(net=net, inp=inputs, label=targets)
                inputs_adv.to(device)
                outputs_adv = net(inputs_adv)
                loss_model = criterion(outputs_adv, targets)

                second_best_expert_indices = net.router.get_second_expert(inputs)
                loss_expert = 0.0
                for expert_id, expert in enumerate(net.experts):
                    selected_mask = (second_best_expert_indices == expert_id)

                    if selected_mask.sum() == 0:
                        continue
                    selected_inputs = inputs[selected_mask]
                    selected_targets = targets[selected_mask]
                    selected_inputs, selected_targets = selected_inputs.detach(), selected_targets.detach()

                    expert_inputs_adv = pgd_train.attack(net=expert, inp=selected_inputs, label=selected_targets)
                    expert_inputs_adv = expert_inputs_adv.to(device)

                    expert_outputs = expert(selected_inputs)
                    expert_outputs_adv = expert(expert_inputs_adv)

                    loss_expert += criterion_kl(
                        F.log_softmax(expert_outputs_adv, dim=1),
                        F.softmax(expert_outputs, dim=1)
                    )
                loss = loss_model + beta * (1.0 / BATCH_SIZE) * loss_expert
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            scheduler.step()

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
        torch.save(state, './checkpoint/' + args.net + '-RT_ER-{}.t7'.format(args.beta))
        return test_loss, acc, ra_val

    net.cuda()
    for epoch in range(start_epoch, args.n_epochs):
        start = time.time()
        trainloss, acc_train, ra_train = train(epoch)
        if (epoch + 1) % 10 == 0:
            val_loss, acc, ra_val = test(epoch)
            if usewandb:
                wandb.log({'val_loss': val_loss,
                           'val_sa': acc, 'val_ra': ra_val})

        # Log training..
        if usewandb:
            wandb.log({'epoch': epoch, 'train_loss': trainloss,
                       "sa": acc_train, 'ra': ra_train,
                       "lr": optimizer.param_groups[0]["lr"],
                       "epoch_time": time.time() - start})

    # writeout wandb
    if usewandb:
        wandb.save("wandb_{}.h5".format(args.net))


if __name__ == '__main__':
    main()