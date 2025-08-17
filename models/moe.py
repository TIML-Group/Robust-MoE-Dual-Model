import torch
import torch.nn as nn
import timm
from models.vit_small import ViT
from models import *


class Router(nn.Module):
    def __init__(self, num_experts, size):
        super(Router, self).__init__()
        self.gate = nn.Linear(3*size*size, num_experts)
    def forward(self, x):
        #Top-1 strategy
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        gate_logits = self.gate(x_flat)
        top1_expert = torch.argmax(gate_logits, dim=1)
        return top1_expert
    def get_second_expert(self, x):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        gate_logits = self.gate(x_flat)
        _, top2_indices = torch.topk(gate_logits, k=2, dim=1)
        second_expert = top2_indices[:, 1]
        return second_expert


class ResnetExpert(nn.Module):
    def __init__(self):
        super(ResnetExpert, self).__init__()
        self.net = ResNet18()

    def forward(self, x):
        return self.net(x)


class ViTExpert(nn.Module):
    def __init__(self, size):
        super(ViTExpert, self).__init__()
        self.vit = timm.create_model('vit_small_patch16_224', pretrained=True, img_size=(size, size))
        self.vit.head = nn.Linear(self.vit.head.in_features, 200)

    def forward(self, x):
        return self.vit(x)


class MOE_Resnet18(nn.Module):
    def __init__(self, num_experts, num_classes, size):
        super(MOE_Resnet18, self).__init__()
        self.experts = nn.ModuleList([ResnetExpert() for _ in range(num_experts)])
        self.router = Router(num_experts, size)

    def forward(self, x):
        # Flatten the input to (batch_size, -1) for the router
        batch_size = x.size(0)

        #Top-1 Strategy
        top1_expert_indices = self.router(x)

        # Forward the input through the selected expert
        outputs = torch.zeros(batch_size, 10).to(x.device)  # 10: dataset class
        for i in range(batch_size):
            expert_output = self.experts[top1_expert_indices[i]](x[i].unsqueeze(0)) # It should be done in batch-size
            outputs[i] = expert_output

        return outputs


class MOE_ViT(nn.Module):
    def __init__(self, num_experts, size):
        super(MOE_ViT, self).__init__()
        self.experts = nn.ModuleList([ViTExpert(size=size) for _ in range(num_experts)])
        self.router = Router(num_experts, size)

    def forward(self, x):
        # Flatten the input to (batch_size, -1) for the router
        batch_size = x.size(0)

        #Top-1 Strategy
        top1_expert_indices = self.router(x)

        # Forward the input through the selected expert
        outputs = torch.zeros(batch_size, 200).to(x.device)  # 200: dataset class
        for i in range(batch_size):
            expert_output = self.experts[top1_expert_indices[i]](x[i].unsqueeze(0)) # It should be done in batch-size
            outputs[i] = expert_output
        return outputs


class Dual_Model_resnet18(nn.Module):
    def __init__(self, alpha, size):
        super(Dual_Model_resnet18, self).__init__()
        self.alpha = alpha
        self.Smoe = MOE_Resnet18(num_experts=4, num_classes=10, size=size)
        self.Rmoe = MOE_Resnet18(num_experts=4, num_classes=10, size=size)

    def forward(self, x):
        outputs = (1- self.alpha) * self.Smoe(x) + self.alpha * self.Rmoe(x)
        return outputs


class Dual_Model_vit(nn.Module):
    def __init__(self, alpha, size):
        super(Dual_Model_vit, self).__init__()
        self.alpha = alpha
        self.Smoe = MOE_ViT(num_experts=4, num_classes=10, size=size)
        self.Rmoe = MOE_ViT(num_experts=4, num_classes=10, size=size)

    def forward(self, x):
        outputs = (1- self.alpha) * self.Smoe(x) + self.alpha * self.Rmoe(x)
        return outputs