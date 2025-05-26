import torch
import numpy as np
import torch.nn as nn
import time
from abc import ABCMeta, abstractmethod, abstractproperty


class AttackBase(metaclass=ABCMeta):
    @abstractmethod
    def attack(self, net, inp, label, target=None):
        '''

        :param inp: batched images
        :param target: specify the indexes of target class, None represents untargeted attack
        :return: batched adversaril images
        '''
        pass

    @abstractmethod
    def to(self, device):
        pass


def clip_eta(eta, norm, eps, DEVICE=torch.device('cuda')):
    '''
    helper functions to project eta into epsilon norm ball
    :param eta: Perturbation tensor (should be of size(N, C, H, W))
    :param norm: which norm. should be in [1, 2, np.inf]
    :param eps: epsilon, bound of the perturbation
    :return: Projected perturbation
    '''

    assert norm in [1, 2, np.inf], "norm should be in [1, 2, np.inf]"

    with torch.no_grad():
        avoid_zero_div = torch.tensor(1e-12).to(DEVICE)
        eps = torch.tensor(eps).to(DEVICE)
        one = torch.tensor(1.0).to(DEVICE)

        if norm == np.inf:
            eta = torch.clamp(eta, -eps, eps)
        else:
            normalize = torch.norm(eta.reshape(eta.size(0), -1), p=norm, dim=-1, keepdim=False)
            normalize = torch.max(normalize, avoid_zero_div)

            normalize.unsqueeze_(dim=-1)
            normalize.unsqueeze_(dim=-1)
            normalize.unsqueeze_(dim=-1)

            factor = torch.min(one, eps / normalize)
            eta = eta * factor
    return eta


class PGD(AttackBase):
    # ImageNet pre-trained mean and std
    # _mean = torch.tensor(np.array([0.485, 0.456, 0.406]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis])
    # _std = torch.tensor(np.array([0.229, 0.224, 0.225]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis])

    # _mean = torch.tensor(np.array([0]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis])
    # _std = torch.tensor(np.array([1.0]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis])
    def __init__(self, eps=6 / 255.0, sigma=3 / 255.0, nb_iter=20,
                 norm=np.inf, DEVICE=torch.device('cuda'),
                 mean=torch.tensor(np.array([125.307, 122.961, 113.8575]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]), # Same as the transform in dataset
                 std=torch.tensor(np.array([51.5865, 50.847, 51.255]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]),
                 random_start=True):
        '''
        :param eps: maximum distortion of adversarial examples
        :param sigma: single step size
        :param nb_iter: number of attack iterations
        :param norm: which norm to bound the perturbations
        '''
        self.eps = eps
        self.sigma = sigma
        self.nb_iter = nb_iter
        self.norm = norm
        self.criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
        self.DEVICE = DEVICE
        self._mean = mean.to(DEVICE)
        self._std = std.to(DEVICE)
        self.random_start = random_start

    def single_attack(self, net, inp, label, eta, target=None):
        '''
        Given the original image and the perturbation computed so far, computes
        a new perturbation.
        :param net:
        :param inp: original image
        :param label:
        :param eta: perturbation computed so far
        :return: a new perturbation
        '''

        adv_inp = inp + eta

        # net.zero_grad()

        pred = net(adv_inp)

        if target is None:

            loss = self.criterion(pred, label)

            grad_sign = torch.autograd.grad(loss, adv_inp,
                                            only_inputs=True, retain_graph=False)[0].sign()

            adv_inp = adv_inp + grad_sign * (self.sigma / self._std) * 255
        else:
            loss = self.criterion(pred, target)
            print(loss)
            grad_sign = torch.autograd.grad(loss, adv_inp,
                                            only_inputs=True, retain_graph=False)[0].sign()
            adv_inp = adv_inp - grad_sign * (self.sigma / self._std) * 255

        tmp_adv_inp = adv_inp * self._std + self._mean

        tmp_inp = inp * self._std + self._mean
        tmp_adv_inp = torch.clamp(tmp_adv_inp, 0, 255)
        tmp_eta = tmp_adv_inp - tmp_inp
        tmp_eta = tmp_eta / 255
        tmp_eta = clip_eta(tmp_eta, norm=self.norm, eps=self.eps, DEVICE=self.DEVICE)
        tmp_eta = tmp_eta * 255

        eta = tmp_eta / self._std

        return eta

    def attack(self, net, inp, label, target=None):

        # ffcv [0,255]
        if self.random_start:
            eta = torch.FloatTensor(*inp.shape).uniform_(-self.eps, self.eps)
        else:
            eta = torch.zeros_like(inp)
        eta = eta*255
        eta = eta.to(self.DEVICE)
        eta = (eta - self._mean) / self._std
        net.eval()

        inp.requires_grad = True
        eta.requires_grad = True
        for i in range(self.nb_iter):
            eta = self.single_attack(net, inp, label, eta, target)

        # print(eta.max())
        adv_inp = inp + eta
        tmp_adv_inp = adv_inp * self._std + self._mean
        tmp_adv_inp = torch.clamp(tmp_adv_inp, 0, 255)
        adv_inp = (tmp_adv_inp - self._mean) / self._std

        return adv_inp

    def to(self, device):
        self.DEVICE = device
        self._mean = self._mean.to(device)
        self._std = self._std.to(device)
        self.criterion = self.criterion.to(device)

def pgd_normalization(inp):
    inp_max = []
    inp_min = []
    for i in range(inp.shape[1]):
        inp_max.append(inp[:,i,:,:].max())
        inp_min.append(inp[:,i,:,:].min())

    for i in range(inp.shape[1]):
        inp[:,i,:,:] = (inp[:,i,:,:] - inp_min[i])/(inp_max[i] - inp_min[i])

    return inp, inp_max, inp_min

def pgd_denormalization(inp, inp_max, inp_min):
    for i in range(inp.shape[1]):
        inp[:,i,:,:] = inp[:,i,:,:] * (inp_max[i] - inp_min[i]) + inp_min[i]

    return inp