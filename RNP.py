"""
Author: Bill Wang
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


MAX_GRAD_NORM = 0.1  # clip gradient
LR_GAMMA = 0.1
LR_DECAY_EPOCHS = 5000
xavier_init = False


class RTargetNet(nn.Module):
    def __init__(self, in_c, out_c):
        super(RTargetNet, self).__init__()
        hid_c = int(in_c / 2)
        # architecture def
        self.layers = nn.Sequential(
            nn.Conv3d(in_c, hid_c, kernel_size=1, bias=False, padding=0),
            nn.LeakyReLU(inplace=False, negative_slope=2.5e-1),
            # nn.ReLU(inplace=False),
            nn.Conv3d(hid_c, hid_c, kernel_size=1, bias=False, padding=0),
            nn.LeakyReLU(inplace=False, negative_slope=2.5e-1),
            # nn.ReLU(inplace=False),
            nn.Conv3d(hid_c, out_c, kernel_size=1, bias=False, padding=0),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if not xavier_init:
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    nn.init.constant_(m.bias, 0.0)
                else:
                    torch.nn.init.xavier_uniform_(m.weight)
                    try:
                        torch.nn.init.xavier_uniform_(m.bias)
                    except BaseException:
                        print("Bias of a certain Conv layer cannot be inited")
            elif isinstance(m, nn.Linear):
                if not xavier_init:
                    nn.init.kaiming_normal_(m.weight)
                    nn.init.constant_(m.bias, 0.0)
                else:
                    stdv = 1. / math.sqrt(m.weight.size(1))
                    m.weight.data.uniform_(-stdv, stdv)
                    if m.bias is not None:
                        m.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = self.layers(x)
        return x


class RNet(nn.Module):
    def __init__(self, in_c, out_c, dropout_r):
        super(RNet, self).__init__()
        hid_c = int(in_c/2)

        # architecture def
        self.layers = nn.Sequential(
            nn.Conv3d(in_c, hid_c, kernel_size=1, bias=False, padding=0),
            nn.LeakyReLU(inplace=False, negative_slope=2.5e-1),
            # nn.ReLU(inplace=False),
            nn.Conv3d(hid_c, out_c, kernel_size=1, bias=False, padding=0),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if not xavier_init:
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    nn.init.constant_(m.bias, 0.0)
                else:
                    torch.nn.init.xavier_uniform_(m.weight)
                    try:
                        torch.nn.init.xavier_uniform_(m.bias)
                    except BaseException:
                        print("Bias of a certain Conv layer cannot be inited")
            elif isinstance(m, nn.Linear):
                if not xavier_init:
                    nn.init.kaiming_normal_(m.weight)
                    nn.init.constant_(m.bias, 0.0)
                else:
                    stdv = 1. / math.sqrt(m.weight.size(1))
                    m.weight.data.uniform_(-stdv, stdv)
                    if m.bias is not None:
                        m.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = self.layers(x)
        return x


class RNP:
    def __init__(self, in_c, out_c, logfile=None, USE_GPU=False, LR=1e-2, dropout_r=0.1):
        self.r_target_net = RTargetNet(in_c, out_c)
        self.r_net = RNet(in_c, out_c, dropout_r)
        self.USE_GPU = USE_GPU
        self.LR = LR
        self.logfile = logfile

        print(self.r_target_net)
        if self.logfile:
            self.logfile.write(str(self.r_target_net))
        print(self.r_net)
        if self.logfile:
            self.logfile.write(str(self.r_net))

        if USE_GPU:
            self.r_target_net = self.r_target_net.cuda()
            self.r_net = self.r_net.cuda()

        # define optimizer for predict network
        # self.r_net_optim = torch.optim.Adam(self.r_net.parameters(), lr=LR)
        self.r_net_optim = torch.optim.SGD(self.r_net.parameters(), lr=LR, momentum=0.9)

        self.epoch = 0
        # torch.autograd.set_detect_anomaly(True)

    def train_model(self, x, epoch=None):
        # print('In train_model()...')
        self.r_net.train()
        x = torch.FloatTensor(x)

        if self.USE_GPU:
            x = x.cuda()

        if epoch is not None and epoch % LR_DECAY_EPOCHS == 0 and self.epoch != epoch:
            self.adjust_learning_rate()
            self.epoch = epoch

        r_target = self.r_target_net(x).detach()
        r_pred = self.r_net(x)
        loss = torch.mean(F.mse_loss(r_pred, r_target, reduction='none'), dim=1).mean()
        # loss = torch.mean(F.l1_loss(r_pred, r_target, reduction='none'), dim=1).mean()

        self.r_net_optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.r_net.parameters(), MAX_GRAD_NORM)
        self.r_net_optim.step()
        return loss.data.cpu().numpy()

    def eval_model(self, x):
        # print('In eval_model()...')
        self.r_net.eval()
        x = torch.FloatTensor(x)

        if self.USE_GPU:
            x = x.cuda()
        r_target = self.r_target_net(x).detach()
        r_pred = self.r_net(x)
        r_gap = F.mse_loss(r_pred, r_target, reduction='none')
        # r_gap = F.l1_loss(r_pred, r_target, reduction='none')
        return r_gap.detach()

    def adjust_learning_rate(self):
        self.LR *= LR_GAMMA
        print(' * adjust C_LR == {}'.format(self.LR))
        if self.logfile:
            self.logfile.write(' * adjust C_LR == {}\n'.format(self.LR))

        for param_group in self.r_net_optim.param_groups:
            param_group['lr'] = self.LR

    def save_model(self, path):
        dict_to_save = {
            'r_net': self.r_net.state_dict(),
            'r_target_net': self.r_target_net.state_dict(),
            # 'r_net_optim': self.r_net_optim,
            # 'LR': self.LR,
        }
        torch.save(dict_to_save, path)

    def load_model(self, path):
        states = torch.load(path)
        self.r_net.load_state_dict(states['r_net'])
        self.r_target_net.load_state_dict(states['r_target_net'])
        if 'r_net_optim' in states:
            self.r_net_optim = states['r_net_optim']
        if 'LR' in states:
            self.LR = states['LR']
