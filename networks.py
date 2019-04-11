  # Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
  # 
  # Licensed under the CC BY-NC-SA 4.0 license 
  #     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
  #

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *

from torch.autograd import Variable

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def weights_init_horizontal1(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        filter = torch.FloatTensor(1, 1, 1, 2)
        filter[0, 0, 0, 0] = -1
        filter[0, 0, 0, 1] = 1
        m.weight.data = filter


def weights_init_horizontal2(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        filter = torch.FloatTensor(1, 1, 1, 2)
        filter[0, 0, 0, 0] = 1
        filter[0, 0, 0, 1] = -1
        m.weight.data = filter


def weights_init_vertical1(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        filter = torch.FloatTensor(1, 1, 2, 1)
        filter[0, 0, 0, 0] = -1
        filter[0, 0, 1, 0] = 1
        m.weight.data = filter


def weights_init_vertical2(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        filter = torch.FloatTensor(1, 1, 2, 1)
        filter[0, 0, 0, 0] = 1
        filter[0, 0, 1, 0] = -1
        m.weight.data = filter


def weights_init_gblur(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        gfilter = gaussian_filter(shape=(5, 5), sigma=1)
        gfilter = np.reshape(gfilter, [1, 1, 5, 5])
        m.weight.data = torch.from_numpy(gfilter).float()


class NetEdgeHorizontal1(nn.Module):
    def __init__(self):
        super(NetEdgeHorizontal1, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=(1, 2), stride=1, padding=0, bias=False)
        self.conv.apply(weights_init_horizontal1)
        self.conv.weight.requires_grad = False

        self.pad = nn.ZeroPad2d((1, 0, 0, 0))
        self.relu = nn.ReLU()

    def forward(self, mask):
        edge = self.conv(mask)
        edge = self.pad(edge)
        edge = self.relu(edge)
        return edge


class NetEdgeHorizontal2(nn.Module):
    def __init__(self):
        super(NetEdgeHorizontal2, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=(1, 2), stride=1, padding=0, bias=False)
        self.conv.apply(weights_init_horizontal2)
        self.conv.weight.requires_grad = False

        self.pad = nn.ZeroPad2d((0, 1, 0, 0))
        self.relu = nn.ReLU()

    def forward(self, mask):
        edge = self.conv(mask)
        edge = self.pad(edge)
        edge = self.relu(edge)
        return edge


class NetEdgeVertical1(nn.Module):
    def __init__(self):
        super(NetEdgeVertical1, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=(2, 1), stride=1, padding=0, bias=False)
        self.conv.apply(weights_init_vertical1)
        self.conv.weight.requires_grad = False

        self.pad = nn.ZeroPad2d((0, 0, 1, 0))
        self.relu = nn.ReLU()

    def forward(self, mask):
        edge = self.conv(mask)
        edge = self.pad(edge)
        edge = self.relu(edge)
        return edge


class NetEdgeVertical2(nn.Module):
    def __init__(self):
        super(NetEdgeVertical2, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=(2, 1), stride=1, padding=0, bias=False)
        self.conv.apply(weights_init_vertical2)
        self.conv.weight.requires_grad = False

        self.pad = nn.ZeroPad2d((0, 0, 0, 1))
        self.relu = nn.ReLU()

    def forward(self, mask):
        edge = self.conv(mask)
        edge = self.pad(edge)
        edge = self.relu(edge)
        return edge


class GaussianBlur(nn.Module):
    def __init__(self):
        super(GaussianBlur, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=(5, 5), stride=1, padding=2, bias=False)
        self.conv.apply(weights_init_gblur)
        self.conv.weight.requires_grad = False

    def forward(self, mask):
        blurred = self.conv(mask)

        return blurred


class STN(nn.Module):
    def __init__(self, opt):
        super(STN, self).__init__()
        self.opt = opt

        f_dim = opt.gf_dim
        self.onlyy = Variable(torch.from_numpy(np.array([0., 0., 0., 0., 1., 1.])).float()).cuda()
        self.offset = Variable(torch.from_numpy(np.array([1., 0., 0., 0., 0., 0.])).float()).cuda()

        self.fc = nn.Sequential(
            nn.Linear(opt.embed_dim_small * 1, f_dim * 2),
            nn.ReLU(),
            nn.Linear(f_dim * 2, f_dim * 1),
            nn.ReLU(),
        )

        self.fc_loc = nn.Linear(f_dim, 3 * 2)

        self.fc_loc.weight.data.fill_(0)
        self.fc_loc.bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

    def forward(self, newobj, input, rotFix, output_sizex, output_sizey):
        em = self.fc(input.view(self.opt.batch_size, -1))
        theta_lin = self.fc_loc(em)
        theta_lin = theta_lin * rotFix

        theta = theta_lin.view(-1, 2, 3)

        newobj_up = nn.Upsample(scale_factor=8)(newobj)
        grid = F.affine_grid(theta, torch.Size([self.opt.batch_size, 1, output_sizey, output_sizex]))
        box = F.grid_sample(newobj_up, grid)

        return box, grid, theta_lin


class STN_fixTheta(nn.Module):
    def __init__(self, opt):
        super(STN_fixTheta, self).__init__()
        self.batch_size = opt.batch_size
        self.onlyy = Variable(torch.from_numpy(np.array([0., 0., 0., 0., 1., 1.])).float()).cuda()
        self.offset = Variable(torch.from_numpy(np.array([1., 0., 0., 0., 0., 0.])).float()).cuda()

    def forward(self, gt, theta, output_sizex, output_sizey):
        thetay = theta * self.onlyy + self.offset
        thetay = thetay.view(-1, 2, 3)

        theta = theta.view(-1, 2, 3)

        gt_up = nn.Upsample(scale_factor=8)(gt)
        grid = F.affine_grid(theta, torch.Size([self.batch_size, 1, output_sizey, output_sizex]))
        box = F.grid_sample(gt_up, grid)

        gridy = F.affine_grid(thetay, torch.Size([self.batch_size, 1, output_sizey, output_sizex]))
        boxy = F.grid_sample(gt_up, gridy)

        return box, boxy

class Where_Encoder_Sup(nn.Module):
    def __init__(self, opt):
        super(Where_Encoder_Sup, self).__init__()

        f_dim = opt.gf_dim
        self.conv0 = nn.Sequential(
            nn.Conv2d(6, f_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(f_dim, f_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2),
        )
        self.fc_mu = nn.Sequential(
            nn.Conv2d(f_dim, opt.z_dim_spatial,
                      kernel_size=1, stride=1, padding=0),
        )
        self.fc_logvar = nn.Sequential(
            nn.Conv2d(f_dim, opt.z_dim_spatial,
                      kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        e0 = self.conv0(x.view([-1, 6, 1, 1]))
        mu = self.fc_mu(e0)
        logvar = self.fc_logvar(e0)

        return mu, logvar


class Where_Encoder(nn.Module):
    def __init__(self, opt):
        super(Where_Encoder, self).__init__()

        f_dim = opt.gf_dim
        self.conv0 = nn.Sequential(
            nn.Conv2d(opt.nClass + 1 + opt.z_dim_spatial, f_dim,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(f_dim, f_dim * 2,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 2, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(f_dim * 2, f_dim * 4,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 4, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(f_dim * 4, f_dim * 8,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 8, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(f_dim * 8, f_dim * 8,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 8, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(f_dim * 8, opt.embed_dim_small,
                      kernel_size=(4, 8), stride=1, padding=0),
        )
        self.reconz = nn.Sequential(
            nn.Dropout2d(p=0.4),
            nn.Conv2d(opt.embed_dim_small, opt.z_dim_spatial,
                      kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        e0 = self.conv0(x)
        e1 = self.conv1(e0)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        y = self.conv7(e4)
        z = self.reconz(y)

        return y, z


class What_Encoder(nn.Module):
    def __init__(self, opt):
        super(What_Encoder, self).__init__()

        f_dim = opt.gf_dim
        self.conv0 = nn.Sequential(
            nn.Conv2d(opt.nClass + 1 + opt.z_dim_appr, f_dim,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(f_dim, f_dim * 2,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 2, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(f_dim * 2, f_dim * 4,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 4, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(f_dim * 4, f_dim * 8,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 8, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(f_dim * 8, f_dim * 8,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 8, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(f_dim * 8, f_dim * 8,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 8, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(f_dim * 8, f_dim * 8,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 8, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(f_dim * 8, opt.embed_dim_big,
                      kernel_size=(4, 8), stride=1, padding=0),
        )
        self.reconz = nn.Sequential(
            nn.Dropout2d(p=0.4),
            nn.Conv2d(opt.embed_dim_big, opt.z_dim_appr,
                      kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        e0 = self.conv0(x)
        e1 = self.conv1(e0)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        e5 = self.conv5(e4)
        e6 = self.conv6(e5)
        y = self.conv7(e6)
        z = self.reconz(y)

        return y, z

class What_Encoder_Sup(nn.Module):
    def __init__(self, opt):
        super(What_Encoder_Sup, self).__init__()

        f_dim = opt.gf_dim
        self.conv0 = nn.Sequential(
            nn.Conv2d(1, f_dim,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(f_dim, f_dim * 2,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 2, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(f_dim * 2, f_dim * 4,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 4, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(f_dim * 4, f_dim * 8,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 8, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(f_dim * 8, f_dim * 8,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 8, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.fc_mu = nn.Sequential(
            nn.Conv2d(f_dim * 8, opt.z_dim_appr,
                      kernel_size=4, stride=1, padding=0),
        )
        self.fc_logvar = nn.Sequential(
            nn.Conv2d(f_dim * 8, opt.z_dim_appr,
                      kernel_size=4, stride=1, padding=0),
        )

    def forward(self, x):
        e0 = self.conv0(x)
        e1 = self.conv1(e0)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        mu = self.fc_mu(e4)
        logvar = self.fc_logvar(e4)

        return mu, logvar


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()

        f_dim = opt.gf_dim
        self.convT0 = nn.Sequential(
            nn.ConvTranspose2d(opt.embed_dim_big, f_dim * 8,
                               kernel_size=4, stride=1, padding=0),
            nn.InstanceNorm2d(f_dim * 8, affine=False),
            nn.ReLU(),
        )
        self.convT1 = nn.Sequential(
            nn.ConvTranspose2d(f_dim * 8, f_dim * 8,
                               kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 8, affine=False),
            nn.ReLU(),
        )
        self.convT2 = nn.Sequential(
            nn.ConvTranspose2d(f_dim * 8, f_dim * 4,
                               kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 4, affine=False),
            nn.ReLU(),
        )
        self.convT3 = nn.Sequential(
            nn.ConvTranspose2d(f_dim * 4, f_dim * 2,
                               kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 2, affine=False),
            nn.ReLU(),
        )
        self.convT4 = nn.Sequential(
            nn.ConvTranspose2d(f_dim * 2, f_dim * 1,
                               kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 1, affine=False),
            nn.ReLU(),
        )
        self.convT5 = nn.Sequential(
            nn.ConvTranspose2d(f_dim * 1, 1,
                               kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, em):
        m0 = self.convT0(em)
        m1 = self.convT1(m0)
        m2 = self.convT2(m1)
        m3 = self.convT3(m2)
        m4 = self.convT4(m3)
        mask = self.convT5(m4)
        # mask = nn.Hardtanh()(mask)
        # mask = (mask + 1.) / 2.

        return mask


class Where_Reconstructor(nn.Module):
    def __init__(self, opt):
        super(Where_Reconstructor, self).__init__()

        f_dim = opt.gf_dim
        self.convT0 = nn.Sequential(
            nn.ConvTranspose2d(opt.embed_dim_small, f_dim * 8,
                               kernel_size=(4, 8), stride=1, padding=0),
            nn.InstanceNorm2d(f_dim * 8, affine=False),
            nn.ReLU(),
        )
        self.convT1 = nn.Sequential(
            nn.ConvTranspose2d(f_dim * 8, f_dim * 8,
                               kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 8, affine=False),
            nn.ReLU(),
        )
        self.convT2 = nn.Sequential(
            nn.ConvTranspose2d(f_dim * 8, f_dim * 4,
                               kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 4, affine=False),
            nn.ReLU(),
        )
        self.convT3 = nn.Sequential(
            nn.ConvTranspose2d(f_dim * 4, f_dim * 2,
                               kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 2, affine=False),
            nn.ReLU(),
        )
        self.convT4 = nn.Sequential(
            nn.ConvTranspose2d(f_dim * 2, f_dim * 1,
                               kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 1, affine=False),
            nn.ReLU(),
        )
        self.convT7 = nn.Sequential(
            nn.ConvTranspose2d(f_dim * 1, opt.nClass + 1,
                               kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, em):
        m0 = self.convT0(em)
        m1 = self.convT1(m0)
        m2 = self.convT2(m1)
        m3 = self.convT3(m2)
        m4 = self.convT4(m3)
        mask = self.convT7(m4)

        return mask

class What_Reconstructor(nn.Module):
    def __init__(self, opt):
        super(What_Reconstructor, self).__init__()

        f_dim = opt.gf_dim
        self.convT0 = nn.Sequential(
            nn.ConvTranspose2d(opt.embed_dim_big, f_dim * 8,
                               kernel_size=(4, 8), stride=1, padding=0),
            nn.InstanceNorm2d(f_dim * 8, affine=False),
            nn.ReLU(),
        )
        self.convT1 = nn.Sequential(
            nn.ConvTranspose2d(f_dim * 8, f_dim * 8,
                               kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 8, affine=False),
            nn.ReLU(),
        )
        self.convT2 = nn.Sequential(
            nn.ConvTranspose2d(f_dim * 8, f_dim * 8,
                               kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 8, affine=False),
            nn.ReLU(),
        )
        self.convT3 = nn.Sequential(
            nn.ConvTranspose2d(f_dim * 8, f_dim * 8,
                               kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 8, affine=False),
            nn.ReLU(),
        )
        self.convT4 = nn.Sequential(
            nn.ConvTranspose2d(f_dim * 8, f_dim * 4,
                               kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 4, affine=False),
            nn.ReLU(),
        )
        self.convT5 = nn.Sequential(
            nn.ConvTranspose2d(f_dim * 4, f_dim * 2,
                               kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 2, affine=False),
            nn.ReLU(),
        )
        self.convT6 = nn.Sequential(
            nn.ConvTranspose2d(f_dim * 2, f_dim * 1,
                               kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 1, affine=False),
            nn.ReLU(),
        )
        self.convT7 = nn.Sequential(
            nn.ConvTranspose2d(f_dim * 1, opt.nClass + 1 ,
                               kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, em):
        m0 = self.convT0(em)
        m1 = self.convT1(m0)
        m2 = self.convT2(m1)
        m3 = self.convT3(m2)
        m4 = self.convT4(m3)
        m5 = self.convT5(m4)
        m6 = self.convT6(m5)
        mask = self.convT7(m6)

        return mask

class DiscriminatorSTN(nn.Module):
    def __init__(self, opt):
        super(DiscriminatorSTN, self).__init__()

        f_dim = opt.gf_dim

        self.conv0 = nn.Sequential(
            nn.Conv2d(6, f_dim, kernel_size=1),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(f_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        e0 = self.conv0(x.view([-1, 6, 1, 1]))
        disc = self.conv1(e0)

        return disc


class DiscriminatorWhere(nn.Module):
    def __init__(self, opt):
        super(DiscriminatorWhere, self).__init__()

        f_dim = opt.df_dim

        self.conv0 = nn.Sequential(
            nn.Conv2d((opt.nClass + 1 + 1), f_dim * 1,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(f_dim * 1, f_dim * 2,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 2, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(f_dim * 2, f_dim * 4,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 4, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(f_dim * 4, 1,
                      kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, seg, edge, mask):
        d0 = self.conv0(torch.cat([seg, edge, mask], 1))
        d1 = self.conv1(d0)
        d2 = self.conv2(d1)
        disc = self.conv3(d2)
        return disc


class DiscriminatorObj(nn.Module):
    def __init__(self, opt):
        super(DiscriminatorObj, self).__init__()

        f_dim = opt.df_dim

        self.conv0 = nn.Sequential(
            nn.Conv2d(1, f_dim * 1,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(f_dim * 1, f_dim * 2,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 2, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(f_dim * 2, f_dim * 4,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 4, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(f_dim * 4, f_dim * 8,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 8, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(f_dim * 8, f_dim * 8,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 8, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(f_dim * 8, 1,
                      kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        d0 = self.conv0(x)
        d1 = self.conv1(d0)
        d2 = self.conv2(d1)
        d3 = self.conv3(d2)
        d4 = self.conv4(d3)
        disc = self.conv5(d4)
        return disc


class DiscriminatorWhat(nn.Module):
    def __init__(self, opt):
        super(DiscriminatorWhat, self).__init__()

        f_dim = opt.df_dim

        self.conv0 = nn.Sequential(
            nn.Conv2d((opt.nClass + 1), f_dim * 1,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(f_dim * 1, f_dim * 2,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(f_dim * 2, affine=False),
            nn.LeakyReLU(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(f_dim * 2, 1,
                      kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, seg, edge):
        d0 = self.conv0(torch.cat([seg, edge], 1))
        d1 = self.conv1(d0)
        disc = self.conv2(d1)
        return disc


