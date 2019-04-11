  # Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
  # 
  # Licensed under the CC BY-NC-SA 4.0 license 
  #     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
  #

import itertools
import os

from networks import *
from utils import *

class InstanceAdder():
    def __init__(self, opt):
        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.c_dim = self.opt.c_dim
        self.nClass = self.opt.nClass
        self.output_size = self.opt.output_size
        self.z_dim_appr = self.opt.z_dim_appr
        self.z_dim_spatial = self.opt.z_dim_spatial
        self.target_channel = opt.target_channel

        if self.opt.use_gpu:
            self.Tensor = torch.cuda.FloatTensor
        else:
            self.Tensor = torch.Tensor

        self.bb_white = Variable(self.Tensor(self.batch_size, 1, self.opt.compact_sizey, self.opt.compact_sizex).fill_(1.))
        self.max_width_ratio = Variable(self.Tensor(1).fill_(opt.max_width_ratio))
        self.max_height_ratio = Variable(self.Tensor(1).fill_(opt.max_height_ratio))
        self.rotFix = Variable(torch.from_numpy(np.array([1., 0., 1., 0., 1., 1.])).float()).cuda()

        self.pad_small = Variable(torch.zeros(self.opt.batch_size, self.nClass - 1, self.opt.image_sizey_small, self.opt.image_sizex_small)).cuda()
        self.pad_before_small = Variable(
            torch.zeros(self.opt.batch_size, self.target_channel, self.opt.image_sizey_small, self.opt.image_sizex_small)).cuda()
        self.pad_after_small = Variable(
            torch.zeros(self.opt.batch_size, self.nClass - self.target_channel - 1, self.opt.image_sizey_small, self.opt.image_sizex_small)).cuda()

        self.pad_big = Variable(torch.zeros(self.opt.batch_size, self.nClass - 1, self.opt.image_sizey_big, self.opt.image_sizex_big)).cuda()
        self.pad_before_big = Variable(torch.zeros(self.opt.batch_size, self.target_channel, self.opt.image_sizey_big, self.opt.image_sizex_big)).cuda()
        self.pad_after_big = Variable(
            torch.zeros(self.opt.batch_size, self.nClass - self.target_channel - 1, self.opt.image_sizey_big,
                        self.opt.image_sizex_big)).cuda()

        transform_list = [transforms.ToTensor()]

        self.transform = transforms.Compose(transform_list)

    def define_networks(self):
        # where
        self.where_encoder_sup = Where_Encoder_Sup(self.opt)
        self.where_encoder = Where_Encoder(self.opt)
        self.where_reconstructor = Where_Reconstructor(self.opt)
        self.stn = STN(self.opt)

        self.disSTN = DiscriminatorSTN(self.opt)
        self.disWhere = DiscriminatorWhere(self.opt)

        # what
        self.what_encoder_sup = What_Encoder_Sup(self.opt)
        self.what_encoder = What_Encoder(self.opt)
        self.generator = Generator(self.opt)
        self.what_reconstructor = What_Reconstructor(self.opt)

        self.disObj = DiscriminatorObj(self.opt)
        self.disWhat = DiscriminatorWhat(self.opt)

        self.stn_fix = STN_fixTheta(self.opt)

        if self.opt.cont_train:
            self.load(self.opt.start_epoch)
        else:
            # initialize where
            self.where_encoder_sup.apply(weights_init)
            self.where_encoder.apply(weights_init)
            self.where_reconstructor.apply(weights_init)

            self.disSTN.apply(weights_init)
            self.disWhere.apply(weights_init)

            # initialize what
            self.what_encoder_sup.apply(weights_init)
            self.what_encoder.apply(weights_init)
            self.generator.apply(weights_init)
            self.what_reconstructor.apply(weights_init)

            self.disObj.apply(weights_init)
            self.disWhat.apply(weights_init)

        self.conv_edge_horizontal1 = NetEdgeHorizontal1()
        self.conv_edge_horizontal2 = NetEdgeHorizontal2()
        self.conv_edge_vertical1 = NetEdgeVertical1()
        self.conv_edge_vertical2 = NetEdgeVertical2()

        if self.opt.use_gpu:
            torch.cuda.set_device(self.opt.gpu_id)
            self.where_encoder_sup.cuda()
            self.where_encoder.cuda()
            self.where_reconstructor.cuda()
            self.stn.cuda()

            self.disSTN.cuda()
            self.disWhere.cuda()

            self.what_encoder_sup.cuda()
            self.what_encoder.cuda()
            self.generator.cuda()
            self.what_reconstructor.cuda()

            self.disObj.cuda()
            self.disWhat.cuda()

            self.conv_edge_vertical1.cuda()
            self.conv_edge_vertical2.cuda()
            self.conv_edge_horizontal1.cuda()
            self.conv_edge_horizontal2.cuda()
            self.stn_fix.cuda()

    def define_optimizers(self):
        self.optimizer_where_encoder_sup = torch.optim.Adam(self.where_encoder_sup.parameters(),
                                                  lr=self.opt.learning_rate,
                                                  betas=(self.opt.beta1, 0.999))

        self.optimizer_what_encoder_sup = torch.optim.Adam(self.what_encoder_sup.parameters(),
                                                           lr=self.opt.learning_rate,
                                                           betas=(self.opt.beta1, 0.999))

        self.optimizer_G = torch.optim.Adam(itertools.chain(self.where_encoder.parameters(),
                                                            self.where_reconstructor.parameters(),
                                                            self.stn.parameters(),
                                                            self.what_encoder.parameters(),
                                                            self.generator.parameters(),
                                                            self.what_reconstructor.parameters()),
                                            lr=self.opt.learning_rate,
                                            betas=(self.opt.beta1, 0.999))

        self.optimizer_D_stn = torch.optim.Adam(self.disSTN.parameters(),
                                                  lr=self.opt.learning_rate,
                                                  betas=(self.opt.beta1, 0.999))

        self.optimizer_D_where = torch.optim.Adam(self.disWhere.parameters(),
                                                lr=self.opt.learning_rate,
                                                betas=(self.opt.beta1, 0.999))

        self.optimizer_D_obj = torch.optim.Adam(self.disObj.parameters(),
                                                lr=self.opt.learning_rate,
                                                betas=(self.opt.beta1, 0.999))

        self.optimizer_D_what = torch.optim.Adam(self.disWhat.parameters(),
                                                lr=self.opt.learning_rate,
                                                betas=(self.opt.beta1, 0.999))

        self.criterionGAN = nn.BCELoss()
        if self.opt.use_gpu:
            self.criterionGAN = self.criterionGAN.cuda()

    def set_z(self, z_appr, z_spatial):
        self.input_z_appr = Variable(z_appr)
        self.input_z_spatial = Variable(z_spatial)
        if self.opt.use_gpu:
            self.input_z_appr = self.input_z_appr.cuda()
            self.input_z_spatial = self.input_z_spatial.cuda()

        self.input_z_spatial_expand = self.input_z_spatial.expand([-1, -1, self.opt.image_sizey_small, self.opt.image_sizex_small])
        self.input_z_appr_expand = self.input_z_appr.expand([-1, -1, self.opt.image_sizey_big, self.opt.image_sizex_big])

    def set_real_input(self, real_seg_small, real_ins_small, real_seg_big, real_ins_big, real_ins_compact, real_ins_theta):
        self.real_seg_small = Variable(self.Tensor(self.batch_size, self.opt.nClass, self.opt.image_sizey_small, self.opt.image_sizex_small))
        self.real_ins_small = Variable(self.Tensor(self.batch_size, 1, self.opt.image_sizey_small, self.opt.image_sizex_small))
        self.real_seg_big = Variable(self.Tensor(self.batch_size, self.opt.nClass, self.opt.image_sizey_big, self.opt.image_sizex_big))
        self.real_ins_big = Variable(self.Tensor(self.batch_size, 1, self.opt.image_sizey_big, self.opt.image_sizex_big))
        self.real_ins_compact = Variable(self.Tensor(self.batch_size, 1, self.opt.compact_sizey, self.opt.compact_sizex))
        self.real_ins_theta = Variable(self.Tensor(self.batch_size, 6))

        # stack tensors
        for i in range(len(real_ins_compact)):
            self.real_seg_small[i, :, :, :] = self.transform(real_seg_small[i])
            self.real_ins_small[i, :, :, :] = self.transform(real_ins_small[i])
            self.real_seg_big[i, :, :, :] = self.transform(real_seg_big[i])
            self.real_ins_big[i, :, :, :] = self.transform(real_ins_big[i])
            self.real_ins_compact[i, :, :, :] = self.transform(real_ins_compact[i])
            self.real_ins_theta[i] = torch.from_numpy(real_ins_theta[i]).float()

        if self.opt.use_gpu:
            self.real_seg_small = self.real_seg_small.cuda()
            self.real_ins_small = self.real_ins_small.cuda()
            self.real_seg_big = self.real_seg_big.cuda()
            self.real_ins_big = self.real_ins_big.cuda()
            self.real_ins_compact = self.real_ins_compact.cuda()
            self.real_ins_theta = self.real_ins_theta.cuda()

        self.real_bb_transformed_small, self.real_bbsy_small = \
            self.stn_fix(self.bb_white, self.real_ins_theta, self.opt.image_sizex_small, self.opt.image_sizey_small)
        self.real_bb_transformed_big, _ = \
            self.stn_fix(self.bb_white, self.real_ins_theta, self.opt.image_sizex_big, self.opt.image_sizey_big)
        self.real_ins_transformed_big, _ = \
            self.stn_fix(self.real_ins_compact, self.real_ins_theta, self.opt.image_sizex_big, self.opt.image_sizey_big)

        self.real_bbs_pad_small = self.pad_to_nClass_small(self.real_bb_transformed_small)
        self.bb_on_real_seg_small_masked = self.real_seg_small * (1. - self.real_bb_transformed_small)
        self.bb_on_real_seg_small = self.bb_on_real_seg_small_masked + self.real_bbs_pad_small

        self.real_bbs_pad_big = self.pad_to_nClass_big(self.real_bb_transformed_big)
        self.bb_on_real_seg_big_masked = self.real_seg_big * (1. - self.real_bb_transformed_big)
        self.bb_on_real_seg_big = self.bb_on_real_seg_big_masked + self.real_bbs_pad_big

        self.real_edge_small = self.compute_edge_for_input(self.real_ins_small)
        self.bb_on_real_edge_small = self.real_edge_small * (1. - self.real_bb_transformed_small)

        self.real_edge_big = self.compute_edge_for_input(self.real_ins_big)
        self.bb_on_real_edge_big = self.real_edge_big * (1. - self.real_bb_transformed_big)

        self.real_ins_pad_big = self.pad_to_nClass_big(self.real_ins_transformed_big)
        self.ins_on_real_seg_big_masked = self.real_seg_big * (1. - self.real_ins_transformed_big)
        self.ins_on_real_seg_big = self.ins_on_real_seg_big_masked + self.real_ins_pad_big

        self.ins_on_real_edge_big = self.real_edge_big * (1. - self.real_ins_transformed_big)

        self.real_ins_compact_edge = self.compute_edge(self.real_ins_compact)
        self.real_ins_transformed_big_edge = self.compute_edge(self.real_ins_transformed_big)

        self.real_input_small = torch.cat((self.real_seg_small, self.real_edge_small), dim=1)

    def set_conditional_input(self, cond_seg_small, cond_ins_small, cond_seg_big, cond_ins_big):
        self.cond_seg_small = Variable(self.Tensor(self.batch_size, self.nClass, self.opt.image_sizey_small, self.opt.image_sizex_small))
        self.cond_ins_small = Variable(self.Tensor(self.batch_size, 1, self.opt.image_sizey_small, self.opt.image_sizex_small))
        self.cond_seg_big = Variable(self.Tensor(self.batch_size, self.nClass, self.opt.image_sizey_big, self.opt.image_sizex_big))
        self.cond_ins_big = Variable(self.Tensor(self.batch_size, 1, self.opt.image_sizey_big, self.opt.image_sizex_big))

        # stack tensors
        for i in range(len(cond_seg_big)):
            self.cond_seg_small[i, :, :, :] = self.transform(cond_seg_small[i])
            self.cond_ins_small[i, :, :, :] = self.transform(cond_ins_small[i])
            self.cond_seg_big[i, :, :, :] = self.transform(cond_seg_big[i])
            self.cond_ins_big[i, :, :, :] = self.transform(cond_ins_big[i])

        if self.opt.use_gpu:
            self.cond_seg_small = self.cond_seg_small.cuda()
            self.cond_ins_small = self.cond_ins_small.cuda()
            self.cond_seg_big = self.cond_seg_big.cuda()
            self.cond_ins_big = self.cond_ins_big.cuda()

        self.cond_edge_small = self.compute_edge_for_input(self.cond_ins_small)
        self.cond_edge_big = self.compute_edge_for_input(self.cond_ins_big)

    def reparameterize(self, mu, logvar, mode):
        if mode == 'train':
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def compute_edge(self, map):
        horizontal_edge1 = self.conv_edge_horizontal1(map)
        horizontal_edge2 = self.conv_edge_horizontal2(map)
        horizontal_edge = torch.max(horizontal_edge1, horizontal_edge2)
        vertical_edge1 = self.conv_edge_vertical1(map)
        vertical_edge2 = self.conv_edge_vertical2(map)
        vertical_edge = torch.max(vertical_edge1, vertical_edge2)
        return torch.max(horizontal_edge, vertical_edge)

    def compute_edge_for_input(self, map):
        horizontal_edge1 = self.conv_edge_horizontal1(map)
        horizontal_edge2 = self.conv_edge_horizontal2(map)
        horizontal_edge = torch.max(horizontal_edge1, horizontal_edge2)
        vertical_edge1 = self.conv_edge_vertical1(map)
        vertical_edge2 = self.conv_edge_vertical2(map)
        vertical_edge = torch.max(vertical_edge1, vertical_edge2)
        edge = torch.max(horizontal_edge, vertical_edge) > 0
        return edge.type('torch.cuda.FloatTensor')

    def pad_to_nClass_small(self, x):
        if self.target_channel == 0:
            padded = torch.cat((x, self.pad_small), 1)
        elif self.target_channel == (self.nClass - 1):
            padded = torch.cat((self.pad_small, x), 1)
        else:
            padded = torch.cat((self.pad_before_small, x, self.pad_after_small), 1)

        return padded

    def pad_to_nClass_big(self, x):
        if self.target_channel == 0:
            padded = torch.cat((x, self.pad_big), 1)
        elif self.target_channel == (self.nClass - 1):
            padded = torch.cat((self.pad_big, x), 1)
        else:
            padded = torch.cat((self.pad_before_big, x, self.pad_after_big), 1)

        return padded

    def forward_where_sup(self, mode):
        self.where_sup_mu, self.where_sup_logvar = self.where_encoder_sup(self.real_ins_theta)
        self.where_sup_reparam = self.reparameterize(self.where_sup_mu, self.where_sup_logvar, mode)
        self.where_sup_reparam_expand = self.where_sup_reparam.expand([-1, -1, self.opt.image_sizey_small, self.opt.image_sizex_small], 1)

        self.real_input_small_and_reparam = torch.cat([self.real_input_small, self.where_sup_reparam_expand], 1)

        self.where_sup_embed, _ = self.where_encoder(self.real_input_small_and_reparam)
        self.where_sup_transformed, _, self.stn_theta_where_sup = \
            self.stn(self.bb_white, self.where_sup_embed, self.rotFix, self.opt.image_sizex_small, self.opt.image_sizey_small)

        self.where_sup_pad = self.pad_to_nClass_small(self.where_sup_transformed)
        self.where_sup_masked = self.real_seg_small * (1. - self.where_sup_transformed)
        self.where_sup_bb_on_seg = self.where_sup_masked + self.where_sup_pad

    def forward_where_sup_with_z(self, mode):
        if mode == 'train':
            self.where_encoder.train()
        else:
            self.where_encoder.eval()

        self.real_input_small_with_z = torch.cat([self.real_input_small, self.input_z_spatial_expand], 1)
        self.where_sup_embed_using_z, self.recon_real_z_spatial = self.where_encoder(self.real_input_small_with_z)
        self.real_recon_small = self.where_reconstructor(self.where_sup_embed_using_z)

        self.where_sup_transformed_using_z, _, self.stn_theta_using_z = \
            self.stn(self.bb_white, self.where_sup_embed_using_z, self.rotFix, self.opt.image_sizex_small, self.opt.image_sizey_small)

    def forward(self, mode):
        if mode == 'train':
            self.where_encoder.train()
            self.what_encoder.train()
        else:
            self.where_encoder.eval()
            self.what_encoder.eval()

        # prepare input
        self.cond_input_small = torch.cat((self.cond_seg_small, self.cond_edge_small), dim=1)
        self.cond_input_small_and_z_spatial = torch.cat((self.cond_input_small, self.input_z_spatial_expand), 1)

        # embed input
        self.embed_small, self.recon_z_spatial = self.where_encoder(self.cond_input_small_and_z_spatial)
        self.cond_recon_small = self.where_reconstructor(self.embed_small)

        # stn
        self.bb_transformed_small, self.grid, self.stn_theta = \
            self.stn(self.bb_white, self.embed_small, self.rotFix, self.opt.image_sizex_small, self.opt.image_sizey_small)
        self.bb_transformed_big, _, _ = \
            self.stn(self.bb_white, self.embed_small, self.rotFix, self.opt.image_sizex_big, self.opt.image_sizey_big)

        # post processing
        self.bb_padded_small = self.pad_to_nClass_small(self.bb_transformed_small)
        self.bb_on_seg_small_masked = self.cond_seg_small * (1. - self.bb_transformed_small)
        self.bb_on_seg_small = self.bb_on_seg_small_masked + self.bb_padded_small
        self.bb_on_edge_small = self.cond_edge_small * (1. - self.bb_transformed_small)

        self.bb_padded_big = self.pad_to_nClass_big(self.bb_transformed_big)
        self.bb_on_seg_big_masked = self.cond_seg_big * (1. - self.bb_transformed_big)
        self.bb_on_seg_big = self.bb_on_seg_big_masked + self.bb_padded_big
        self.bb_on_edge_big = self.cond_edge_big * (1. - self.bb_transformed_big)

        # start to generate shape
        # prepare input
        self.bb_on_cond_big = torch.cat([self.bb_on_seg_big, self.bb_on_edge_big], 1)
        self.cond_input_big_and_z_appr = torch.cat([self.bb_on_cond_big, self.input_z_appr_expand], 1)

        # embed input
        self.embed_big, self.recon_z_appr = self.what_encoder(self.cond_input_big_and_z_appr)

        # generate
        self.newobj = self.generator(self.embed_big)
        self.newobj_edge = self.compute_edge(self.newobj)
        self.cond_recon_big = self.what_reconstructor(self.embed_big)

        # use the shared stn to put new obj on map
        self.newobj_transformed_big, _, _ = \
            self.stn(self.newobj, self.embed_small, self.rotFix, self.opt.image_sizex_big, self.opt.image_sizey_big)
        self.newobj_transformed_edge_big = self.compute_edge(self.newobj_transformed_big)

        # post processing
        self.newobj_padded_big = self.pad_to_nClass_big(self.newobj_transformed_big)
        self.newobj_on_seg_big_masked = self.cond_seg_big * (1. - self.newobj_transformed_big)
        self.newobj_on_seg_big = self.newobj_on_seg_big_masked + self.newobj_padded_big

        self.newobj_on_edge_big_masked = self.cond_edge_big * (1 - self.newobj_transformed_big)
        self.newobj_on_edge_big = self.newobj_on_edge_big_masked + self.newobj_transformed_edge_big

    def forward_ins_sup(self, mode):
        # prepare input
        self.bb_on_real_big = torch.cat([self.bb_on_real_seg_big, self.bb_on_real_edge_big], 1)

        self.real_ins_mu, self.real_ins_logvar = self.what_encoder_sup(self.real_ins_compact)
        self.real_ins_reparam = self.reparameterize(self.real_ins_mu, self.real_ins_logvar, mode)
        self.real_ins_reparam_expand = self.real_ins_reparam.expand([-1, -1, self.opt.image_sizey_big, self.opt.image_sizex_big])

        self.bb_on_real_big_and_reparam = torch.cat([self.bb_on_real_big, self.real_ins_reparam_expand], 1)

        self.real_embed_using_reparam, _ = self.what_encoder(self.bb_on_real_big_and_reparam)
        self.sup_newobj_using_reparam = self.generator(self.real_embed_using_reparam)
        # self.sup_real_recon_big_reparam = self.reconstructor2(self.real_embed2_using_reparam)

    def forward_ins_sup_with_z(self, mode):
        if mode == 'train':
            self.what_encoder.train()
        else:
            self.what_encoder.eval()

        self.bb_on_real_big_and_z_appr = torch.cat([self.bb_on_real_big, self.input_z_appr_expand], 1)

        self.real_embed_using_z_appr, self.recon_real_z_appr = self.what_encoder(self.bb_on_real_big_and_z_appr)
        self.sup_newobj_using_z_appr = self.generator(self.real_embed_using_z_appr)
        self.real_recon_big = self.what_reconstructor(self.real_embed_using_z_appr)

    def backward_D(self):
        self.bb_on_cond_big = torch.cat([self.bb_on_seg_big, self.bb_on_edge_big], 1)

        # forward D
        self.d_stn_real = self.disSTN(self.real_ins_theta)
        self.d_stn_fake = self.disSTN(self.stn_theta.detach())
        self.d_stn_fake2 = self.disSTN(self.stn_theta_where_sup.detach())
        self.d_stn_fake3 = self.disSTN(self.stn_theta_using_z.detach())

        self.d_where_real = self.disWhere(self.bb_on_real_seg_small, self.bb_on_real_edge_small, self.real_bb_transformed_small)
        self.d_where_fake = self.disWhere(self.bb_on_seg_small.detach(), self.bb_on_edge_small.detach(), self.bb_transformed_small.detach())

        self.d_obj_real = self.disObj(self.real_ins_compact)
        self.d_obj_fake = self.disObj(self.newobj.detach())
        self.d_obj_fake2 = self.disObj(self.sup_newobj_using_reparam.detach())
        self.d_obj_fake3 = self.disObj(self.sup_newobj_using_z_appr.detach())

        self.d_what_real = self.disWhat(self.ins_on_real_seg_big, self.ins_on_real_edge_big)
        self.d_what_fake = self.disWhat(self.newobj_on_seg_big.detach(), self.newobj_on_edge_big.detach())

        # update D stn
        true_tensor = Variable(self.Tensor(self.d_stn_real.data.size()).fill_(1.))
        fake_tensor = Variable(self.Tensor(self.d_stn_real.data.size()).fill_(0.))

        self.optimizer_D_stn.zero_grad()
        self.d_stn_loss = self.criterionGAN(self.d_stn_real, true_tensor) + \
                          self.criterionGAN(self.d_stn_fake, fake_tensor) + \
                          self.criterionGAN(self.d_stn_fake2, fake_tensor) + \
                          self.criterionGAN(self.d_stn_fake3, fake_tensor)
        self.d_stn_loss.backward()
        self.optimizer_D_stn.step()

        # update D where
        true_tensor = Variable(self.Tensor(self.d_where_real.data.size()).fill_(1.))
        fake_tensor = Variable(self.Tensor(self.d_where_real.data.size()).fill_(0.))

        self.optimizer_D_where.zero_grad()
        self.d_where_loss = self.criterionGAN(self.d_where_real, true_tensor) + \
                            self.criterionGAN(self.d_where_fake, fake_tensor)
        self.d_where_loss.backward()
        self.optimizer_D_where.step()

        # update D obj
        true_tensor = Variable(self.Tensor(self.d_obj_real.data.size()).fill_(1.))
        fake_tensor = Variable(self.Tensor(self.d_obj_real.data.size()).fill_(0.))

        self.optimizer_D_obj.zero_grad()
        self.d_obj_loss = self.criterionGAN(self.d_obj_real, true_tensor) + \
                          self.criterionGAN(self.d_obj_fake, fake_tensor) + \
                          self.criterionGAN(self.d_obj_fake2, fake_tensor) + \
                          self.criterionGAN(self.d_obj_fake3, fake_tensor)
        self.d_obj_loss.backward()
        self.optimizer_D_obj.step()

        # update D what
        true_tensor = Variable(self.Tensor(self.d_what_real.data.size()).fill_(1.))
        fake_tensor = Variable(self.Tensor(self.d_what_real.data.size()).fill_(0.))

        self.optimizer_D_what.zero_grad()
        self.d_what_loss = self.criterionGAN(self.d_what_real, true_tensor) + \
                           self.criterionGAN(self.d_what_fake, fake_tensor)
        self.d_what_loss.backward()
        self.optimizer_D_what.step()

    def backward_G(self):
        # forward D without detach
        self.d_stn_fake = self.disSTN(self.stn_theta)
        self.d_stn_fake2 = self.disSTN(self.stn_theta_where_sup)
        self.d_stn_fake3 = self.disSTN(self.stn_theta_using_z)

        self.d_where_fake = self.disWhere(self.bb_on_seg_small, self.bb_on_edge_small, self.bb_transformed_small)

        self.d_obj_fake = self.disObj(self.newobj)
        self.d_obj_fake2 = self.disObj(self.sup_newobj_using_reparam)
        self.d_obj_fake3 = self.disObj(self.sup_newobj_using_z_appr)

        self.d_what_fake = self.disWhat(self.newobj_on_seg_big, self.newobj_on_edge_big)

        # update G
        self.coord_loss = torch.max(1 + self.grid[:, :, :, 0].min(), Variable(self.Tensor(1).fill_(0.))) + \
                          torch.max(1 + self.grid[:, :, :, 1].min(), Variable(self.Tensor(1).fill_(0.))) + \
                          torch.max(1 - self.grid[:, :, :, 0].max(), Variable(self.Tensor(1).fill_(0.))) + \
                          torch.max(1 - self.grid[:, :, :, 1].max(), Variable(self.Tensor(1).fill_(0.)))

        self.stn_theta_loss = torch.max(self.stn_theta[:, 0] - self.max_width_ratio, Variable(self.Tensor(1).fill_(0.))) + \
                              torch.max(0. - self.stn_theta[:, 0], Variable(self.Tensor(1).fill_(0.))) + \
                              torch.max(self.stn_theta[:, 4] - self.max_height_ratio, Variable(self.Tensor(1).fill_(0.))) + \
                              torch.max(0. - self.stn_theta[:, 4], Variable(self.Tensor(1).fill_(0.)))

        true_tensor = Variable(self.Tensor(self.d_stn_fake.data.size()).fill_(1.))
        self.g_stn_loss = self.criterionGAN(self.d_stn_fake, true_tensor) + \
                          self.criterionGAN(self.d_stn_fake2, true_tensor) + \
                          self.criterionGAN(self.d_stn_fake3, true_tensor)

        true_tensor = Variable(self.Tensor(self.d_where_fake.data.size()).fill_(1.))
        self.g_where_loss = self.criterionGAN(self.d_where_fake, true_tensor)

        true_tensor = Variable(self.Tensor(self.d_obj_fake.data.size()).fill_(1.))
        self.g_obj_loss = self.criterionGAN(self.d_obj_fake, true_tensor) + \
                          self.criterionGAN(self.d_obj_fake2, true_tensor) + \
                          self.criterionGAN(self.d_obj_fake3, true_tensor)

        true_tensor = Variable(self.Tensor(self.d_what_fake.data.size()).fill_(1.))
        self.g_what_loss = self.criterionGAN(self.d_what_fake, true_tensor)

        self.where_KL_loss = - 0.5 * torch.mean(1 + self.where_sup_logvar - self.where_sup_mu.pow(2) - self.where_sup_logvar.exp())
        self.what_KL_loss = - 0.5 * torch.mean(1 + self.real_ins_logvar - self.real_ins_mu.pow(2) - self.real_ins_logvar.exp())

        self.where_recon_loss = torch.mean(torch.abs(self.cond_recon_small - self.cond_input_small)) + \
                                torch.mean(torch.abs(self.real_recon_small - self.real_input_small)) + \
                                torch.mean(torch.abs(self.recon_z_spatial - self.input_z_spatial)) + \
                                torch.mean(torch.abs(self.recon_real_z_spatial - self.input_z_spatial))
        self.where_vae_recon_loss = torch.mean(torch.abs(self.stn_theta_where_sup - self.real_ins_theta))

        self.what_recon_loss = torch.mean(torch.abs(self.cond_recon_big - self.bb_on_cond_big)) + \
                               torch.mean(torch.abs(self.real_recon_big - self.bb_on_real_big)) + \
                               torch.mean(torch.abs(self.recon_z_appr - self.input_z_appr)) + \
                               torch.mean(torch.abs(self.recon_real_z_appr - self.input_z_appr))
        self.what_vae_recon_loss = torch.mean(torch.abs(self.sup_newobj_using_reparam - self.real_ins_compact))

        self.where_encoder_sup_loss = 1 * self.where_KL_loss + self.where_vae_recon_loss
        self.what_encoder_sup_loss = 1 * self.what_KL_loss + self.what_vae_recon_loss

        self.g_loss = self.g_stn_loss + self.g_where_loss + self.g_obj_loss + self.g_what_loss + \
                      self.coord_loss + self.stn_theta_loss + \
                      10 * (self.where_recon_loss + self.what_recon_loss) + \
                      10 * (self.where_vae_recon_loss + self.what_vae_recon_loss)

        self.optimizer_where_encoder_sup.zero_grad()
        self.where_encoder_sup_loss.backward(retain_graph=True)
        self.optimizer_where_encoder_sup.step()

        self.optimizer_what_encoder_sup.zero_grad()
        self.what_encoder_sup_loss.backward(retain_graph=True)
        self.optimizer_what_encoder_sup.step()

        self.optimizer_G.zero_grad()
        self.g_loss.backward()
        self.optimizer_G.step()

    def save(self, epoch):
        self.save_network(self.where_encoder_sup, epoch, 'where_enc_sup')
        self.save_network(self.where_encoder, epoch, 'where_enc')
        self.save_network(self.where_reconstructor, epoch, 'where_recon')
        self.save_network(self.stn, epoch, 'stn')

        self.save_network(self.disSTN, epoch, 'disSTN')
        self.save_network(self.disWhere, epoch, 'disWhere')

        self.save_network(self.what_encoder_sup, epoch, 'what_enc_sup')
        self.save_network(self.what_encoder, epoch, 'what_enc')
        self.save_network(self.generator, epoch, 'gen')
        self.save_network(self.what_reconstructor, epoch, 'what_recon')

        self.save_network(self.disObj, epoch, 'disObj')
        self.save_network(self.disWhat, epoch, 'disWhat')

    def load(self, epoch):
        self.load_network(self.where_encoder_sup, epoch, 'where_enc_sup')
        self.load_network(self.where_encoder, epoch, 'where_enc')
        self.load_network(self.where_reconstructor, epoch, 'where_recon')
        self.load_network(self.stn, epoch, 'stn')

        self.load_network(self.disSTN, epoch, 'disSTN')
        self.load_network(self.disWhere, epoch, 'disWhere')

        self.load_network(self.what_encoder_sup, epoch, 'what_enc_sup')
        self.load_network(self.what_encoder, epoch, 'what_enc')
        self.load_network(self.generator, epoch, 'gen')
        self.load_network(self.what_reconstructor, epoch, 'what_recon')

        self.load_network(self.disObj, epoch, 'disObj')
        self.load_network(self.disWhat, epoch, 'disWhat')

    def load_where(self, epoch):
        self.load_network_where(self.where_encoder, epoch, 'where_enc')
        self.load_network_where(self.stn, epoch, 'stn')

    def save_network(self, network, epoch, net_name):
        save_filename = 'epoch_%s_net_%s.pth' % (epoch, net_name)
        save_path = os.path.join(self.opt.net_save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if self.opt.use_gpu:
            network.cuda()

    def load_network(self, network, epoch, net_name):
        save_filename = 'epoch_%s_net_%s.pth' % (epoch, net_name)
        save_path = os.path.join(self.opt.net_save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    def load_network_where(self, network, epoch, net_name):
        save_filename = 'epoch_%s_net_%s.pth' % (epoch, net_name)
        save_path = os.path.join(self.opt.net_save_dir_where, save_filename)
        network.load_state_dict(torch.load(save_path))
