  # Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
  # 
  # Licensed under the CC BY-NC-SA 4.0 license 
  #     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
  #

from __future__ import print_function
import random
import torch.utils.data
import time
import torch.backends.cudnn as cudnn

from options import *
from dataset import *
from utils import *
from model import InstanceAdder

######################################################### Options
opt = Options().parse()
print(opt)

print("Random Seed: ", opt.random_seed)
random.seed(opt.random_seed)
torch.manual_seed(opt.random_seed)
if opt.use_gpu:
    torch.cuda.manual_seed_all(opt.random_seed)

cudnn.benchmark = True

######################################################### Dataset
db = Dataset(opt)
db.load()

num_train = len(db.seg_list)
print("Dataset done")

######################################################### Model
model = InstanceAdder(opt)
model.define_networks()
model.define_optimizers()
print("Model done")

######################################################### Training
isFlip = False
nTest = 16
test_seg_small = prepare_input(db.test_seg_list[:nTest], isFlip, opt, 'seg_small')
test_seg_big = prepare_input(db.test_seg_list[:nTest], isFlip, opt, 'seg_big')
test_ins = prepare_input(db.test_ins_list[:nTest], isFlip, opt, 'insMap')

test_ins_box = [test_ins[i].resize([opt.image_sizex_small, opt.image_sizey_small]) for i in range(nTest)]
test_ins_box = [np.expand_dims(np.array(test_ins_box[i]), 2) for i in range(nTest)]
test_ins_shape = [test_ins[i].resize([opt.image_sizex_big, opt.image_sizey_big]) for i in range(nTest)]
test_ins_shape = [np.expand_dims(np.array(test_ins_shape[i]), 2) for i in range(nTest)]

test_z_appr = torch.FloatTensor(nTest, opt.z_dim_appr, 1, 1).normal_(0, 1)
test_z_spatial = torch.FloatTensor(nTest, opt.z_dim_spatial, 1, 1).normal_(0, 1)
test_z2_appr = torch.FloatTensor(nTest, opt.z_dim_appr, 1, 1).normal_(0, 1)
test_z2_spatial = torch.FloatTensor(nTest, opt.z_dim_spatial, 1, 1).normal_(0, 1)

cnt = 0
start_time = time.time()
for epoch in range(opt.epoch):
    shuff_idx_cond = np.random.permutation(num_train)
    shuff_idx_real = np.random.permutation(num_train)
    num_batches = num_train // opt.batch_size

    for b in range(num_batches):
        continue_flag = 1

        b_real_idx = shuff_idx_real[b * opt.batch_size: b * opt.batch_size + opt.batch_size]
        b_cond_idx = shuff_idx_cond[b * opt.batch_size: b * opt.batch_size + opt.batch_size]

        b_real_seg_list, b_real_ins_list = db.seg_list[b_real_idx], db.ins_list[b_real_idx]
        b_cond_seg_list, b_cond_ins_list = db.seg_list[b_cond_idx], db.ins_list[b_cond_idx]

        b_z_appr = torch.FloatTensor(opt.batch_size, opt.z_dim_appr, 1, 1).normal_(0, 1)
        b_z_spatial = torch.FloatTensor(opt.batch_size, opt.z_dim_spatial, 1, 1).normal_(0, 1)

        if np.random.rand() > 0.5:
            isFlip = True
        else:
            isFlip = False
        b_real_seg_small = prepare_input(b_real_seg_list, isFlip, opt, 'seg_small')
        b_real_seg_big = prepare_input(b_real_seg_list, isFlip, opt, 'seg_big')
        b_real_ins_ = prepare_input(b_real_ins_list, isFlip, opt, 'insMap')

        b_real_ins_box = [np.expand_dims(np.array(b_real_ins_[i].resize([opt.image_sizex_small, opt.image_sizey_small])), 2) for i in range(opt.batch_size)]
        b_real_ins_shape = [np.expand_dims(np.array(b_real_ins_[i].resize([opt.image_sizex_big, opt.image_sizey_big])), 2) for i in range(opt.batch_size)]
        b_real_ins = [np.array(b_real_ins_[i]) for i in range(opt.batch_size)]

        b_real_ins_idx = []
        has_ins = 1
        for i in range(opt.batch_size):
            contained_idx = np.unique(np.squeeze(b_real_ins_box[i]) *
                                      (b_real_seg_small[i][:, :, opt.target_channel] > 0))
            b_real_ins_idx.append(contained_idx)
            if len(contained_idx) == 0:
                has_ins = 0

        if has_ins:
            b_ins_random = []
            b_ins_compact = []
            for i in range(opt.batch_size):
                idx_valid_ins = []
                for j in range(1, len(b_real_ins_idx[i])):
                    ins_map = np.squeeze(b_real_ins_shape[i] == b_real_ins_idx[i][j])
                    ins_area = np.sum(ins_map)

                    if ins_area > (opt.image_sizey_big * opt.image_sizex_big * opt.min_size_percent / 100):
                        yy, xx = np.where(ins_map > 0)
                        height = float(yy.max() - yy.min() + 1)
                        width = float(xx.max() - xx.min() + 1)

                        box_area = height * width
                        area_ratio = float(ins_area) / box_area

                        width_ratio = float(opt.image_sizex_big) / width
                        height_ratio = float(opt.image_sizey_big) / height

                        aspect_ratio = np.max((height / width, width / height))

                        if width_ratio < opt.max_width_ratio and height_ratio < opt.max_height_ratio and \
                           area_ratio > opt.min_area_ratio and aspect_ratio < opt.max_aspect_ratio:
                            idx_valid_ins.append(b_real_ins_idx[i][j])

                if len(idx_valid_ins) == 0:
                    continue_flag = 0
                else:
                    idx = np.random.choice(idx_valid_ins)
                    picked_ins_map = (b_real_ins_shape[i] == idx).astype(np.uint8) * 255
                    b_ins_random.append(picked_ins_map)

                    picked_ins_map_original_size = (b_real_ins[i] == idx).astype(np.uint8) * 255
                    yy, xx = np.where(picked_ins_map_original_size > 0)

                    compact = picked_ins_map_original_size[yy.min():yy.max() + 1, xx.min():xx.max() + 1]
                    compact = Image.fromarray(compact).resize((opt.compact_sizey, opt.compact_sizex), Image.BILINEAR)
                    b_ins_compact.append(compact)

            if continue_flag:
                b_theta = []
                for i in range(opt.batch_size):
                    map = np.squeeze(b_ins_random[i])
                    yy, xx = np.where(map > 0)

                    yt, yb = float(yy.min()) / opt.image_sizey_big * 2 - 1, float(yy.max() + 1) / opt.image_sizey_big * 2 - 1
                    xl, xr = float(xx.min()) / opt.image_sizex_big * 2 - 1, float(xx.max() + 1) / opt.image_sizex_big * 2 - 1

                    theta11 = 2 / (xr - xl)
                    theta13 = 1 - 2 * xr / (xr - xl)
                    theta22 = 2 / (yb - yt)
                    theta23 = 1 - 2 * yb / (yb - yt)

                    b_theta.append(np.array([theta11, 0., theta13, 0., theta22, theta23]))

                if np.random.rand() > 0.5:
                    isFlip = True
                else:
                    isFlip = False
                b_cond_seg_small = prepare_input(b_cond_seg_list, isFlip, opt, 'seg_small')
                b_cond_seg_big = prepare_input(b_cond_seg_list, isFlip, opt, 'seg_big')
                b_cond_ins = prepare_input(b_cond_ins_list, isFlip, opt, 'insMap')

                b_cond_ins_box = [np.expand_dims(np.array(b_cond_ins[i].resize([opt.image_sizex_small, opt.image_sizey_small])), 2) for i in range(opt.batch_size)]
                b_cond_ins_shape = [np.expand_dims(np.array(b_cond_ins[i].resize([opt.image_sizex_big, opt.image_sizey_big])), 2) for i in range(opt.batch_size)]

                # forward run
                model.set_z(b_z_appr, b_z_spatial)
                model.set_real_input(b_real_seg_small, b_real_ins_box,
                                     b_real_seg_big, b_real_ins_shape,
                                     b_ins_compact, b_theta)
                model.set_conditional_input(b_cond_seg_small, b_cond_ins_box,
                                            b_cond_seg_big, b_cond_ins_shape)

                model.forward_where_sup(mode='train')
                model.forward_where_sup_with_z(mode='train')
                model.forward_ins_sup(mode='train')
                model.forward_ins_sup_with_z(mode='train')
                model.forward(mode='train')
                model.backward_D()
                model.backward_G()

                model.forward_where_sup(mode='train')
                model.forward_where_sup_with_z(mode='train')
                model.forward_ins_sup(mode='train')
                model.forward_ins_sup_with_z(mode='train')
                model.forward(mode='train')
                model.backward_G()

                cnt += 1

                if (cnt % 100 == 1):
                    print('epoch: %02d/%02d, iter: %04d/%04d,'
                          ' dstn: %.2f, gstn: %.2f,'
                          ' dwhere: %.2f, gwhere: %.2f,'
                          ' dobj: %.2f, gobj: %.2f,'
                          ' dwhat: %.2f, gwhat: %.2f,'
                          ' theta: %.2f, KL: (%.2f, %.2f), Re: (%.2f, %.2f), %.2f sec'
                          % (epoch + 1, opt.epoch, b, num_batches,
                             model.d_stn_loss.cpu().data.numpy(), model.g_stn_loss.cpu().data.numpy(),
                             model.d_where_loss.cpu().data.numpy(), model.g_where_loss.cpu().data.numpy(),
                             model.d_obj_loss.cpu().data.numpy(), model.g_obj_loss.cpu().data.numpy(),
                             model.d_what_loss.cpu().data.numpy(), model.g_what_loss.cpu().data.numpy(),
                             model.stn_theta_loss.cpu().data.numpy(),
                             model.where_KL_loss.cpu().data.numpy(), model.what_KL_loss.cpu().data.numpy(),
                             model.where_recon_loss.cpu().data.numpy(), model.what_recon_loss.cpu().data.numpy(),
                             time.time() - start_time))

                if (cnt % 100 == 1):
                    num_img_rows = 4
                    num_img_cols = 6
                    im_h = opt.image_sizey_big
                    im_w = opt.image_sizex_big
                    image_save = Image.new('RGB',
                                           (num_img_cols * im_w + (num_img_cols - 1) * 3,
                                            num_img_rows * im_h + (num_img_rows - 1) * 3), 'green')

                    for t in range(num_img_cols):
                        model.set_conditional_input([test_seg_small[t]], [test_ins_box[t]],
                                                    [test_seg_big[t]], [test_ins_shape[t]])

                        test_seg = colorize_mask(seg_to_single_channel(model.cond_seg_big.cpu().data[0].numpy(), 'chw'), 'cityscape')

                        model.set_z(test_z_appr[t].unsqueeze(0), test_z_spatial[t].unsqueeze(0))
                        model.forward(mode='test')
                        bb1 = colorize_mask(seg_to_single_channel(model.newobj_on_seg_big.cpu().data[0].numpy(), 'chw'), 'cityscape')

                        model.set_z(test_z2_appr[t].unsqueeze(0), test_z_spatial[t].unsqueeze(0))
                        model.forward(mode='test')
                        bb2 = colorize_mask(seg_to_single_channel(model.newobj_on_seg_big.cpu().data[0].numpy(), 'chw'), 'cityscape')

                        model.set_z(test_z_appr[t].unsqueeze(0), test_z2_spatial[t].unsqueeze(0))
                        model.forward(mode='test')
                        bb3 = colorize_mask(seg_to_single_channel(model.newobj_on_seg_big.cpu().data[0].numpy(), 'chw'), 'cityscape')

                        image_save.paste(test_seg, ((im_w + 3) * t, (im_h + 3) * 0))
                        image_save.paste(bb1, ((im_w + 3) * t, (im_h + 3) * 1))
                        image_save.paste(bb2, ((im_w + 3) * t, (im_h + 3) * 2))
                        image_save.paste(bb3, ((im_w + 3) * t, (im_h + 3) * 3))

                    save_name = "epoch_%02d_iter_%04d.png" % (epoch, b)
                    save_image_path = os.path.join(opt.sample_dir, save_name)

                    image_save.save(save_image_path)
                    aa = 2

    model.save(epoch)