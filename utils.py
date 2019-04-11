  # Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
  # 
  # Licensed under the CC BY-NC-SA 4.0 license 
  #     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
  #

from PIL import Image
import numpy as np
import torchvision.transforms as transforms

palette_toy = [0,0,0, 0, 0, 255, 0, 255, 0, 255, 0, 0]
zero_pad = 256 * 3 - len(palette_toy)
for i in range(zero_pad):
    palette_toy.append(0)

palette_cityscape = [0,0,0, 0,0,0, 111,74,0, 81,0,81, 128, 64, 128, 244, 35, 232, 250,170,160, 230,150,140,
                     70, 70, 70, 102, 102, 156, 190, 153, 153, 180,165,180, 150,100,100, 150,120,90, 153, 153, 153,
                     250, 170, 30, 220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0,
                     0, 0, 142, 0, 0, 70, 0, 60, 100, 0,0,90, 0,0,110, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette_cityscape)
for i in range(zero_pad):
    palette_cityscape.append(0)

def colorize_mask(mask, dataset):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    if dataset == 'toy':
        new_mask.putpalette(palette_toy)
    else:
        new_mask.putpalette(palette_cityscape)

    return new_mask

def prepare_input(image_list, isFlip, opt, mode=None):
    if mode == 'seg_small':
        images = [get_seg_channels(image_list[i], isFlip, opt.image_sizex_small, opt.image_sizey_small, opt.nClass, opt.chn_and_segID) for i in range(len(image_list))]
    elif mode == 'seg_big':
        images = [get_seg_channels(image_list[i], isFlip, opt.image_sizex_big, opt.image_sizey_big, opt.nClass, opt.chn_and_segID) for i in range(len(image_list))]
    elif mode == 'insMap':
        images = [get_insMap(image_list[i], isFlip) for i in range(len(image_list))]

    return images

def get_seg_channels(seg_path, isFlip, sizex, sizey, nClass, chn_and_segID):
    seg_ = Image.open(seg_path)
    if isFlip:
        seg_ = seg_.transpose(Image.FLIP_LEFT_RIGHT)

    seg_ = seg_.resize([sizex, sizey])

    seg_ = np.array(seg_)
    seg = seg_.copy()
    for k, v in chn_and_segID.items():
        seg[seg_ == k] = v

    # change 1 channel seg map to n channel binary maps
    mask = np.zeros((seg.shape[0], seg.shape[1], nClass), np.uint8)
    for n in range(nClass):
        mask[seg == n, n] = 255

    return mask

def get_insMap(ins_path, isFlip):
    ins = Image.open(ins_path)
    if isFlip:
        ins = ins.transpose(Image.FLIP_LEFT_RIGHT)

    return ins

def seg_to_single_channel(seg, order='hwc'):
    if order=='chw':
        seg = np.transpose(np.squeeze(seg), [1,2,0])
    single_channel = np.argmax(seg, axis=2).astype(np.uint8)

    return single_channel

def gaussian_filter(shape=(5,5), sigma=1):
    x, y = [edge /2 for edge in shape]
    grid = np.array([[((i**2+j**2)/(2.0*sigma**2)) for i in xrange(-x, x+1)] for j in xrange(-y, y+1)])
    g_filter = np.exp(-grid)/(2*np.pi*sigma**2)
    g_filter /= np.sum(g_filter)
    return g_filter