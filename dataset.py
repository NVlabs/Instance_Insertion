  # Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
  # 
  # Licensed under the CC BY-NC-SA 4.0 license 
  #     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
  #


import os.path
import numpy as np

class Dataset():
    def __init__(self, opt):
        self.opt = opt
        self.db_root = opt.db_root

    def load(self):
        self.target_class = self.opt.target_class

        if self.target_class == 'person':
            self.segID = '24'
            self.opt.target_channel = 20
            self.opt.min_size_percent = 0.2
            self.opt.min_area_ratio = 0.4
            self.opt.max_aspect_ratio = 4
            self.opt.max_width_ratio = 50
            self.opt.max_height_ratio = 10
        elif self.target_class == 'car':
            self.segID = '26'
            self.opt.target_channel = 22
            self.opt.min_size_percent = 0.5
            self.opt.min_area_ratio = 0.4
            self.opt.max_aspect_ratio = 3
            self.opt.max_width_ratio = 30
            self.opt.max_height_ratio = 10
        else:
            print('wrong object')

        self.seg_list, self.ins_list = self.cityscape_load_seg_and_ins(self.opt.db_root, quality='fine', mode='train')
        self.test_seg_list, self.test_ins_list = self.cityscape_load_seg_and_ins(self.opt.db_root, quality='fine', mode='val')

        self.opt.nClass = 29 + 1
        ignore_label = 0
        self.opt.ignore_label = ignore_label
        self.opt.chn_and_segID = {-1: 22, 0: ignore_label, 1: 1, 2: ignore_label,
                                  3: ignore_label, 4: ignore_label, 5: 2, 6: 3,
                                  7: 4, 8: 5, 9: 6, 10: 7, 11: 8, 12: 9, 13: 10,
                                  14: 11, 15: 12, 16: 13, 17: 14,
                                  18: 14, 19: 15, 20: 16, 21: 17, 22: 18, 23: 19, 24: 20, 25: 21, 26: 22,
                                  27: 23, 28: 24, 29: 25, 30: 26, 31: 27, 32: 28, 33: 29}

    def cityscape_load_seg_and_ins(self, root, quality, mode):
        assert (quality == 'fine' and mode in ['train', 'val']) or \
               (quality == 'coarse' and mode in ['train', 'train_extra', 'val'])

        if quality == 'coarse':
            seg_path = os.path.join(root, 'gtCoarse', 'gtCoarse', mode)
            seg_postfix = '_gtCoarse_labelIds.png'
            ins_postfix = '_gtCoarse_instanceIds.png'
        else:
            seg_path = os.path.join(root, 'gtFine', mode)
            seg_postfix = '_gtFine_labelIds.png'
            ins_postfix = '_gtFine_instanceIds.png'

        img_path = os.path.join(root, 'leftImg8bit', mode)
        seg_list = []
        ins_list = []
        categories = os.listdir(seg_path)
        for c in categories:
            c_items = [name.split('_leftImg8bit.png')[0] for name in os.listdir(os.path.join(img_path, c))]
            for it in c_items:
                seg_list.append(os.path.join(seg_path, c, it + seg_postfix))
                ins_list.append(os.path.join(seg_path, c, it + ins_postfix))
        return np.array(seg_list), np.array(ins_list)