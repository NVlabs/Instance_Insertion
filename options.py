  # Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
  # 
  # Licensed under the CC BY-NC-SA 4.0 license 
  #     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
  #

import argparse
import os

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        ### DATABASE OPTIONS ###
        self.parser.add_argument('--db_root', default='/mnt/hi/dataset/cityscapes')
        self.parser.add_argument('--target_class', default='person')

        ### TRAINING OPTIONS ###
        self.parser.add_argument('--epoch', type=int, default=100)
        self.parser.add_argument('--learning_rate', type=float, default=0.0002)
        self.parser.add_argument('--beta1', type=float, default=0.5)
        self.parser.add_argument('--batch_size', type=int, default=1)
        self.parser.add_argument('--image_sizex_small', type=int, default=256)
        self.parser.add_argument('--image_sizey_small', type=int, default=128)
        self.parser.add_argument('--image_sizex_big', type=int, default=1024)
        self.parser.add_argument('--image_sizey_big', type=int, default=512)
        self.parser.add_argument('--compact_sizex', type=int, default=128)
        self.parser.add_argument('--compact_sizey', type=int, default=128)
        self.parser.add_argument('--c_dim', type=int, default=3)
        self.parser.add_argument('--df_dim', type=int, default=8)
        self.parser.add_argument('--gf_dim', type=int, default=32)
        self.parser.add_argument('--z_dim_appr', type=int, default=16)
        self.parser.add_argument('--z_dim_spatial', type=int, default=4)
        self.parser.add_argument('--embed_dim_small', type=int, default=128)
        self.parser.add_argument('--embed_dim_big', type=int, default=512)
        self.parser.add_argument('--cont_train', default=False)
        self.parser.add_argument('--start_epoch', default=0)
        self.parser.add_argument('--sample_dir', default='results/samples')
        self.parser.add_argument('--net_dir', default='trained_model')
        self.parser.add_argument('--num_samples', type=int, default=128)

        ### MISC ###
        self.parser.add_argument('--use_gpu', default=True)
        self.parser.add_argument('--gpu_id', type=int, default=0)
        self.parser.add_argument('--random_seed', default=1004)

    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        for k,v in sorted(args.items()):
            print('%s: %s' %(str(k), str(v)))

        self.opt.output_size = self.opt.image_sizey_big
        self.prepare_folders()
        
        return self.opt

    def prepare_folders(self):
        save_dir_str = 'o' + str(self.opt.output_size) + '_b' + str(self.opt.batch_size)
        self.opt.sample_dir = os.path.join(self.opt.sample_dir, save_dir_str)
        self.opt.net_save_dir = os.path.join(self.opt.net_dir, save_dir_str)

        if not os.path.exists(self.opt.sample_dir):
            os.makedirs(self.opt.sample_dir)
        if not os.path.exists(self.opt.net_save_dir):
            os.makedirs(self.opt.net_save_dir)
