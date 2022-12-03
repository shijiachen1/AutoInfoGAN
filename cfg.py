# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='../lmdbs/art_landscape_1k', help='path of resource dataset, should be a folder that has one or many sub image folders inside')
    parser.add_argument('--cuda', type=int, default=0, help='index of gpu to use')
    parser.add_argument('--name', type=str, default='skull', help='experiment name')
    parser.add_argument('--iter', type=int, default=50000, help='number of iterations')
    parser.add_argument('--start_iter', type=int, default=0, help='the iteration to start training')
    parser.add_argument('--batch_size', type=int, default=8, help='mini batch number of images')
    parser.add_argument('--num_imgs', type=int, default=400, help='mini batch number of images')
    parser.add_argument('--print_interval', type=int, default=10, help='mini batch number of images')
    parser.add_argument('--im_size', type=int, default=512, help='image resolution')
    parser.add_argument('--ckpt', type=str, default='None', help='checkpoint weight path if have one')
    parser.add_argument('--fake_ins', type=float, default=0.1, help='hyperparameters of ins loss')
    parser.add_argument('--dataloader_workers', type=int, default=4, help='dataloader_workers')
    parser.add_argument('--max_epoch', type=int, default=200, help='number of epochs of training')
    parser.add_argument('--max_iter', type=int, default=None, help='set the max iteration number')
    parser.add_argument('--eval_path', type=str)
    parser.add_argument('--g_lr', type=float, default=0.0002, help='adam: gen learning rate')
    parser.add_argument('--batch', default=16, type=int, help='eval batch size')
    parser.add_argument('--change_epoch', default=10, type=int, help='eval batch size')
    parser.add_argument('--n_sample', type=int, default=400)
    parser.add_argument('--fid_weight', type=int, default=0.001)
    parser.add_argument('--artifacts', type=str, default=".", help='path to artifacts.')
    parser.add_argument('--start_epoch', type=int, default=5)
    parser.add_argument('--end_epoch', type=int, default=5)
    parser.add_argument(
        '--d_lr',
        type=float,
        default=0.0002,
        help='adam: disc learning rate')
    parser.add_argument(
        '--ctrl_lr',
        type=float,
        default=3.5e-4,
        help='adam: ctrl learning rate')
    parser.add_argument(
        '--lr_decay',
        action='store_true',
        help='learning rate decay or not')
    parser.add_argument(
        '--beta1',
        type=float,
        default=0.5,
        help='adam: decay of first order momentum of gradient')
    parser.add_argument(
        '--beta2',
        type=float,
        default=0.999,
        help='adam: decay of first order momentum of gradient')
    parser.add_argument(
        '--latent_dim',
        type=int,
        default=256,
        help='dimensionality of the latent space')

    parser.add_argument(
        '--channels',
        type=int,
        default=3,
        help='number of image channels')

    parser.add_argument(
        '--val_freq',
        type=int,
        default=20,
        help='interval between each validation')
    parser.add_argument(
        '--print_freq',
        type=int,
        default=100,
        help='interval between each verbose')
    parser.add_argument(
        '--load_path',
        type=str,
        help='The reload model path')
    parser.add_argument(
        '--exp_name',
        default='autosearch',
        type=str,
        help='The name of exp')

    parser.add_argument('--init_type', type=str, default='normal',
                        choices=['normal', 'orth', 'xavier_uniform', 'false'],
                        help='The init type')
    parser.add_argument('--gf_dim', type=int, default=64,
                        help='The base channel num of gen')
    parser.add_argument('--df_dim', type=int, default=64,
                        help='The base channel num of disc')
    parser.add_argument(
        '--gen_model',
        type=str,
        default='shared_gan',
        help='path of gen model')
    parser.add_argument(
        '--dis_model',
        type=str,
        default='shared_gan',
        help='path of dis model')
    parser.add_argument(
        '--controller',
        type=str,
        default='controller',
        help='path of controller')
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--num_eval_imgs', type=int, default=50000)

    parser.add_argument('--random_seed', type=int, default=12345)

    # search
    parser.add_argument('--shared_epoch', type=int, default=30,
                        help='the number of epoch to train the shared gan at each search iteration')
    parser.add_argument('--shared_epoch1', type=int, default=15,
                        help='the number of epoch to train the shared gan at 1 stage search')
    parser.add_argument('--shared_epoch2', type=int, default=30,
                        help='the number of epoch to train the shared gan at 2 stage search')
    parser.add_argument('--grow_step', type=int, default=30,
                        help='the number of epoch to train the shared gan at 2 stage search')
    parser.add_argument('--max_search_iter', type=int, default=60,
                        help='max search iterations of this algorithm')
    parser.add_argument('--ctrl_step', type=int, default=30,
                        help='number of steps to train the controller at each search iteration')
    parser.add_argument('--ctrl_step1', type=int, default=15,
                        help='number of steps to train the controller at each search iteration')
    parser.add_argument('--ctrl_step2', type=int, default=30,
                        help='number of steps to train the controller at each search iteration')
    parser.add_argument('--ctrl_sample_batch', type=int, default=1,
                        help='sample size of controller of each step')
    parser.add_argument('--hid_size', type=int, default=100,
                        help='the size of hidden vector')
    parser.add_argument('--baseline_decay', type=float, default=0.9,
                        help='baseline decay rate in RL')
    parser.add_argument('--rl_num_eval_img', type=int, default=5000,
                        help='number of images to be sampled in order to get the reward')
    parser.add_argument('--get_top_eval_img', type=int, default=5000,
                        help='number of images to be sampled in order to get the reward')
    parser.add_argument('--num_candidate', type=int, default=10,
                        help='number of candidate architectures to be sampled')
    parser.add_argument('--topk', type=int, default=5,
                        help='preserve topk models architectures after each stage' )
    parser.add_argument('--entropy_coeff', type=float, default=1e-3,
                        help='to encourage the exploration')
    parser.add_argument('--dynamic_reset_threshold', type=float, default=1e-3,
                        help='var threshold')
    parser.add_argument('--dynamic_reset_window', type=int, default=500,
                        help='the window size')
    parser.add_argument('--arch', nargs='+', type=int,
                        help='the vector of a discovered architecture')

    opt = parser.parse_args()

    return opt
