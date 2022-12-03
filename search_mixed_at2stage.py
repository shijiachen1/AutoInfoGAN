# -*- coding: utf-8 -*-
# @Date    : 2019-09-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import cfg
from functions import train_shared_mixed, train_controller, get_topk_arch_hidden, train_shared_mixed_NoCL
from utils.utils import set_log_dir, save_checkpoint, create_logger, RunningStats
# from utils.inception_score import _init_inception
# from utils.fid_score import create_inception_graph, check_or_download_inception

from operation import ImageFolder, InfiniteSamplerWrapper

from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from models_search.shared_gan_2stage import GeneratorNoAttnOrigin,Discriminator
from models_search.controller import Controller,Controller3,ControllerNoAttn,ControllerSkip
from models import CLHead

import torch
import os
import torch.nn as nn
from tensorboardX import SummaryWriter
from torchvision import utils as vutils
from tqdm import tqdm

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0")


# def create_ctrler_skip(args, weights_init):
#     controller = ControllerSkip(args=args).cuda()
#     controller.apply(weights_init)
#     ctrl_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, controller.parameters()),
#                                      args.ctrl_lr, (args.beta1, args.beta2))
#     return controller, ctrl_optimizer

def create_ctrler_attn(args, weights_init):
    controller = ControllerNoAttn(args=args).cuda()
    controller.apply(weights_init)
    ctrl_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, controller.parameters()),
                                     args.ctrl_lr, (args.beta1, args.beta2))
    return controller, ctrl_optimizer

def create_shared_gan(args, weights_init, cur_stage=1):
    gen_net = GeneratorNoAttnOrigin(ngf=args.gf_dim, nz=args.latent_dim, nc=3, im_size=args.im_size, cur_stage=cur_stage).cuda()
    dis_net = Discriminator(ndf=args.df_dim, nc=3, im_size=args.im_size, cur_stage=cur_stage).cuda()
    gen_net.apply(weights_init)
    dis_net.apply(weights_init)
    gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_net.parameters()),
                                     args.g_lr, (args.beta1, args.beta2))
    dis_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dis_net.parameters()),
                                     args.d_lr, (args.beta1, args.beta2))
    return gen_net, dis_net, gen_optimizer, dis_optimizer

# def create_clhead(args, weights_init, cur_stage=1):
#
#     clhead_big = CLHead(inplanes=args.gf_dim * 8)
#     clhead_big.apply(weights_init)
#     clhead_small = CLHead(inplanes=args.gf_dim * 4)
#     clhead_small.apply(weights_init)
#
#     clhead_big.to(device)
#     clhead_small.to(device)
#
#     optimizerCL_big = torch.optim.Adam(clhead_big.mlp.parameters(), args.g_lr, (args.beta1, args.beta2))
#     optimizerCL_small = torch.optim.Adam(clhead_small.mlp.parameters(), args.g_lr, (args.beta1, args.beta2))
#
#     return clhead_big, clhead_small, optimizerCL_big, optimizerCL_small

def main():
    args = cfg.parse_args()
    torch.cuda.manual_seed(args.random_seed)

    # # set tf env
    # _init_inception()
    # inception_path = check_or_download_inception(None)
    # create_inception_graph(inception_path)

    # weight init
    # weight init
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            if args.init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif args.init_type == 'orth':
                nn.init.orthogonal_(m.weight.data)
            elif args.init_type == 'xavier_uniform':
                nn.init.xavier_uniform(m.weight.data, 1.)
            else:
                raise NotImplementedError('{} unknown inital type'.format(args.init_type))
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    change_step = [i*args.change_epoch for i in range(1, (args.max_search_iter // args.change_epoch) + 1)]

    bk_dic = {}

    datasets_dic = {0: './data/few-shot-images/100-shot-grumpy_cat',
                    1: './data/few-shot-images/100-shot-obama/img',
                    2: './data/few-shot-images/100-shot-panda',
                    3: './data/few-shot-images/AnimalFace-cat/img',
                    4: './data/few-shot-images/AnimalFace-dog/img',
                    5: './data/few-shot-images/anime-face/img',
                    # 6: './data/few-shot-images/art-painting/img',
                    # 7: './data/few-shot-images/fauvism-still-life/img',
                    # 8: './data/few-shot-images/flat-colored/patterns',
                    # 9: './data/few-shot-images/moongate/img',
                    # 10: './data/few-shot-images/pokemon/img',
                    # 11: './data/few-shot-images/shells/img',
                    # 12: './data/few-shot-images/skulls/img',
                    }

    # data_idex = random.sample(datasets_dic.keys(),k=1)
    # if args.data_root:
    #     data_root = args.data_root
    # else:
    #     data_root = datasets_dic[int(data_idex[0])]
    #     datasets_dic.pop(data_idex[0])

    # initial
    start_search_iter = 0

    # set writer
    if args.load_path:
        print(f'=> resuming from {args.load_path}')
        assert os.path.exists(args.load_path)
        checkpoint_file = os.path.join(args.load_path, 'Model', 'checkpoint.pth')
        assert os.path.exists(checkpoint_file)
        checkpoint = torch.load(checkpoint_file)
        # set controller && its optimizer
        cur_stage = checkpoint['cur_stage']
        gen_net, dis_net, gen_optimizer, dis_optimizer = create_shared_gan(args, weights_init, cur_stage)
        # clhead_big, clhead_small, optimizerCL_big, optimizerCL_small = create_clhead(args, weights_init)
        # controller_skip, ctrl_skip_optimizer = create_ctrler_skip(args, weights_init)
        controller_attn, ctrl_attn_optimizer = create_ctrler_attn(args, weights_init)

        start_search_iter = checkpoint['search_iter']
        gen_net.load_state_dict(checkpoint['gen_state_dict'])
        dis_net.load_state_dict(checkpoint['dis_state_dict'])
        # clhead_big.load_state_dict(checkpoint['clhead_big'])
        # clhead_small.load_state_dict(checkpoint['clhead_samll'])
        # controller_skip.load_state_dict(checkpoint['ctrl_skip_state_dict'])
        controller_attn.load_state_dict(checkpoint['ctrl_attn_state_dict'])

        gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        dis_optimizer.load_state_dict(checkpoint['dis_optimizer'])
        # ctrl_skip_optimizer.load_state_dict(checkpoint['ctrl_skip_optimizer'])
        ctrl_attn_optimizer.load_state_dict(checkpoint['ctrl_attn_optimizer'])
        # optimizerCL_big.load_state_dict(checkpoint['cloptimizer_big'])
        # optimizerCL_small.load_state_dict(checkpoint['cloptimizer_small'])

        top_skip_archs = checkpoint['top_skip_archs']
        data_root = checkpoint['data_root']
        datasets_dic = checkpoint['data_dic']
        args.path_helper = checkpoint['path_helper']
        logger = create_logger(args.path_helper['log_path'])
        logger.info(f'=> loaded checkpoint {checkpoint_file} (search iteration {start_search_iter})')
    else:
        # create new log dir
        assert args.exp_name
        args.path_helper = set_log_dir('logs', args.exp_name)
        logger = create_logger(args.path_helper['log_path'])
        data_idex = random.sample(datasets_dic.keys(), k=1)
        data_root = datasets_dic[int(data_idex[0])]
        datasets_dic.pop(data_idex[0])
    #     top_skip_archs = None
    #     cur_stage = 1
    #
    #     # set controller && its optimizer
    #     controller_skip, ctrl_skip_optimizer = create_ctrler_skip(args, weights_init)
    #     controller_attn, ctrl_attn_optimizer = create_ctrler_attn(args, weights_init)
        cur_stage = 2
        gen_net, dis_net, gen_optimizer, dis_optimizer = create_shared_gan(args, weights_init, cur_stage)
        # clhead_big, clhead_small, optimizerCL_big, optimizerCL_small = create_clhead(args, weights_init, cur_stage)
        # controller_skip, ctrl_skip_optimizer = create_ctrler_skip(args, weights_init)
        controller_attn, ctrl_attn_optimizer = create_ctrler_attn(args, weights_init)
        top_skip_archs = [torch.tensor([0, 0, 0, 1, 2]).cuda(), torch.tensor([0, 0, 0, 0, 1]).cuda(), torch.tensor([0, 0, 0, 0, 1]).cuda(), torch.tensor([0, 0, 0, 0, 2]).cuda(), torch.tensor([0, 0, 0, 0, 1]).cuda()]

    # set up data_loader
    # data_root = './data/few-shot-images/moongate/img'
    logger.info(f"<Initial Dataset is: {data_root[22:]}>")
    transform_list = [
        transforms.Resize((int(args.im_size), int(args.im_size))),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
    trans = transforms.Compose(transform_list)

    if 'lmdb' in data_root:
        from operation import MultiResolutionDataset
        dataset = MultiResolutionDataset(data_root, trans, 1024)
    else:
        dataset = ImageFolder(root=data_root, transform=trans)

    dataloader = iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                 sampler=InfiniteSamplerWrapper(dataset), num_workers=args.dataloader_workers,
                                 pin_memory=True))

    transform_list_dist = [
        transforms.Resize((int(args.im_size), int(args.im_size))),
        transforms.ToTensor(),
    ]
    trans_a = transforms.Compose(transform_list_dist)
    dataset_a = ImageFolder(root=data_root, transform=trans_a)
    loader = DataLoader(dataset_a, batch_size=64, num_workers=8)
    dist = './fid_buffer/origin/'
    os.system('rm -f {}*'.format(dist))

    for i, imgs in enumerate(loader):
        for j, img in enumerate(imgs):
            vutils.save_image(img,
                              os.path.join(dist,
                                           '%d.png' % (i * 64 + j)))  # , normalize=True, range=(-1,1))
    # im_nums = min(len(dataset), args.n_sample)
    im_nums = args.n_sample
    logger.info(args)
    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
        'controller_steps': start_search_iter * args.ctrl_step
    }

    g_loss_history = RunningStats(args.dynamic_reset_window)
    d_loss_history = RunningStats(args.dynamic_reset_window)

    # train loop
    for search_iter in tqdm(range(int(start_search_iter), int(args.max_search_iter)), desc='search progress'):

        # if search_iter == args.grow_step:
        #     logger.info(f"<start search stage2>")
        #     cur_stage = 2
        #     top_skip_archs = get_topk_arch_hidden(args, controller_skip, gen_net, dist)
        #     logger.info(f"discovered archs by fid: {top_skip_archs}")


        if search_iter in change_step:
            data_idex = random.sample(datasets_dic.keys(), 1)
            data_root = datasets_dic[int(data_idex[0])]
            datasets_dic.pop(data_idex[0])
            logger.info(f"< change dataset of selected dataset {data_root[22:]}>")
            if 'lmdb' in data_root:
                from operation import MultiResolutionDataset
                dataset = MultiResolutionDataset(data_root, trans, 1024)
            else:
                dataset = ImageFolder(root=data_root, transform=trans)
            # im_nums = min(len(dataset),args.n_sample)
            dataloader = iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                         sampler=InfiniteSamplerWrapper(dataset),
                                         num_workers=args.dataloader_workers,
                                         pin_memory=True))

            dataset_a = ImageFolder(root=data_root, transform=trans_a)
            loader = DataLoader(dataset_a, batch_size=64, num_workers=8)
            dist = './fid_buffer/origin/'
            os.system('rm -f {}*'.format(dist))

            for i, imgs in enumerate(loader):
                for j, img in enumerate(imgs):
                    vutils.save_image(img,
                                      os.path.join(dist,
                                                   '%d.png' % (i * 64 + j)))  # , normalize=True, range=(-1,1))

            del gen_net, dis_net, gen_optimizer, dis_optimizer
            g_loss_history.clear()
            d_loss_history.clear()
            gen_net, dis_net, gen_optimizer, dis_optimizer = create_shared_gan(args, weights_init,
                                                                               cur_stage=cur_stage)
            # clhead_big, clhead_small, optimizerCL_big, optimizerCL_small = create_clhead(args, weights_init, cur_stage=cur_stage)

        logger.info(f"<start search iteration {search_iter}>")

            # del gen_net,dis_net,gen_optimizer,dis_optimizer,clhead_big, clhead_small, optimizerCL_big, optimizerCL_small
            # g_loss_history.clear()
            # d_loss_history.clear()
            # gen_net, dis_net, gen_optimizer, dis_optimizer = create_shared_gan(args, weights_init, cur_stage=cur_stage)
            # clhead_big, clhead_small, optimizerCL_big, optimizerCL_small = create_clhead(args, weights_init)

        # if search_iter >= args.grow_step:

        dynamic_reset = train_shared_mixed_NoCL(args, gen_net, dis_net, g_loss_history,
                                           d_loss_history, controller_attn, gen_optimizer,
                                           dis_optimizer, dataloader, top_skip_archs=top_skip_archs, cur_stage=cur_stage)
        train_controller(args, controller_attn, ctrl_attn_optimizer, gen_net, im_nums, dist, writer_dict, top_skip_archs=top_skip_archs, cur_stage=cur_stage)

        # else:
        #     dynamic_reset = train_shared_mixed(args, gen_net, dis_net, clhead_big, clhead_small, g_loss_history,
        #                                        d_loss_history, controller_skip, gen_optimizer,
        #                                        dis_optimizer, optimizerCL_big, optimizerCL_small, dataloader, top_skip_archs=top_skip_archs, cur_stage=cur_stage)
        #     train_controller(args, controller_skip, ctrl_skip_optimizer, gen_net, im_nums, dist, writer_dict, top_skip_archs=top_skip_archs, cur_stage=cur_stage)

        if dynamic_reset:
            logger.info('re-initialize share GAN')
            del gen_net, dis_net, gen_optimizer, dis_optimizer
            gen_net, dis_net, gen_optimizer, dis_optimizer = create_shared_gan(args, weights_init, cur_stage=cur_stage)
            # clhead_big, clhead_small, optimizerCL_big, optimizerCL_small = create_clhead(args, weights_init, cur_stage=cur_stage)

        save_checkpoint({
            'search_iter': search_iter + 1,
            'gen_model': args.gen_model,
            'dis_model': args.dis_model,
            'controller': args.controller,
            'gen_state_dict': gen_net.state_dict(),
            'dis_state_dict': dis_net.state_dict(),
            # 'ctrl_skip_state_dict': controller_skip.state_dict(),
            'ctrl_attn_state_dict': controller_attn.state_dict(),
            # 'clhead_big': clhead_big.state_dict(),
            # 'clhead_samll': clhead_small.state_dict(),
            'gen_optimizer': gen_optimizer.state_dict(),
            'dis_optimizer': dis_optimizer.state_dict(),
            # 'cloptimizer_big': optimizerCL_big.state_dict(),
            # 'cloptimizer_small': optimizerCL_small.state_dict(),
            # 'ctrl_skip_optimizer': ctrl_skip_optimizer.state_dict(),
            'ctrl_attn_optimizer': ctrl_attn_optimizer.state_dict(),
            'cur_stage': cur_stage,
            'top_skip_archs': top_skip_archs,
            'data_root': data_root,
            'data_dic': datasets_dic,
            'path_helper': args.path_helper
        }, False, args.path_helper['ckpt_path'])

    final_archs_fid = get_topk_arch_hidden(args, controller_attn, gen_net, dist, top_skip_archs=top_skip_archs)
    logger.info(f"discovered archs by fid: {final_archs_fid}")


if __name__ == '__main__':
    main()
