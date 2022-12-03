# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0


import torch.optim as optim
import torch.nn.functional as F
import torch_fidelity
import torch
import torch.nn as nn
from torchvision import utils as vutils

import logging
import operator
import os
from copy import deepcopy
import numpy as np

# import numpy as np
# from imageio import imsave
from tqdm import tqdm
from diffaug import DiffAugment

# from utils.fid_score import calculate_fid_given_paths
# from utils.inception_score import get_inception_score

device = torch.device("cuda:0")
logger = logging.getLogger(__name__)

policy = 'color,translation'
q_aug = 'color,cutout'
k_aug = 'color,translation'
import lpips
import random

percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)


# torch.backends.cudnn.benchmark = True


def crop_image_by_part(image, part):
    hw = image.shape[2] // 2
    if part == 0:
        return image[:, :, :hw, :hw]
    if part == 1:
        return image[:, :, :hw, hw:]
    if part == 2:
        return image[:, :, hw:, :hw]
    if part == 3:
        return image[:, :, hw:, hw:]


def train_d(net, data, label="real"):
    """Train function of discriminator"""
    if label == "real":
        part = random.randint(0, 3)
        pred, [rec_all, rec_small, rec_part] = net(data, label, part=part)
        err = F.relu(torch.rand_like(pred) * 0.2 + 0.8 - pred).mean() + \
              percept(rec_all, F.interpolate(data, rec_all.shape[2])).sum() + \
              percept(rec_small, F.interpolate(data, rec_small.shape[2])).sum() + \
              percept(rec_part, F.interpolate(crop_image_by_part(data, part), rec_part.shape[2])).sum()
        err.backward()
        return pred.mean().item(), rec_all, rec_small, rec_part
    else:
        pred = net(data, label)
        err = F.relu(torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
        err.backward()
        return pred.mean().item()

def train_shared(args, gen_net: nn.Module, dis_net: nn.Module, clhead: nn.Module, g_loss_history, d_loss_history, controller, gen_optimizer
                 , dis_optimizer, train_loader):
    dynamic_reset = False
    logger.info('=> train shared GAN...')
    step = 0

    # train mode
    gen_net.train()
    dis_net.train()
    batch_idex = args.num_imgs // args.batch_size
    # eval mode
    controller.eval()
    for epoch in range(args.shared_epoch):
        for iter_idx in range(batch_idex):
            imgs = next(train_loader)
            # sample an arch
            arch = controller.sample(1)[0][0]
            gen_net.set_arch(arch)
            # Adversarial ground truths

            real_image = imgs.to(device)
            current_batch_size = real_image.size(0)
            noise = torch.Tensor(current_batch_size, args.latent_dim).normal_(0, 1).to(device)

            fake_images = gen_net(noise)

            real_image = DiffAugment(real_image, policy=policy)
            fake_images = [DiffAugment(fake, policy=policy) for fake in fake_images]
            fake_q = [DiffAugment(fake, policy=q_aug) for fake in fake_images]
            fake_k = [DiffAugment(fake, policy=k_aug) for fake in fake_images]

            ## 2. train Discriminator
            dis_net.zero_grad()

            err_dr, _, _, _ = train_d(dis_net, real_image, label="real")
            train_d(dis_net, [fi.detach() for fi in fake_images], label="fake")
            dis_optimizer.step()

            ## 3. train Generator
            gen_net.zero_grad()
            pred_g = dis_net(fake_images, "fake")
            logits_q = dis_net(fake_q, "fake", mapping=True)
            logits_k = dis_net(fake_k, "fake", mapping=True)

            clhead.requires_grad_(False)
            loss_ins = clhead(logits_q, logits_k, loss_only=True)
            loss_g = torch.nn.functional.softplus(-pred_g).mean()
            # loss_g = -pred_g.mean()
            err_g = loss_g + args.fake_ins * loss_ins

            err_g.backward()
            gen_optimizer.step()

            # for p, avg_p in zip(gen_net.parameters(), avg_param_G):
            #     avg_p.mul_(0.999).add_(0.001 * p.data)
            # verbose
            if iter_idx % args.print_interval == 0:
                logger.info(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                    (epoch, args.shared_epoch, iter_idx, batch_idex, err_dr, err_g.item()))

            # check window
            if g_loss_history.is_full():
                if g_loss_history.get_var() < args.dynamic_reset_threshold \
                        or d_loss_history.get_var() < args.dynamic_reset_threshold:
                    dynamic_reset = True
                    logger.info("=> dynamic resetting triggered")
                    g_loss_history.clear()
                    d_loss_history.clear()
                    return dynamic_reset

            step += 1

    return dynamic_reset

def train_shared_mixed_NoCL(args, gen_net: nn.Module, dis_net: nn.Module, g_loss_history, d_loss_history, controller, gen_optimizer
                 , dis_optimizer, train_loader, cur_stage=1, top_skip_archs=None):
    dynamic_reset = False
    logger.info('=> train shared GAN with different datasets...')
    step = 0
    if cur_stage == 1:
        shared_epochs = args.shared_epoch1
        # latent_dim = args.latent_dim
    else:
        shared_epochs = args.shared_epoch2
        # latent_dim = args.latent_dim // 2

    # train mode
    gen_net.train()
    dis_net.train()
    batch_idex = args.num_imgs // args.batch_size

    # eval mode
    controller.eval()
    for epoch in range(shared_epochs):
        for iter_idx in range(batch_idex):
            imgs = next(train_loader)
            # sample an arch
            arch = controller.sample(1, prev_archs=top_skip_archs)[0][0]
            gen_net.set_arch(arch)
            # Adversarial ground truths

            real_image = imgs.to(device)
            # real_image = F.interpolate(real_image, args.im_size)
            current_batch_size = real_image.size(0)
            noise = torch.Tensor(current_batch_size, args.latent_dim).normal_(0, 1).to(device)

            fake_images = gen_net(noise)

            real_image = DiffAugment(real_image, policy=policy)
            fake_images = [DiffAugment(fake, policy=policy) for fake in fake_images]
            # fake_q = [DiffAugment(fake, policy=q_aug) for fake in fake_images]
            # fake_k = [DiffAugment(fake, policy=k_aug) for fake in fake_images]

            ## 2. train Discriminator
            dis_net.zero_grad()

            err_dr, _, _, _ = train_d(dis_net, real_image, label="real")
            train_d(dis_net, [fi.detach() for fi in fake_images], label="fake")
            dis_optimizer.step()

            ## 3. train Generator
            gen_net.zero_grad()
            # clhead_big.zero_grad()
            # clhead_small.zero_grad()

            pred_g = dis_net(fake_images, "fake")
            # logits_q_big, logits_q_small = dis_net(fake_q, "fake", mapping=True)
            # logits_k_big, logits_k_small = dis_net(fake_k, "fake", mapping=True)

            # print('logits_q_shape:{}, logits_k_shape:{}'.format(logits_q.size(), logits_k.size()))

            # # clhead.requires_grad_(False)
            # loss_ins_big = clhead_big(logits_q_big, logits_k_big)
            # loss_ins_small = clhead_small(logits_q_small, logits_k_small)
            # info_loss = loss_ins_big + loss_ins_small
            loss_g = -pred_g.mean()
            # err_g = loss_g + args.fake_ins * info_loss

            # err_g = -pred_g.mean()

            loss_g.backward()

            # optimizerCL_big.step()
            # optimizerCL_small.step()
            gen_optimizer.step()
            # clhead_big._momentum_update_key_encoder()
            # clhead_small._momentum_update_key_encoder()

            # for p, avg_p in zip(gen_net.parameters(), avg_param_G):
            #     avg_p.mul_(0.999).add_(0.001 * p.data)
            # verbose
            if iter_idx % args.print_interval == 0:
                logger.info(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %.5f] [G_orign loss: %.5f] || with no info loss." %
                    (epoch, shared_epochs, iter_idx, batch_idex, err_dr, loss_g.item()))

            # check window
            if g_loss_history.is_full():
                if g_loss_history.get_var() < args.dynamic_reset_threshold \
                        or d_loss_history.get_var() < args.dynamic_reset_threshold:
                    dynamic_reset = True
                    logger.info("=> dynamic resetting triggered")
                    g_loss_history.clear()
                    d_loss_history.clear()
                    return dynamic_reset

            step += 1

    return dynamic_reset


def train_shared_mixed(args, gen_net: nn.Module, dis_net: nn.Module, clhead_big, clhead_small, g_loss_history, d_loss_history, controller, gen_optimizer
                 , dis_optimizer, optimizerCL_big, optimizerCL_small, train_loader, cur_stage=1, top_skip_archs=None):
    dynamic_reset = False
    logger.info('=> train shared GAN with different datasets...')
    step = 0
    if cur_stage == 1:
        shared_epochs = args.shared_epoch1
        # latent_dim = args.latent_dim
    else:
        shared_epochs = args.shared_epoch2
        # latent_dim = args.latent_dim // 2

    # train mode
    gen_net.train()
    dis_net.train()
    batch_idex = args.num_imgs // args.batch_size

    # eval mode
    controller.eval()
    for epoch in range(shared_epochs):
        for iter_idx in range(batch_idex):
            imgs = next(train_loader)
            # sample an arch
            arch = controller.sample(1, prev_archs=top_skip_archs)[0][0]
            gen_net.set_arch(arch)
            # Adversarial ground truths

            real_image = imgs.to(device)
            # real_image = F.interpolate(real_image, args.im_size)
            current_batch_size = real_image.size(0)
            noise = torch.Tensor(current_batch_size, args.latent_dim).normal_(0, 1).to(device)

            fake_images = gen_net(noise)

            real_image = DiffAugment(real_image, policy=policy)
            fake_images = [DiffAugment(fake, policy=policy) for fake in fake_images]
            fake_q = [DiffAugment(fake, policy=q_aug) for fake in fake_images]
            fake_k = [DiffAugment(fake, policy=k_aug) for fake in fake_images]

            ## 2. train Discriminator
            dis_net.zero_grad()

            err_dr, _, _, _ = train_d(dis_net, real_image, label="real")
            train_d(dis_net, [fi.detach() for fi in fake_images], label="fake")
            dis_optimizer.step()

            ## 3. train Generator
            gen_net.zero_grad()
            clhead_big.zero_grad()
            clhead_small.zero_grad()

            pred_g = dis_net(fake_images, "fake")
            logits_q_big, logits_q_small = dis_net(fake_q, "fake", mapping=True)
            logits_k_big, logits_k_small = dis_net(fake_k, "fake", mapping=True)

            # print('logits_q_shape:{}, logits_k_shape:{}'.format(logits_q.size(), logits_k.size()))

            # clhead.requires_grad_(False)
            loss_ins_big = clhead_big(logits_q_big, logits_k_big)
            loss_ins_small = clhead_small(logits_q_small, logits_k_small)
            info_loss = loss_ins_big + loss_ins_small
            loss_g = -pred_g.mean()
            err_g = loss_g + args.fake_ins * info_loss

            # err_g = -pred_g.mean()

            err_g.backward()

            optimizerCL_big.step()
            optimizerCL_small.step()
            gen_optimizer.step()
            # clhead_big._momentum_update_key_encoder()
            # clhead_small._momentum_update_key_encoder()

            # for p, avg_p in zip(gen_net.parameters(), avg_param_G):
            #     avg_p.mul_(0.999).add_(0.001 * p.data)
            # verbose
            if iter_idx % args.print_interval == 0:
                logger.info(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %.5f] [G_orign loss: %.5f] [info_loss : %.5f] [G loss: %.5f] || with info loss." %
                    (epoch, shared_epochs, iter_idx, batch_idex, err_dr, loss_g.item(), info_loss.item(), err_g.item()))

            # check window
            if g_loss_history.is_full():
                if g_loss_history.get_var() < args.dynamic_reset_threshold \
                        or d_loss_history.get_var() < args.dynamic_reset_threshold:
                    dynamic_reset = True
                    logger.info("=> dynamic resetting triggered")
                    g_loss_history.clear()
                    d_loss_history.clear()
                    return dynamic_reset

            step += 1

    return dynamic_reset

def train_controller(args, controller, ctrl_optimizer, gen_net, im_nums, data_root, writer_dict, top_skip_archs=None, cur_stage=1):
    logger.info("=> train controller...")
    writer = writer_dict['writer']
    baseline = None
    if cur_stage == 1:
        ctrl_epochs = args.ctrl_step1
    else:
        ctrl_epochs = args.ctrl_step2
    # train mode
    controller.train()

    for step in range(ctrl_epochs):
        controller_step = writer_dict['controller_steps']

        archs, selected_log_probs, entropies = controller.sample(args.ctrl_sample_batch, prev_archs=top_skip_archs)
        cur_batch_rewards = []
        for arch in archs:
            logger.info(f'arch: {arch}')
            gen_net.set_arch(arch)
            # is_score = get_is(args, gen_net, args.rl_num_eval_img)
            # logger.info(f'get Inception score of {is_score}')
            is_score, fid = get_reward(args, gen_net, data_root, im_nums)
            fid /= 100.0
            # fid_weight = args.fid_weight
            # reward = is_score - fid_weight * fid
            reward = 5.0 * np.exp(-fid)
            logger.info(f'get Inception score of {is_score}, FID of {fid*100}， reward of {reward}')
            cur_batch_rewards.append(reward)
        cur_batch_rewards = torch.tensor(cur_batch_rewards, requires_grad=False).cuda()
        cur_batch_rewards = cur_batch_rewards.unsqueeze(-1) + args.entropy_coeff * entropies  # bs * 1
        if baseline is None:
            baseline = cur_batch_rewards
        else:
            baseline = args.baseline_decay * baseline.detach() + (1 - args.baseline_decay) * cur_batch_rewards
        adv = cur_batch_rewards - baseline

        # policy loss
        loss = -selected_log_probs * adv #每个reward加上关于所有可能操作的entropy * action的概率
        loss = loss.sum()

        # update controller
        ctrl_optimizer.zero_grad()
        loss.backward()
        ctrl_optimizer.step()

        logger.info(
            "[Step %d/%d] [ctrl loss: %f]" %
            (step, ctrl_epochs, loss.item()))

        # write
        mean_reward = cur_batch_rewards.mean().item()
        mean_adv = adv.mean().item()
        mean_entropy = entropies.mean().item()
        writer.add_scalar('controller/loss', loss.item(), controller_step)
        writer.add_scalar('controller/reward', mean_reward, controller_step)
        writer.add_scalar('controller/entropy', mean_entropy, controller_step)
        writer.add_scalar('controller/adv', mean_adv, controller_step)

        writer_dict['controller_steps'] = controller_step + 1


# def get_is(args, gen_net: nn.Module, num_img):
#     """
#     Get inception score.
#     :param args:
#     :param gen_net:
#     :param num_img:
#     :return: Inception score
#     """
#
#     # eval mode
#     gen_net = gen_net.eval()
#
#     eval_iter = num_img // args.eval_batch_size
#     img_list = list()
#     with torch.no_grad():
#         for _ in range(eval_iter):
#             # z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim)))
#             #
#             # # Generate a batch of images
#             # gen_imgs = gen_net(z)[0].mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu',
#             #                                                                                         torch.uint8).numpy()
#             # img_list.extend(list(gen_imgs))
#             z = torch.randn(args.eval_batch_size, args.latent_dim).to(device)
#
#             # Generate a batch of images
#             gen_imgs = gen_net(z)[0].mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
#             # for j, img in enumerate(gen_imgs):
#             #     file_name = os.path.join(dist, f'{i * args.batch + j}.png')
#             #     imsave(file_name, img)
#             img_list.extend(list(gen_imgs))
#
#     # get inception score
#     logger.info('calculate Inception score...')
#     mean, std = get_inception_score(img_list)
#
#     return mean

def get_reward(args, gen_net: nn.Module, data_root, im_nums=5000, clean_dir=True):
    """
    Get inception score.
    :param args:
    :param gen_net:
    :param num_img:
    :return: Inception score
    """

    # eval mode
    gen_net = gen_net.eval()

    # get fid and inception score
    dist = args.path_helper['sample_path']
    # dist = os.path.join(dist, 'img')
    os.makedirs(dist, exist_ok=True)

    # origin_data = './fid_buffer/origin_data_buffer'
    #
    # img = Image.open(file).convert('RGB')
    #
    # if self.transform:
    #     img = self.transform(img)

    # img_list = list()
    with torch.no_grad():
        for i in tqdm(range(im_nums // args.batch)):
            z = torch.randn(args.batch, args.latent_dim).to(device)

            # Generate a batch of images
            gen_imgs = gen_net(z)[0]
            for j, g_img in enumerate(gen_imgs):
                vutils.save_image(g_img.add(1).mul(0.5),
                                  os.path.join(dist,
                                               '%d.png' % (i * args.batch + j)))  # , normalize=True, range=(-1,1))

    # # get inception score
    # logger.info('=> calculate inception score')
    # mean, std = get_inception_score(img_list)
    # # print(f"Inception score: {mean}")
    #
    # # get fid score
    # logger.info('=> calculate fid score')
    # fid_stat = './data/few-shot-images/AnimalFace-dog/img/'
    # fid_score = calculate_fid_given_paths([dist, fid_stat], im_nums, inception_path=None)

    metrics_dict = torch_fidelity.calculate_metrics(
        input1=dist,
        input2=data_root,
        cuda=True,
        isc=True,
        fid=True,
        verbose=False,
    )
    # if clean_dir:
    #     os.system('rm -f {}*'.format(dist))

    return metrics_dict['inception_score_mean'],metrics_dict['frechet_inception_distance']

def get_reward_train(args, gen_net: nn.Module, data_root, im_nums=5000, clean_dir=True):
    """
    Get inception score.
    :param args:
    :param gen_net:
    :param num_img:
    :return: Inception score
    """

    # eval mode
    gen_net = gen_net.eval()

    # get fid and inception score
    dist = args.path_helper['sample_path']
    dist_is = os.path.join(dist, 'img_is')
    dist_fid = os.path.join(dist, 'img_fid')
    os.makedirs(dist_is, exist_ok=True)
    os.makedirs(dist_fid, exist_ok=True)

    # origin_data = './fid_buffer/origin_data_buffer'
    #
    # img = Image.open(file).convert('RGB')
    #
    # if self.transform:
    #     img = self.transform(img)

    # img_list = list()
    with torch.no_grad():
        for i in tqdm(range(im_nums // args.batch)):
            z = torch.randn(args.batch, args.latent_dim).to(device)

            # Generate a batch of images
            gen_imgs = gen_net(z)[0]
            for j, g_img in enumerate(gen_imgs):
                idex = i * args.batch + j
                vutils.save_image(g_img.add(1).mul(0.5),
                                  os.path.join(dist_is,
                                               '%d.png' % (idex)))  # , normalize=True, range=(-1,1))
                if idex < 5000:
                    vutils.save_image(g_img.add(1).mul(0.5),
                                      os.path.join(dist_fid,
                                                   '%d.png' % (idex)))  # , normalize=True, range=(-1,1))

    # # get inception score
    # logger.info('=> calculate inception score')
    # mean, std = get_inception_score(img_list)
    # # print(f"Inception score: {mean}")
    #
    # # get fid score
    # logger.info('=> calculate fid score')
    # fid_stat = './data/few-shot-images/AnimalFace-dog/img/'
    # fid_score = calculate_fid_given_paths([dist, fid_stat], im_nums, inception_path=None)

    metrics_dict_is = torch_fidelity.calculate_metrics(
        input1=dist_is,
        input2=data_root,
        cuda=True,
        isc=True,
        verbose=False,
    )
    metrics_dict_fid = torch_fidelity.calculate_metrics(
        input1=dist_fid,
        input2=data_root,
        cuda=True,
        fid=True,
        verbose=False,
    )
    if clean_dir:
        os.system('rm -r {}'.format(dist))

    return metrics_dict_is['inception_score_mean'],metrics_dict_fid['frechet_inception_distance']

# def validate(args, fixed_z, fid_stat, gen_net: nn.Module, writer_dict, clean_dir=True):
#     writer = writer_dict['writer']
#     global_steps = writer_dict['valid_global_steps']
#
#     # eval mode
#     gen_net = gen_net.eval()
#
#     # generate images
#     sample_imgs = gen_net(fixed_z)
#     img_grid = make_grid(sample_imgs, nrow=5, normalize=True, scale_each=True)
#
#     # get fid and inception score
#     fid_buffer_dir = os.path.join(args.path_helper['sample_path'], 'fid_buffer')
#     os.makedirs(fid_buffer_dir, exist_ok=True)
#
#     eval_iter = args.num_eval_imgs // args.eval_batch_size
#     img_list = list()
#     for iter_idx in tqdm(range(eval_iter), desc='sample images'):
#         z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim)))
#
#         # Generate a batch of images
#         gen_imgs = gen_net(z).mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu',
#                                                                                                 torch.uint8).numpy()
#         for img_idx, img in enumerate(gen_imgs):
#             file_name = os.path.join(fid_buffer_dir, f'iter{iter_idx}_b{img_idx}.png')
#             imsave(file_name, img)
#         img_list.extend(list(gen_imgs))
#
#     # get inception score
#     logger.info('=> calculate inception score')
#     mean, std = get_inception_score(img_list)
#     print(f"Inception score: {mean}")
#
#     # get fid score
#     logger.info('=> calculate fid score')
#     fid_score = calculate_fid_given_paths([fid_buffer_dir, fid_stat], inception_path=None)
#     print(f"FID score: {fid_score}")
#
#     if clean_dir:
#         os.system('rm -r {}'.format(fid_buffer_dir))
#     else:
#         logger.info(f'=> sampled images are saved to {fid_buffer_dir}')
#
#     writer.add_image('sampled_images', img_grid, global_steps)
#     writer.add_scalar('Inception_score/mean', mean, global_steps)
#     writer.add_scalar('Inception_score/std', std, global_steps)
#     writer.add_scalar('FID_score', fid_score, global_steps)
#
#     writer_dict['valid_global_steps'] = global_steps + 1
#
#     return mean, fid_score


def get_topk_arch_hidden(args, controller, gen_net, data_root, top_skip_archs=None):
    """
    ~
    :param args:
    :param controller:
    :param gen_net:
    :param prev_archs: previous architecture
    :param prev_hiddens: previous hidden vector
    :return: a list of topk archs and hiddens.
    """
    logger.info(f'=> get top{args.topk} archs out of {args.num_candidate} candidate archs...')
    assert args.num_candidate >= args.topk
    controller.eval()
    with torch.no_grad():

        archs, _, _, = controller.sample(args.num_candidate, prev_archs=top_skip_archs)
        # hxs, cxs = hiddens
        arch_idx_perf_table_fid = {}
        arch_idx_perf_table_is = {}
        for arch_idx in range(len(archs)):
            logger.info(f'arch: {archs[arch_idx]}')
            gen_net.set_arch(archs[arch_idx])
            is_score,fid = get_reward_train(args, gen_net, data_root, args.get_top_eval_img)
            logger.info(f'get Inception score of {is_score}, fid score of {fid}')
            arch_idx_perf_table_fid[arch_idx] = fid
            arch_idx_perf_table_is[arch_idx] = is_score

        topk_arch_idx_perf = sorted(arch_idx_perf_table_fid.items(), key=operator.itemgetter(1), reverse=True)[::-1][:args.topk]
        topk_archs_fid = []
        logger.info(f'top{args.topk} archs by fid:')
        for arch_idx_perf in topk_arch_idx_perf:
            logger.info(arch_idx_perf)
            arch_idx = arch_idx_perf[0]
            topk_archs_fid.append(archs[arch_idx])

        # topk_arch_idx_perf = sorted(arch_idx_perf_table_is.items(), key=operator.itemgetter(1))[::-1][:args.topk]
        # topk_archs_is = []
        # logger.info(f'top{args.topk} archs by IS:')
        # for arch_idx_perf in topk_arch_idx_perf:
        #     logger.info(arch_idx_perf)
        #     arch_idx = arch_idx_perf[0]
        #     topk_archs_is.append(archs[arch_idx])

    return topk_archs_fid


class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten
