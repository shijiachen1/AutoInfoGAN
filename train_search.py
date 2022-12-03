import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils
import torch_fidelity

import cfg
import random
from tqdm import tqdm
import os
from copy import deepcopy

from utils.utils import set_log_dir, save_checkpoint, create_logger, RunningStats
from models_search.shared_gan_2stage import Generator,Discriminator
from models import weights_init, CLHead
from operation import copy_G_params, load_params, get_dir
from operation import ImageFolder, InfiniteSamplerWrapper
from diffaug import DiffAugment
from functions import get_reward_train

policy = 'color,translation'
q_aug = 'color,cutout'
k_aug = 'color,translation'
import lpips
percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)


#torch.backends.cudnn.benchmark = True


def crop_image_by_part(image, part):
    hw = image.shape[2]//2
    if part==0:
        return image[:,:,:hw,:hw]
    if part==1:
        return image[:,:,:hw,hw:]
    if part==2:
        return image[:,:,hw:,:hw]
    if part==3:
        return image[:,:,hw:,hw:]

def train_d(net, data, label="real"):
    """Train function of discriminator"""
    if label=="real":
        part = random.randint(0, 3)
        pred, [rec_all, rec_small, rec_part] = net(data, label, part=part)
        err = F.relu(  torch.rand_like(pred) * 0.2 + 0.8 -  pred).mean() + \
            percept( rec_all, F.interpolate(data, rec_all.shape[2]) ).sum() +\
            percept( rec_small, F.interpolate(data, rec_small.shape[2]) ).sum() +\
            percept( rec_part, F.interpolate(crop_image_by_part(data, part), rec_part.shape[2]) ).sum()
        err.backward()
        return pred.mean().item(), rec_all, rec_small, rec_part
    else:
        pred = net(data, label)
        err = F.relu( torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
        err.backward()
        return pred.mean().item()
        
def get_reward(args, gen_net: nn.Module, data_root, im_nums=5000, clean_dir=True):
    """
    Get inception score.
    :param args:
    :param gen_net:
    :param num_img:
    :return: Inception score
    """

    # eval mode
    # gen_net = gen_net.eval()

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
            z = torch.randn(args.batch, 256).cuda()

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
    # if clean_dir:
    os.system('rm -r {}'.format('./' + dist_is + '/'))

    return metrics_dict_is['inception_score_mean'],metrics_dict_fid['frechet_inception_distance']


def train(args):

    data_root = args.path
    total_iterations = args.iter
    batch_size = args.batch_size
    im_size = args.im_size
    ndf = 64
    ngf = 64
    nz = 256
    nlr = 0.0002
    nbeta1 = 0.5
    best_fid = 1e4
    use_cuda = True
    multi_gpu = False
    dataloader_workers = 8
    start_epoch = args.start_iter
    save_interval = 100

    device = torch.device("cpu")
    if use_cuda:
        device = torch.device("cuda:0")


    transform_list = [
            transforms.Resize((int(im_size),int(im_size))),
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

   
    dataloader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      sampler=InfiniteSamplerWrapper(dataset), num_workers=dataloader_workers, pin_memory=True))
    '''
    loader = MultiEpochsDataLoader(dataset, batch_size=batch_size, 
                               shuffle=True, num_workers=dataloader_workers, 
                               pin_memory=True)
    dataloader = CudaDataLoader(loader, 'cuda')
    '''

    transform_list_dist = [
        transforms.Resize((int(args.im_size), int(args.im_size))),
        transforms.ToTensor(),
    ]
    trans_a = transforms.Compose(transform_list_dist)
    dataset_a = ImageFolder(root=data_root, transform=trans_a)
    loader = DataLoader(dataset_a, batch_size=64, num_workers=8)
    dist = './fid_buffer/train/'
    os.system('rm -f {}*'.format(dist))

    for i, imgs in enumerate(loader):
        for j, img in enumerate(imgs):
            vutils.save_image(img,
                              os.path.join(dist,
                                           '%d.png' % (i * 64 + j)))  # , normalize=True, range=(-1,1))
    
    #from model_s import Generator, Discriminator
    netG = Generator(ngf=ngf, nz=nz, im_size=im_size,cur_stage=2)
    netG.set_arch(args.arch)
    netG.apply(weights_init)

    netD = Discriminator(ndf=ndf, im_size=im_size)
    netD.apply(weights_init)

    clhead_big = CLHead(inplanes=ndf * 8)
    clhead_big.apply(weights_init)
    clhead_small = CLHead(inplanes=ndf * 4)
    clhead_small.apply(weights_init)

    clhead_big.cuda()
    clhead_small.cuda()
    netG.cuda()
    netD.cuda()

    fixed_noise = torch.FloatTensor(8, nz).normal_(0, 1).cuda()
    avg_param_G = copy_G_params(netG)

    optimizerG = optim.Adam(netG.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    optimizerCL_big = optim.Adam(clhead_big.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    optimizerCL_small = optim.Adam(clhead_small.parameters(), lr=nlr, betas=(nbeta1, 0.999))

    # set writer
    if args.load_path:
        print(f'=> resuming from {args.load_path}')
        assert os.path.exists(args.load_path)
        checkpoint_file = os.path.join(args.load_path, 'Model', 'all_20000.pth')
        assert os.path.exists(checkpoint_file)
        checkpoint = torch.load(checkpoint_file)
        start_epoch = checkpoint['epoch']
        # best_fid = checkpoint['best_fid']
        netG.load_state_dict(checkpoint['gen_state_dict'])
        netD.load_state_dict(checkpoint['dis_state_dict'])
        optimizerG.load_state_dict(checkpoint['gen_optimizer'])
        optimizerD.load_state_dict(checkpoint['dis_optimizer'])
        clhead_big.load_state_dict(checkpoint['clhead_big'])
        clhead_small.load_state_dict(checkpoint['clhead_samll'])
        optimizerCL_big.load_state_dict(checkpoint['cloptimizer_big'])
        optimizerCL_small.load_state_dict(checkpoint['cloptimizer_small'])
        # avg_gen_net = deepcopy(netG)
        # avg_gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
        # gen_avg_param = copy_G_params(avg_gen_net)
        # del avg_gen_net
        avg_param_G = checkpoint['g_ema']

        args.path_helper = checkpoint['path_helper']
        logger = create_logger(args.path_helper['log_path'])
        logger.info(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')
    else:
        # create new log dir
        assert args.exp_name
        args.path_helper = set_log_dir('logs', args.exp_name)
        logger = create_logger(args.path_helper['log_path'])
        image_path = os.path.join(args.path_helper['prefix'], 'Images')
        os.makedirs(image_path)
        args.path_helper['images_path'] = image_path

    saved_image_folder = args.path_helper['images_path']
    saved_model_folder = args.path_helper['ckpt_path']
    logger.info(args)

    # if checkpoint != 'None':
    #     ckpt = torch.load(checkpoint)
    #     netG.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['g'].items()})
    #     netD.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['d'].items()})
    #     avg_param_G = ckpt['g_ema']
    #     optimizerG.load_state_dict(ckpt['opt_g'])
    #     optimizerD.load_state_dict(ckpt['opt_d'])
    #     current_iteration = int(checkpoint.split('_')[-1].split('.')[0])
    #     del ckpt

    if multi_gpu:
        netG = nn.DataParallel(netG.to(device))
        netD = nn.DataParallel(netD.to(device))

    # for iteration in tqdm(range(current_iteration, total_iterations + 1)):
    #     for iter_idx,imgs in enumerate(tqdm(dataloader)):
    #         real_image = imgs
    #         real_image = real_image.to(device)
    #         current_batch_size = real_image.size(0)
    #         noise = torch.Tensor(current_batch_size, nz).normal_(0, 1).to(device)
    #
    #         fake_images = netG(noise)
    #
    #         real_image = DiffAugment(real_image, policy=policy)
    #         fake_images = [DiffAugment(fake, policy=policy) for fake in fake_images]
    #
    #         ## 2. train Discriminator
    #         netD.zero_grad()
    #
    #         err_dr, rec_img_all, rec_img_small, rec_img_part = train_d(netD, real_image, label="real")
    #         train_d(netD, [fi.detach() for fi in fake_images], label="fake")
    #         optimizerD.step()
    #
    #         ## 3. train Generator
    #         netG.zero_grad()
    #         pred_g = netD(fake_images, "fake")
    #         err_g = -pred_g.mean()
    #
    #         err_g.backward()
    #         optimizerG.step()
    #
    #         for p, avg_p in zip(netG.parameters(), avg_param_G):
    #             avg_p.mul_(0.999).add_(0.001 * p.data)
    #
    #         if iter_idx % 50 == 0:
    #             print("%d batch || %d epoch  GAN: loss d: %.5f    loss g: %.5f" % (iter_idx, iteration, err_dr, -err_g.item()))
    #
    #         if iteration % (save_interval * 10) == 0:
    #             backup_para = copy_G_params(netG)
    #             load_params(netG, avg_param_G)
    #             with torch.no_grad():
    #                 vutils.save_image(netG(fixed_noise)[0].add(1).mul(0.5), saved_image_folder + '/%d.jpg' % iteration,
    #                                   nrow=4)
    #                 vutils.save_image(torch.cat([
    #                     F.interpolate(real_image, 128),
    #                     rec_img_all, rec_img_small,
    #                     rec_img_part]).add(1).mul(0.5), saved_image_folder + '/rec_%d.jpg' % iteration)
    #             load_params(netG, backup_para)
    #
    #         if iteration % (save_interval * 50) == 0 or iteration == total_iterations:
    #             backup_para = copy_G_params(netG)
    #             load_params(netG, avg_param_G)
    #             torch.save({'g': netG.state_dict(), 'd': netD.state_dict()}, saved_model_folder + '/%d.pth' % iteration)
    #             load_params(netG, backup_para)
    #             torch.save({'g': netG.state_dict(),
    #                         'd': netD.state_dict(),
    #                         'g_ema': avg_param_G,
    #                         'opt_g': optimizerG.state_dict(),
    #                         'opt_d': optimizerD.state_dict()}, saved_model_folder + '/all_%d.pth' % iteration)

    # print(len(dataloader))
    for iteration in tqdm(range(start_epoch, total_iterations + 1)):

        real_image = next(dataloader)
        real_image = real_image.cuda()
        current_batch_size = real_image.size(0)
        noise = torch.Tensor(current_batch_size, nz).normal_(0, 1).cuda()

        fake_images = netG(noise)

        real_image = DiffAugment(real_image, policy=policy)
        fake_images = [DiffAugment(fake, policy=policy) for fake in fake_images]
        fake_q = [DiffAugment(fake, policy=q_aug) for fake in fake_images]
        fake_k = [DiffAugment(fake, policy=k_aug) for fake in fake_images]

        ## 2. train Discriminator
        netD.zero_grad()

        err_dr, rec_img_all, rec_img_small, rec_img_part = train_d(netD, real_image, label="real")
        train_d(netD, [fi.detach() for fi in fake_images], label="fake")
        optimizerD.step()

        ## 3. train Generator
        netG.zero_grad()
        clhead_big.zero_grad()
        clhead_small.zero_grad()

        pred_g = netD(fake_images, "fake")
        logits_q_big, logits_q_small = netD(fake_q, "fake", mapping=True)
        logits_k_big, logits_k_small = netD(fake_k, "fake", mapping=True)

        loss_ins_big = clhead_big(logits_q_big, logits_k_big)
        loss_ins_small = clhead_small(logits_q_small, logits_k_small)
        info_loss = loss_ins_big + loss_ins_small
        loss_g = -pred_g.mean()
        err_g = loss_g + args.fake_ins * info_loss

        # err_g = -pred_g.mean()

        err_g.backward()

        optimizerCL_big.step()
        optimizerCL_small.step()
        optimizerG.step()

        # for p, avg_p in zip(netG.parameters(), gen_avg_param):
        #     avg_p.mul_(0.999).add_(0.001 * p.data)
        #
        # if iteration > 0 and (iteration % (save_interval*100) == 0 or iteration == total_iterations):
        #     backup_param = copy_G_params(netG)
        #     load_params(netG, gen_avg_param)
        #     logger.info('=> calculate fid score')
        #     inception_score, fid_score = get_fid(args, netG, im_nums=5000)
        #     logger.info(f'Inception score: {inception_score}, FID score: {fid_score} || @ epoch {iteration}.')
        #     load_params(netG, backup_param)
        #
        #     if fid_score < best_fid:
        #         best_fid = fid_score
        #         is_best = True
        #     else:
        #         is_best = False
        # else:
        #     is_best = False
        #
        # avg_gen_net = deepcopy(netG)
        # load_params(avg_gen_net, gen_avg_param)
        # save_checkpoint({
        #     'epoch': iteration + 1,
        #     'gen_model': args.gen_model,
        #     'dis_model': args.dis_model,
        #     'gen_state_dict': netG.state_dict(),
        #     'dis_state_dict': netD.state_dict(),
        #     'avg_gen_state_dict': avg_gen_net.state_dict(),
        #     'gen_optimizer': optimizerG.state_dict(),
        #     'dis_optimizer': optimizerD.state_dict(),
        #     'best_fid': best_fid,
        #     'path_helper': args.path_helper
        # }, is_best, args.path_helper['ckpt_path'])
        # del avg_gen_net

        for p, avg_p in zip(netG.parameters(), avg_param_G):
            avg_p.mul_(0.999).add_(0.001 * p.data)

        if iteration % 100 == 0:
            logger.info("loss d: %.5f  loss_orign: %.5f  info_loss : %.5f || loss g: %.5f   epoch: %d" % (
            err_dr, loss_g.item(), info_loss.item(),
            err_g.item(), iteration))

        if iteration % (save_interval * 10) == 0:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            with torch.no_grad():
                vutils.save_image(netG(fixed_noise)[0].add(1).mul(0.5), saved_image_folder + '/%d.jpg' % iteration,
                                  nrow=4)
                vutils.save_image(torch.cat([
                    F.interpolate(real_image, 128),
                    rec_img_all, rec_img_small,
                    rec_img_part]).add(1).mul(0.5), saved_image_folder + '/rec_%d.jpg' % iteration)
            load_params(netG, backup_para)

        if iteration and (iteration % (save_interval * 100) == 0 or iteration == total_iterations):
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            inception_score, fid = get_reward(args, netG, dist, 5000)

            logger.info("inception score is : %.5f  fid score is : %.5f" % (inception_score, fid))
            if fid < best_fid:
                # backup_para = copy_G_params(netG)
                # load_params(netG, avg_param_G)
                torch.save({'g': netG.state_dict(), 'd': netD.state_dict()}, saved_model_folder + '/check.pth')
                load_params(netG, backup_para)
                torch.save({'epoch': iteration + 1,
                            'gen_state_dict': netG.state_dict(),
                            'dis_state_dict': netD.state_dict(),
                            'g_ema': avg_param_G,
                            'clhead_big': clhead_big.state_dict(),
                            'clhead_samll': clhead_small.state_dict(),
                            'gen_optimizer': optimizerG.state_dict(),
                            'dis_optimizer': optimizerD.state_dict(),
                            'cloptimizer_big': optimizerCL_big.state_dict(),
                            'cloptimizer_small': optimizerCL_small.state_dict(),
                            'path_helper': args.path_helper}, saved_model_folder + '/all_check.pth')
                best_fid = fid
            load_params(netG, backup_para)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='region gan')
    #
    # parser.add_argument('--path', type=str, default='../lmdbs/art_landscape_1k', help='path of resource dataset, should be a folder that has one or many sub image folders inside')
    # parser.add_argument('--cuda', type=int, default=0, help='index of gpu to use')
    # parser.add_argument('--name', type=str, default='skull', help='experiment name')
    # parser.add_argument('--iter', type=int, default=50000, help='number of iterations')
    # parser.add_argument('--start_iter', type=int, default=0, help='the iteration to start training')
    # parser.add_argument('--batch_size', type=int, default=8, help='mini batch number of images')
    # parser.add_argument('--im_size', type=int, default=512, help='image resolution')
    # parser.add_argument('--ckpt', type=str, default='None', help='checkpoint weight path if have one')
    # parser.add_argument('--fake_ins', type=float, default=0.1, help='hyperparameters of ins loss')

    args = cfg.parse_args()
    print(args)

    train(args)
