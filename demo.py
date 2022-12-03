import pathlib
import numpy as np
import torch
from imageio import imread
import torch.nn.functional as F
from random import randint
import torch_fidelity
from models_search.shared_gan import Generator
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from torchvision import utils as vutils
import os
# def random_crop(image, size):
#     h, w = image.shape[:2]
#     print(h,w)
#     ch = randint(0, h - size )
#     cw = randint(0, w - size )
#     return image[ch:ch + size, cw:cw + size, :]
#
# path = './fid_buffer/animalface-dog-auto_eval/img/'
# path = pathlib.Path(path)
# files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
# # x = np.array([imread(str(fn)).astype(np.float32) for fn in files[:1]])
# files = files[:10]
# x = np.array( [random_crop(imread(str(fn)).astype(np.float32), 256) for fn in files])
# #
# print("img.shape:{}".format(x.shape))
# # net_ig = Generator( ngf=64, nz=256, nc=3, im_size=256).cuda() #, big=args.big )
# # net_ig.set_arch([0,0,1,0,2,2])

# transform_list = [
#     transforms.Resize((256, 256)),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ]
# trans = transforms.Compose(transform_list)
# from operation import ImageFolder, InfiniteSamplerWrapper
# data_root = './data/few-shot-images/flat-colored/patterns'
#
# transform = transforms.Compose(
#     [
#         transforms.Resize((256,256)),
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
#     ]
# )
#
# dset = ImageFolder(data_root, transform)
# loader = DataLoader(dset, batch_size=64, num_workers=4)
# dist = './eval_imgs/origin/'
# # wrapped_generator = torch_fidelity.GenerativeModelModuleWrapper(net_ig, 256, 'normal', 0)
# #
# parb = tqdm(loader)
# for imgs,i in parb:
#     for j,img in enumerate(imgs):
#         vutils.save_image(img,
#                           os.path.join(dist,
#                                        '%d.png' % (i * 64 + j)))  # , normalize=True, range=(-1,1))
#
# metrics_dict = torch_fidelity.calculate_metrics(
#     input2=data_root,
#     input1=dist,
#     cuda=True,
#     isc=True,
#     fid=True,
#     verbose=False,
# )
#
# print(metrics_dict)
# dist = './fid_buffer/origin/'
#
# os.system('rm -f {}*'.format(dist))
a = [1]
a+= [2,3,4]
print(a)