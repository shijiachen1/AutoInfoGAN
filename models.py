import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

import random

seq = nn.Sequential

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            pass
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))

def convTranspose2d(*args, **kwargs):
    return spectral_norm(nn.ConvTranspose2d(*args, **kwargs))

def batchNorm2d(*args, **kwargs):
    return nn.BatchNorm2d(*args, **kwargs)

def linear(*args, **kwargs):
    return spectral_norm(nn.Linear(*args, **kwargs))

class PixelNorm(nn.Module):
    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.target_shape = shape

    def forward(self, feat):
        batch = feat.shape[0]
        return feat.view(batch, *self.target_shape)        


class GLU(nn.Module):
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, feat, noise=None):
        if noise is None:
            batch, _, height, width = feat.shape
            noise = torch.randn(batch, 1, height, width).to(feat.device)

        return feat + self.weight * noise


class Swish(nn.Module):
    def forward(self, feat):
        return feat * torch.sigmoid(feat)


class SEBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.main = nn.Sequential(  nn.AdaptiveAvgPool2d(4), 
                                    conv2d(ch_in, ch_out, 4, 1, 0, bias=False), Swish(),
                                    conv2d(ch_out, ch_out, 1, 1, 0, bias=False), nn.Sigmoid() )

    def forward(self, feat_small, feat_big):
        return feat_big * self.main(feat_small)


class InitLayer(nn.Module):
    def __init__(self, nz, channel):
        super().__init__()

        self.init = nn.Sequential(
                        convTranspose2d(nz, channel*2, 4, 1, 0, bias=False),
                        batchNorm2d(channel*2), GLU() )

    def forward(self, noise):
        noise = noise.view(noise.shape[0], -1, 1, 1)
        return self.init(noise)


def UpBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
        #convTranspose2d(in_planes, out_planes*2, 4, 2, 1, bias=False),
        batchNorm2d(out_planes*2), GLU())
    return block


def UpBlockComp(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
        #convTranspose2d(in_planes, out_planes*2, 4, 2, 1, bias=False),
        NoiseInjection(),
        batchNorm2d(out_planes*2), GLU(),
        conv2d(out_planes, out_planes*2, 3, 1, 1, bias=False),
        NoiseInjection(),
        batchNorm2d(out_planes*2), GLU()
        )
    return block


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # tensors_gather = [torch.ones_like(tensor)
    #     for _ in range(torch.distributed.get_world_size())]
    # torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    tensors_gather = [torch.ones_like(tensor).requires_grad_(False),tensor]

    output = torch.cat(tensors_gather, dim=0)
    return output

class CLHead(torch.nn.Module):
    def __init__(self,
        inplanes     = 64*8,   # Number of input features
        temperature  = 0.2,   # Temperature of logits
        queue_size   = 3500,  # Number of stored negative samples
        momentum     = 0.999, # Momentum for updating network
    ):
        super().__init__()
        self.inplanes = inplanes
        self.temperature = temperature
        self.queue_size = queue_size
        self.m = momentum

        self.mlp = nn.Sequential(linear(inplanes, inplanes), nn.ReLU(), linear(inplanes, 128))
        self.momentum_mlp = nn.Sequential(linear(inplanes, inplanes), nn.ReLU(), linear(inplanes, 128))
        self.momentum_mlp.requires_grad_(False)

        for param_q, param_k in zip(self.mlp.parameters(), self.momentum_mlp.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(128, self.queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.mlp.parameters(), self.momentum_mlp.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]
        keys = keys.T
        ptr = int(self.queue_ptr)
        if batch_size > self.queue_size:
            self.queue[:, 0:] = keys[:, :self.queue_size]

        elif ptr + batch_size > self.queue_size:
            self.queue[:, ptr:] = keys[:, :self.queue_size - ptr]
            self.queue[:, :batch_size - (self.queue_size - ptr)] = keys[:, self.queue_size-ptr:]
            self.queue_ptr[0] = batch_size - (self.queue_size - ptr)
        else:
            self.queue[:, ptr:ptr + batch_size] = keys
            self.queue_ptr[0] = ptr + batch_size

    # @torch.no_grad()
    # def _batch_shuffle_ddp(self, x):
    #     """
    #     Batch shuffle, for making use of BatchNorm.
    #     """
    #
    #     # If non-distributed now, return raw input directly.
    #     # We have no idea the effect of disabling shuffle BN to MoCo.
    #     # Thus, we recommand train InsGen with more than 1 GPU always.
    #     if not torch.distributed.is_initialized():
    #         return x, torch.arange(x.shape[0])
    #
    #     # gather from all gpus
    #     device = x.device
    #     batch_size_this = x.shape[0]
    #     x_gather = concat_all_gather(x)
    #     batch_size_all = x_gather.shape[0]
    #
    #     num_gpus = batch_size_all // batch_size_this
    #
    #     # random shuffle index
    #     idx_shuffle = torch.randperm(batch_size_all).cuda(device)
    #
    #     # broadcast to all gpus
    #     torch.distributed.broadcast(idx_shuffle, src=0)
    #
    #     # index for restoring
    #     idx_unshuffle = torch.argsort(idx_shuffle)
    #
    #     # shuffled index for this gpu
    #     gpu_idx = torch.distributed.get_rank()
    #     idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]
    #
    #     return x_gather[idx_this], idx_unshuffle
    #
    # @torch.no_grad()
    # def _batch_unshuffle_ddp(self, x, idx_unshuffle):
    #     """
    #     Undo batch shuffle.
    #     """
    #     # If non-distributed now, return raw input directly.
    #     # We have no idea the effect of disabling shuffle BN to MoCo.
    #     # Thus, we recommand train InsGen with more than 1 GPU always.
    #     if not torch.distributed.is_initialized():
    #         return x
    #
    #     # gather from all gpus
    #     batch_size_this = x.shape[0]
    #     x_gather = concat_all_gather(x)
    #     batch_size_all = x_gather.shape[0]
    #
    #     num_gpus = batch_size_all // batch_size_this
    #
    #     # restored index for this gpu
    #     gpu_idx = torch.distributed.get_rank()
    #     idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]
    #
    #     return x_gather[idx_this]


    def forward(self, im_q, im_k, update_q=False):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        device = im_q.device
        im_q = im_q.to(torch.float32)
        im_k = im_k.to(torch.float32)
        # compute query features
        if im_q.ndim > 2:
            im_q = im_q.mean([2,3])
        q = self.mlp(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            if im_k.ndim > 2:
                im_k = im_k.mean([2,3])
            # shuffle for making use of BN
            # im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            k = self.momentum_mlp(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)
            # # undo shuffle
            # k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', q, k).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', q, self.queue.clone().detach())

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

        # dequeue and enqueue
        # if not loss_only:
        # if update_q:
        #     with torch.no_grad():
        #         # temp_im_q, idx_unshuffle = self._batch_shuffle_ddp(im_q)
        #         temp_q = self.momentum_mlp(im_q)
        #         temp_q = nn.functional.normalize(temp_q, dim=1)
        #         # temp_q = self._batch_unshuffle_ddp(temp_q, idx_unshuffle)
        #         self._dequeue_and_enqueue(temp_q)
        # else:
        self._dequeue_and_enqueue(k)

        # calculate loss
        loss = nn.functional.cross_entropy(logits, labels)

        return loss

class Generator(nn.Module):
    def __init__(self, ngf=64, nz=100, nc=3, im_size=1024):
        super(Generator, self).__init__()

        nfc_multi = {4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ngf)

        self.im_size = im_size

        self.init = InitLayer(nz, channel=nfc[4])
                                
        self.feat_8   = UpBlockComp(nfc[4], nfc[8])
        self.feat_16  = UpBlock(nfc[8], nfc[16])
        self.feat_32  = UpBlockComp(nfc[16], nfc[32])
        self.feat_64  = UpBlock(nfc[32], nfc[64])
        self.feat_128 = UpBlockComp(nfc[64], nfc[128])  
        self.feat_256 = UpBlock(nfc[128], nfc[256]) 

        self.se_64  = SEBlock(nfc[4], nfc[64])
        self.se_128 = SEBlock(nfc[8], nfc[128])
        self.se_256 = SEBlock(nfc[16], nfc[256])

        self.se_4_128 = SEBlock(nfc[4], nfc[128])
        self.se_8_256 = SEBlock(nfc[8], nfc[256])
        self.se_16_512 = SEBlock(nfc[16], nfc[512])
        self.se_32_1024 = SEBlock(nfc[32], nfc[1024])

        self.se_4_16 = SEBlock(nfc[4], nfc[16])
        self.se_8_32 = SEBlock(nfc[8], nfc[32])
        self.se_16_64 = SEBlock(nfc[16], nfc[64])
        self.se_32_128 = SEBlock(nfc[32], nfc[128])
        self.se_64_256 = SEBlock(nfc[64], nfc[256])
        self.se_128_512 = SEBlock(nfc[128], nfc[512])
        self.se_256_1024 = SEBlock(nfc[256], nfc[1024])
        self.se_4_32 = SEBlock(nfc[4], nfc[32])
        self.se_8_64 = SEBlock(nfc[8], nfc[64])
        self.se_16_128 = SEBlock(nfc[16], nfc[128])
        self.se_32_256 = SEBlock(nfc[32], nfc[256])
        self.se_64_512 = SEBlock(nfc[64], nfc[512])
        self.se_128_1024 = SEBlock(nfc[128], nfc[1024])


        self.to_128 = conv2d(nfc[128], nc, 1, 1, 0, bias=False) 
        self.to_big = conv2d(nfc[im_size], nc, 3, 1, 1, bias=False) 
        
        if im_size > 256:
            self.feat_512 = UpBlockComp(nfc[256], nfc[512]) 
            self.se_512 = SEBlock(nfc[32], nfc[512])
        if im_size > 512:
            self.feat_1024 = UpBlock(nfc[512], nfc[1024])  
        
    def forward(self, input, mapping=False):
        
        feat_4   = self.init(input)
        feat_8   = self.feat_8(feat_4)

        # Origin Skip Connection
        feat_16  = self.feat_16(feat_8)
        feat_32  = self.feat_32(feat_16)
        feat_64  = self.se_64( feat_4, self.feat_64(feat_32) )

        feat_128 = self.se_128( feat_8, self.feat_128(feat_64) )

        feat_256 = self.se_256( feat_16, self.feat_256(feat_128) )


        # feat_32  = self.se_4_32(feat_4, self.feat_32(feat_16))
        # feat_64 = self.se_8_64(feat_8, self.feat_64(feat_32))
        # feat_128 = self.se_16_128(feat_16, self.feat_128(feat_64))
        # feat_256 = self.se_32_256(feat_32, self.feat_256(feat_128))

        if self.im_size == 256:
            if not mapping:
                return [self.to_big(feat_256), self.to_128(feat_128)]
            else:
                return feat_256

        feat_512 = self.se_512( feat_32, self.feat_512(feat_256) )
        # feat_512 = self.se_64_512(feat_64, self.feat_512(feat_256))
        if self.im_size == 512:
            if not mapping:
                return [self.to_big(feat_512), self.to_128(feat_128)]
            else:
                return feat_512

        feat_1024 = self.feat_1024(feat_512)
        # feat_1024 = self.se_128_1024(feat_128, self.feat_1024(feat_512))

        im_128 = torch.tanh(self.to_128(feat_128))
        im_1024 = torch.tanh(self.to_big(feat_1024))
        if not mapping:
            return [im_1024, im_128]
        else:
            return feat_1024


class DownBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownBlock, self).__init__()

        self.main = nn.Sequential(
            conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2, inplace=True),
            )

    def forward(self, feat):
        return self.main(feat)


class DownBlockComp(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownBlockComp, self).__init__()

        self.main = nn.Sequential(
            conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2, inplace=True),
            conv2d(out_planes, out_planes, 3, 1, 1, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2)
            )

        self.direct = nn.Sequential(
            nn.AvgPool2d(2, 2),
            conv2d(in_planes, out_planes, 1, 1, 0, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2))

    def forward(self, feat):
        return (self.main(feat) + self.direct(feat)) / 2


class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=3, im_size=512):
        super(Discriminator, self).__init__()
        self.ndf = ndf
        self.im_size = im_size

        nfc_multi = {4:16, 8:16, 16:8, 32:4, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ndf)

        if im_size == 1024:
            self.down_from_big = nn.Sequential( 
                                    conv2d(nc, nfc[1024], 4, 2, 1, bias=False),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    conv2d(nfc[1024], nfc[512], 4, 2, 1, bias=False),
                                    batchNorm2d(nfc[512]),
                                    nn.LeakyReLU(0.2, inplace=True))
        elif im_size == 512:
            self.down_from_big = nn.Sequential( 
                                    conv2d(nc, nfc[512], 4, 2, 1, bias=False),
                                    nn.LeakyReLU(0.2, inplace=True) )
        elif im_size == 256:
            self.down_from_big = nn.Sequential( 
                                    conv2d(nc, nfc[512], 3, 1, 1, bias=False),
                                    nn.LeakyReLU(0.2, inplace=True) )

        self.down_4  = DownBlockComp(nfc[512], nfc[256])
        self.down_8  = DownBlockComp(nfc[256], nfc[128])
        self.down_16 = DownBlockComp(nfc[128], nfc[64])
        self.down_32 = DownBlockComp(nfc[64],  nfc[32])
        self.down_64 = DownBlockComp(nfc[32],  nfc[16])

        self.rf_big = nn.Sequential(
                            conv2d(nfc[16] , nfc[8], 1, 1, 0, bias=False),
                            batchNorm2d(nfc[8]), nn.LeakyReLU(0.2, inplace=True),
                            conv2d(nfc[8], 1, 4, 1, 0, bias=False))

        self.se_2_16 = SEBlock(nfc[512], nfc[64])
        self.se_4_32 = SEBlock(nfc[256], nfc[32])
        self.se_8_64 = SEBlock(nfc[128], nfc[16])
        
        self.down_from_small = nn.Sequential( 
                                            conv2d(nc, nfc[256], 4, 2, 1, bias=False), 
                                            nn.LeakyReLU(0.2, inplace=True),
                                            DownBlock(nfc[256],  nfc[128]),
                                            DownBlock(nfc[128],  nfc[64]),
                                            DownBlock(nfc[64],  nfc[32]), )

        self.rf_small = conv2d(nfc[32], 1, 4, 1, 0, bias=False)

        self.decoder_big = SimpleDecoder(nfc[16], nc)
        self.decoder_part = SimpleDecoder(nfc[32], nc)
        self.decoder_small = SimpleDecoder(nfc[32], nc)
        
    def forward(self, imgs, label, part=None, mapping=False):
        # for real imgs:
        if type(imgs) is not list:
            imgs = [F.interpolate(imgs, size=self.im_size), F.interpolate(imgs, size=128)]

        feat_2 = self.down_from_big(imgs[0])        
        feat_4 = self.down_4(feat_2)
        feat_8 = self.down_8(feat_4)
        
        feat_16 = self.down_16(feat_8)
        feat_16 = self.se_2_16(feat_2, feat_16)

        feat_32 = self.down_32(feat_16)
        feat_32 = self.se_4_32(feat_4, feat_32)
        
        feat_last = self.down_64(feat_32)
        feat_last = self.se_8_64(feat_8, feat_last)

        #rf_0 = torch.cat([self.rf_big_1(feat_last).view(-1),self.rf_big_2(feat_last).view(-1)])
        #rff_big = torch.sigmoid(self.rf_factor_big)
        rf_0 = self.rf_big(feat_last).view(-1)

        feat_small = self.down_from_small(imgs[1])
        #rf_1 = torch.cat([self.rf_small_1(feat_small).view(-1),self.rf_small_2(feat_small).view(-1)])
        rf_1 = self.rf_small(feat_small).view(-1)

        if label=='real':    
            rec_img_big = self.decoder_big(feat_last)
            rec_img_small = self.decoder_small(feat_small)

            assert part is not None
            rec_img_part = None
            if part==0:
                rec_img_part = self.decoder_part(feat_32[:,:,:8,:8])
            if part==1:
                rec_img_part = self.decoder_part(feat_32[:,:,:8,8:])
            if part==2:
                rec_img_part = self.decoder_part(feat_32[:,:,8:,:8])
            if part==3:
                rec_img_part = self.decoder_part(feat_32[:,:,8:,8:])

            return torch.cat([rf_0, rf_1]) , [rec_img_big, rec_img_small, rec_img_part]

        elif not mapping:
            return torch.cat([rf_0, rf_1])

        else:
            return feat_last


class SimpleDecoder(nn.Module):
    """docstring for CAN_SimpleDecoder"""
    def __init__(self, nfc_in=64, nc=3):
        super(SimpleDecoder, self).__init__()

        nfc_multi = {4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*32)

        def upBlock(in_planes, out_planes):
            block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
                batchNorm2d(out_planes*2), GLU())
            return block

        self.main = nn.Sequential(  nn.AdaptiveAvgPool2d(8),
                                    upBlock(nfc_in, nfc[16]) ,
                                    upBlock(nfc[16], nfc[32]),
                                    upBlock(nfc[32], nfc[64]),
                                    upBlock(nfc[64], nfc[128]),
                                    conv2d(nfc[128], nc, 3, 1, 1, bias=False),
                                    nn.Tanh() )

    def forward(self, input):
        # input shape: c x 4 x 4
        return self.main(input)

from random import randint
def random_crop(image, size):
    h, w = image.shape[2:]
    ch = randint(0, h-size-1)
    cw = randint(0, w-size-1)
    return image[:,:,ch:ch+size,cw:cw+size]

class TextureDiscriminator(nn.Module):
    def __init__(self, ndf=64, nc=3, im_size=512):
        super(TextureDiscriminator, self).__init__()
        self.ndf = ndf
        self.im_size = im_size

        nfc_multi = {4:16, 8:8, 16:8, 32:4, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ndf)

        self.down_from_small = nn.Sequential( 
                                            conv2d(nc, nfc[256], 4, 2, 1, bias=False), 
                                            nn.LeakyReLU(0.2, inplace=True),
                                            DownBlock(nfc[256],  nfc[128]),
                                            DownBlock(nfc[128],  nfc[64]),
                                            DownBlock(nfc[64],  nfc[32]), )
        self.rf_small = nn.Sequential(
                            conv2d(nfc[16], 1, 4, 1, 0, bias=False))

        self.decoder_small = SimpleDecoder(nfc[32], nc)
        
    def forward(self, img, label):
        img = random_crop(img, size=128)

        feat_small = self.down_from_small(img)
        rf = self.rf_small(feat_small).view(-1)
        
        if label=='real':    
            rec_img_small = self.decoder_small(feat_small)

            return rf, rec_img_small, img

        return rf