
import torch.nn as nn
import torch
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
from attention import ChannelAttn,SpatialAttn,CoordAtt,SKModule,ChannelGate,SpatialGate
seq = nn.Sequential

UP_TYPE = {0: 'bilinear', 1: 'nearest', 2: 'deconv'}
NORM_TYPE = {0: 'bn', 1: 'in', 2: None}
ATTENTION_TYPE = {0: 'channel', 1: 'spatial', 2: 'branch', 3: None}
SHORT_CUT = {0: False, 1: True}
SKIP_TYPE = {0: False, 1: True}

def decimal2binary(n):
    return bin(n).replace("0b", "")

def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))


def convTranspose2d(*args, **kwargs):
    return spectral_norm(nn.ConvTranspose2d(*args, **kwargs))


def batchNorm2d(*args, **kwargs):
    return nn.BatchNorm2d(*args, **kwargs)


def insNorm2d(*args, **kwargs):
    return nn.InstanceNorm2d(*args, **kwargs)


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
        nc = int(nc / 2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, feat, noise=None):
        if noise is None:
            batch, _, height, width = feat.shape
            noise = torch.randn(batch, 1, height, width).cuda()

        return feat + self.weight * noise


class Swish(nn.Module):
    def forward(self, feat):
        return feat * torch.sigmoid(feat)


class SEBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.main = nn.Sequential(nn.AdaptiveAvgPool2d(4),
                                  conv2d(ch_in, ch_out, 4, 1, 0, bias=False), Swish(),
                                  conv2d(ch_out, ch_out, 1, 1, 0, bias=False), nn.Sigmoid())

    def forward(self, feat_small, feat_big):
        return feat_big * self.main(feat_small)


class InitLayer(nn.Module):
    def __init__(self, nz, channel):
        super().__init__()

        self.init = nn.Sequential(
            convTranspose2d(nz, channel * 2, 4, 1, 0, bias=False),
            batchNorm2d(channel * 2), GLU())

    def forward(self, noise):
        noise = noise.view(noise.shape[0], -1, 1, 1)
        return self.init(noise)



class UpblockComp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpblockComp, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.deconv = convTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.bn = batchNorm2d(out_ch*2)
        self.inn = insNorm2d(out_ch*2)
        self.conv1 = conv2d(self.in_ch, self.out_ch * 2, 3, 1, 1, bias=False)
        self.conv2 = conv2d(self.out_ch, self.out_ch * 2, 3, 1, 1, bias=False)
        self.noise1 = NoiseInjection()
        self.noise2 = NoiseInjection()
        self.glu1 = GLU()
        self.glu2 = GLU()

    def set_arch(self, up_id, norm_id):
        self.up_type = UP_TYPE[up_id]
        self.norm_type = NORM_TYPE[norm_id]

    def forward(self, x):
        if self.up_type == 'deconv':
            x = self.deconv(x)
        else:
            x = F.interpolate(x, scale_factor=2, mode=self.up_type)

        x = self.noise1(self.conv1(x))
        if self.norm_type:
            if self.norm_type == 'bn':
                x = self.bn(x)
            elif self.norm_type == 'in':
                x = self.inn(x)
            else:
                raise NotImplementedError(self.norm_type)
        else:
            x = x
        h = self.glu1(x)

        h = self.noise2(self.conv2(h))
        if self.norm_type:
            if self.norm_type == 'bn':
                h = self.bn(h)
            elif self.norm_type == 'in':
                h = self.inn(h)
            else:
                raise NotImplementedError(self.norm_type)
        else:
            h = h
        h = self.glu2(h)

        return h

class Upblock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Upblock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.deconv = convTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.bn = batchNorm2d(out_ch*2)
        self.inn = insNorm2d(out_ch*2)
        self.conv = conv2d(self.in_ch, self.out_ch * 2, 3, 1, 1, bias=False)
        self.glu = GLU()

    def set_arch(self, up_id, norm_id):
        self.up_type = UP_TYPE[up_id]
        self.norm_type = NORM_TYPE[norm_id]

    def forward(self, x):
        if self.up_type == 'deconv':
            x = self.deconv(x)
        else:
            x = F.interpolate(x, scale_factor=2, mode=self.up_type)

        x = self.conv(x)

        if self.norm_type:
            if self.norm_type == 'bn':
                x = self.bn(x)
            elif self.norm_type == 'in':
                x = self.inn(x)
            else:
                raise NotImplementedError(self.norm_type)
        else:
            x = x

        h = self.glu(x)

        return h

class CellNoAttnOrigin(nn.Module):
    def __init__(self, in_ch, inter_ch, out_ch, skip_ins=[], cur_stage=1):
        super(CellNoAttnOrigin, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.inter_ch = inter_ch
        self.num_skip_in = len(skip_ins)
        self.cur_stage = cur_stage

        self.deconv_in = convTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.deconv_skip = convTranspose2d(inter_ch, inter_ch, kernel_size=2, stride=2)

        # self.bn_comp = batchNorm2d(inter_ch*2)
        # self.inn_comp = insNorm2d(inter_ch*2)
        # self.bn_up = batchNorm2d(out_ch*2)
        # self.inn_up = insNorm2d(out_ch*2)

        # self.UpblockComp = nn.Sequential(
        #     conv2d(self.in_ch, self.inter_ch * 2, 3, 1, 1, bias=False),
        #     NoiseInjection(),
        #     batchNorm2d(inter_ch * 2), GLU(),
        #     conv2d(self.inter_ch, self.inter_ch * 2, 3, 1, 1, bias=False),
        #     NoiseInjection(),
        #     batchNorm2d(inter_ch * 2), GLU()
        # )
        # self.Upblock = nn.Sequential(conv2d(self.inter_ch, self.out_ch * 2, 3, 1, 1, bias=False),
        #                              batchNorm2d(self.out_ch * 2),GLU()
        #                              )

        if cur_stage == 2:
            self.UpComp = UpblockComp(in_ch, inter_ch)
            self.Up = Upblock(inter_ch, out_ch)

        if self.num_skip_in:
            self.skip_in_ops = nn.ModuleList(
                [SEBlock(skip_ins[0], inter_ch),
                 SEBlock(skip_ins[1], out_ch)])
        else:
            self.skip_in_ops = None

        # self.sc_op = SEBlock(in_ch, out_ch)

    def set_arch(self, skip_ins, short_cut_id=None, up_id=None, norm_id=None):
        self.short_cut = SHORT_CUT[short_cut_id]
        self.UpComp.set_arch(up_id=up_id, norm_id=norm_id)
        self.Up.set_arch(up_id=up_id, norm_id=norm_id)

        if self.num_skip_in:
            self.skip_ins = [0 for _ in range(len(self.skip_in_ops))]
            for skip_idx, skip_in in enumerate(decimal2binary(skip_ins)[::-1]):
                self.skip_ins[-(skip_idx + 1)] = int(skip_in)

    def forward(self, input, skip_ft=None):
        h = self.UpComp(input)

        if self.num_skip_in:
            assert len(self.skip_in_ops) == len(self.skip_ins)
            # for skip_flag, ft, skip_in_op in zip(self.skip_ins, skip_ft, self.skip_in_ops):
            #     # print(ft.size(),h.size())
            #     if skip_flag:
            #         h = skip_in_op(ft, h)
            h = self.skip_in_ops[0](skip_ft[0],h)
        skip_out = h

        out = self.Up(skip_out)
        if self.num_skip_in:
            out = self.skip_in_ops[1](skip_ft[1],out)

        return skip_out, out


class CellNoAttnSkip(nn.Module):
    def __init__(self, in_ch, inter_ch, out_ch, skip_ins=[],shor_cuts=[], cur_stage=1):
        super(CellNoAttnSkip, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.inter_ch = inter_ch
        self.num_skip_in = len(skip_ins)
        self.num_short_cut = len(shor_cuts)
        self.cur_stage = cur_stage

        # self.deconv_in = convTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        # self.deconv_skip = convTranspose2d(inter_ch, inter_ch, kernel_size=2, stride=2)

        # self.bn_comp = batchNorm2d(inter_ch*2)
        # self.inn_comp = insNorm2d(inter_ch*2)
        # self.bn_up = batchNorm2d(out_ch*2)
        # self.inn_up = insNorm2d(out_ch*2)

        # self.UpblockComp = nn.Sequential(
        #     conv2d(self.in_ch, self.inter_ch * 2, 3, 1, 1, bias=False),
        #     NoiseInjection(),
        #     batchNorm2d(inter_ch * 2), GLU(),
        #     conv2d(self.inter_ch, self.inter_ch * 2, 3, 1, 1, bias=False),
        #     NoiseInjection(),
        #     batchNorm2d(inter_ch * 2), GLU()
        # )
        # self.Upblock = nn.Sequential(conv2d(self.inter_ch, self.out_ch * 2, 3, 1, 1, bias=False),
        #                              batchNorm2d(self.out_ch * 2),GLU()
        #                              )

        # if cur_stage == 2:
        self.UpComp = UpblockComp(in_ch, inter_ch)
        self.Up = Upblock(inter_ch, out_ch)

        if self.num_skip_in:
            self.skip_in_ops = nn.ModuleList(
                [SEBlock(small_ch, inter_ch) for small_ch in skip_ins])
        else:
            self.skip_in_ops = None

        if self.num_short_cut:
            self.sc_ops = nn.ModuleList(
                [SEBlock(small_ch, out_ch) for small_ch in shor_cuts])
        else:
            self.sc_ops = None


    def set_arch(self, skip_ins, short_cuts, up_id=1, norm_id=0):
        # self.short_cut = SHORT_CUT[short_cut_id]
        self.UpComp.set_arch(up_id=up_id, norm_id=norm_id)
        self.Up.set_arch(up_id=up_id, norm_id=norm_id)

        if self.num_skip_in:
            self.skip_ins = [0 for _ in range(len(self.skip_in_ops))]
            for skip_idx, skip_in in enumerate(decimal2binary(skip_ins)[::-1]):
                self.skip_ins[-(skip_idx + 1)] = int(skip_in)

        if self.num_short_cut:
            self.short_cuts = [0 for _ in range(len(self.sc_ops))]
            for sc_idx, sc_in in enumerate(decimal2binary(short_cuts)[::-1]):
                self.short_cuts[-(sc_idx + 1)] = int(sc_in)

    def forward(self, input, skip_ft=None, sc_ft=None):
        h = self.UpComp(input)
        if self.num_skip_in:
            assert len(self.skip_in_ops) == len(self.skip_ins)
            for skip_flag, ft, skip_in_op in zip(self.skip_ins, skip_ft, self.skip_in_ops):
                # print(ft.size(),h.size())
                if skip_flag:
                    h = skip_in_op(ft, h)
        skip_out = h

        out = self.Up(h)
        if self.num_short_cut:
            assert len(self.sc_ops) == len(self.short_cuts)
            for sc_flag, ft, sc_op in zip(self.short_cuts, sc_ft, self.sc_ops):
                # print(ft.size(),h.size())
                if sc_flag:
                    out = sc_op(ft, out)

        return skip_out, out

class CellAttn(nn.Module):
    def __init__(self, in_ch, inter_ch, out_ch, skip_ins=[], cur_stage=1):
        super(CellAttn, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.inter_ch = inter_ch
        self.num_skip_in = len(skip_ins)
        self.cur_stage = cur_stage

        self.deconv_in = convTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.deconv_skip = convTranspose2d(inter_ch, inter_ch, kernel_size=2, stride=2)

        if cur_stage == 2:
            self.UpComp = UpblockComp(in_ch, inter_ch)
            self.Up = Upblock(inter_ch, out_ch)

            self.channel_attn_comp = ChannelGate(inter_ch)
            self.spatial_attn_comp = SpatialGate()
            # self.channel_spatial_comp = CoordAtt(inter_ch, inter_ch)
            self.branch_attn_comp = SKModule(inter_ch)
            self.channel_attn_up = ChannelGate(out_ch)
            self.spatial_attn_up = SpatialGate()
            # self.channel_spatial_up = CoordAtt(out_ch, out_ch)
            self.branch_attn_up = SKModule(out_ch)

        if self.num_skip_in:
            self.skip_in_ops = nn.ModuleList(
                [SEBlock(small_ch, inter_ch) for small_ch in skip_ins])
        else:
            self.skip_in_ops = None

        self.sc_op = SEBlock(in_ch, out_ch)

    def set_arch(self, skip_ins, short_cut_id=None, up_id=None, norm_id=None, attn_id=None):
        self.short_cut = SHORT_CUT[short_cut_id]
        self.attn_type = ATTENTION_TYPE[attn_id]
        self.UpComp.set_arch(up_id=up_id, norm_id=norm_id)
        self.Up.set_arch(up_id=up_id, norm_id=norm_id)

        if self.num_skip_in:
            self.skip_ins = [0 for _ in range(len(self.skip_in_ops))]
            for skip_idx, skip_in in enumerate(decimal2binary(skip_ins)[::-1]):
                self.skip_ins[-(skip_idx + 1)] = int(skip_in)

    def forward(self, input, skip_ft=None):
        h = self.UpComp(input)
        if self.attn_type:
            if self.attn_type == 'channel':
                h = self.channel_attn_comp(h)
            elif self.attn_type == 'spatial':
                h = self.spatial_attn_comp(h)
            # elif self.attn_type == 'channel_spatial':
            #     h = self.channel_spatial_comp(h)
            else:
                h = self.branch_attn_comp(h)
        skip_out = h

        if self.num_skip_in:
            assert len(self.skip_in_ops) == len(self.skip_ins)
            for skip_flag, ft, skip_in_op in zip(self.skip_ins, skip_ft, self.skip_in_ops):
                # print(ft.size(),h.size())
                if skip_flag:
                    h = skip_in_op(ft, h)

        out = self.Up(h)
        if self.short_cut:
            out = self.sc_op(input,out)

        if self.attn_type:
            if self.attn_type == 'channel':
                out = self.channel_attn_up(out)
            elif self.attn_type == 'spatial':
                out = self.spatial_attn_up(out)
            else:
                out = self.branch_attn_comp(out)

        return skip_out, out


class Generator(nn.Module):
    def __init__(self, ngf=64, nz=100, nc=3, im_size=256, cur_stage = 1):
        super(Generator, self).__init__()

        nfc_multi = {4: 16, 8: 8, 16: 4, 32: 2, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * ngf)

        self.im_size = im_size

        self.cur_stage = cur_stage

        self.init = InitLayer(nz, channel=nfc[4])
        # # skip_origin
        # self.cell1 = CellOrign(nfc[4], nfc[16], nfc[8])
        # self.cell2 = CellOrign(nfc[16], nfc[64], nfc[32])
        # self.cell3 = CellOrign(nfc[64], nfc[256], nfc[128], (nfc[4],nfc[8],nfc[16]))

        # # skip_search
        self.cell1 = CellNoAttnSkip(nfc[4], nfc[8], nfc[16], shor_cuts=[nfc[4]], cur_stage=cur_stage)
        self.cell2 = CellNoAttnSkip(nfc[16], nfc[32], nfc[64], skip_ins=[nfc[8]], shor_cuts=[nfc[4], nfc[16]],
                                    cur_stage=cur_stage)
        self.cell3 = CellNoAttnSkip(nfc[64], nfc[128], nfc[256], skip_ins=[nfc[8], nfc[32]],
                                    shor_cuts=[nfc[4], nfc[16], nfc[64]], cur_stage=cur_stage)

        # self.feat_8 = UpBlockComp(nfc[4], nfc[8])
        # self.feat_16 = UpBlock(nfc[8], nfc[16])
        #
        # self.feat_32 = UpBlockComp(nfc[16], nfc[32])
        # self.feat_64 = UpBlock(nfc[32], nfc[64])
        #
        # self.feat_128 = UpBlockComp(nfc[64], nfc[128])
        # self.feat_256 = UpBlock(nfc[128], nfc[256])

        # self.se_64 = SEBlock(nfc[4], nfc[64])
        # self.se_128 = SEBlock(nfc[8], nfc[128])
        # self.se_256 = SEBlock(nfc[16], nfc[256])

        self.to_128 = conv2d(nfc[128], nc, 1, 1, 0, bias=False)
        self.to_big = conv2d(nfc[im_size], nc, 3, 1, 1, bias=False)

    def set_arch(self, arch_id):
        if not isinstance(arch_id, list):
            arch_id = arch_id.to('cpu').numpy().tolist()
        arch_id = [int(x) for x in arch_id]


        if self.cur_stage == 2:
            arch_stage1 = arch_id[:1] + arch_id[5:7]
            self.cell1.set_arch(short_cuts=arch_stage1[0], skip_ins=[], up_id=arch_stage1[1], norm_id=arch_stage1[2])
            arch_stage2 = arch_id[1:3] + arch_id[7:9]
            self.cell2.set_arch(short_cuts=arch_stage2[0], skip_ins=arch_stage2[1], up_id=arch_stage2[2],
                                norm_id=arch_stage2[3])
            arch_stage3 = arch_id[3:5] + arch_id[9:11]
            self.cell3.set_arch(short_cuts=arch_stage3[0], skip_ins=arch_stage3[1], up_id=arch_stage3[2],
                                norm_id=arch_stage3[3])
        else:
            arch_stage1 = arch_id[:1]
            self.cell1.set_arch(short_cuts=arch_stage1[0], skip_ins=[])
            arch_stage2 = arch_id[1:3]
            self.cell2.set_arch(short_cuts=arch_stage2[0], skip_ins=arch_stage2[1])
            arch_stage3 = arch_id[3:]
            self.cell3.set_arch(short_cuts=arch_stage3[0], skip_ins=arch_stage3[1])

        # # arch_comb : [SC, UP, Attn, Skips]
        # arch_stage1 = arch_id[:3]
        # self.cell1.set_arch(short_cut_id=arch_stage1[0], up_id=arch_stage1[1], attn_id=arch_stage1[2], skip_ins=[])
        # arch_stage2 = arch_id[3:7]
        # self.cell2.set_arch(short_cut_id=arch_stage2[0], up_id=arch_stage2[1], attn_id=arch_stage2[2], skip_ins=arch_stage2[3])
        # arch_stage3 = arch_id[7:]
        # self.cell3.set_arch(short_cut_id=arch_stage3[0], up_id=arch_stage3[1], attn_id=arch_stage3[2], skip_ins=arch_stage2[3])

        # # arch_origin_comb : [Norm, UP, Attn]
        # arch_stage1 = arch_id[:3]
        # self.cell1.set_arch(norm_id=arch_stage1[0], up_id=arch_stage1[1], attn_id=arch_stage1[2])
        # arch_stage2 = arch_id[3:6]
        # self.cell2.set_arch(norm_id=arch_stage2[0], up_id=arch_stage2[1], attn_id=arch_stage2[2])
        # arch_stage3 = arch_id[6:]
        # self.cell3.set_arch(norm_id=arch_stage3[0], up_id=arch_stage3[1], attn_id=arch_stage3[2])

    def forward(self, input):

        feat_4 = self.init(input)

        # # skip_origin
        # skip_out1, se1 = self.cell1(feat_4, feat_4)
        # skip_out2, se2 = self.cell2(skip_out1, se1)
        # skip_out3, se3 = self.cell3((feat_4, skip_out1, se1), se2)

        # skip_search
        skip_out1, se1 = self.cell1(feat_4)
        # print("skip_out1:{}".format(skip_out1.size()))
        skip_out2, se2 = self.cell2(se1, [skip_out1])
        skip_out3, se3 = self.cell3(se2, [skip_out1,skip_out2])

        img_256 = torch.tanh(self.to_big(se3))
        img_128 = torch.tanh(self.to_128(skip_out3))


        return [img_256, img_128]

class GeneratorNoAttnOrigin(nn.Module):
    def __init__(self, ngf=64, nz=100, nc=3, im_size=256, cur_stage=1):
        super(GeneratorNoAttnOrigin, self).__init__()

        nfc_multi = {4: 16, 8: 8, 16: 4, 32: 2, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * ngf)

        self.im_size = im_size

        self.cur_stage = cur_stage

        self.init = InitLayer(nz, channel=nfc[4])
        # skip_origin
        self.cell1 = CellNoAttnOrigin(nfc[4], nfc[8], nfc[16], cur_stage=cur_stage)
        self.cell2 = CellNoAttnOrigin(nfc[16], nfc[32], nfc[64], cur_stage=cur_stage)
        self.cell3 = CellNoAttnOrigin(nfc[64], nfc[128], nfc[256], [nfc[8], nfc[16]], cur_stage=cur_stage)
        self.seblock = SEBlock(nfc[4],nfc[64])

        # # skip_search
        # self.cell1 = CellNoAttn(nfc[4], nfc[8], nfc[16], cur_stage=cur_stage)
        # self.cell2 = CellNoAttn(nfc[16], nfc[32], nfc[64], [nfc[8]], cur_stage=cur_stage)
        # self.cell3 = CellNoAttn(nfc[64], nfc[128], nfc[256], [nfc[8], nfc[32]], cur_stage=cur_stage)

        # self.feat_8 = UpBlockComp(nfc[4], nfc[8])
        # self.feat_16 = UpBlock(nfc[8], nfc[16])
        #
        # self.feat_32 = UpBlockComp(nfc[16], nfc[32])
        # self.feat_64 = UpBlock(nfc[32], nfc[64])
        #
        # self.feat_128 = UpBlockComp(nfc[64], nfc[128])
        # self.feat_256 = UpBlock(nfc[128], nfc[256])

        # self.se_64 = SEBlock(nfc[4], nfc[64])
        # self.se_128 = SEBlock(nfc[8], nfc[128])
        # self.se_256 = SEBlock(nfc[16], nfc[256])

        self.to_128 = conv2d(nfc[128], nc, 1, 1, 0, bias=False)
        self.to_big = conv2d(nfc[im_size], nc, 3, 1, 1, bias=False)

    def set_arch(self, arch_id):
        if not isinstance(arch_id, list):
            arch_id = arch_id.to('cpu').numpy().tolist()
        arch_id = [int(x) for x in arch_id]

        if self.cur_stage == 1:
            # arch_comb : [SC, UP, Norm, Skips]
            arch_stage1 = arch_id[:1]
            self.cell1.set_arch(short_cut_id=arch_stage1[0], skip_ins=[])
            arch_stage2 = arch_id[1:3]
            self.cell2.set_arch(short_cut_id=arch_stage2[0], skip_ins=arch_stage2[1])
            arch_stage3 = arch_id[3:]
            self.cell3.set_arch(short_cut_id=arch_stage3[0], skip_ins=arch_stage3[1])
        elif self.cur_stage == 2:
            # arch_comb : [SC, UP, Norm, Skips]

            arch_stage1 = arch_id[:1] + arch_id[5:7]
            self.cell1.set_arch(short_cut_id=arch_stage1[0], skip_ins=[], up_id=arch_stage1[1], norm_id=arch_stage1[2])
            arch_stage2 = arch_id[1:3] + arch_id[7:9]
            self.cell2.set_arch(short_cut_id=arch_stage2[0], skip_ins=arch_stage2[1], up_id=arch_stage2[2], norm_id=arch_stage2[3])
            arch_stage3 = arch_id[3:5] + arch_id[9:11]
            self.cell3.set_arch(short_cut_id=arch_stage3[0], skip_ins=arch_stage3[1], up_id=arch_stage3[2], norm_id=arch_stage3[3])

        # # arch_comb : [SC, UP, Attn, Skips]
        # arch_stage1 = arch_id[:3]
        # self.cell1.set_arch(short_cut_id=arch_stage1[0], up_id=arch_stage1[1], attn_id=arch_stage1[2], skip_ins=[])
        # arch_stage2 = arch_id[3:7]
        # self.cell2.set_arch(short_cut_id=arch_stage2[0], up_id=arch_stage2[1], attn_id=arch_stage2[2], skip_ins=arch_stage2[3])
        # arch_stage3 = arch_id[7:]
        # self.cell3.set_arch(short_cut_id=arch_stage3[0], up_id=arch_stage3[1], attn_id=arch_stage3[2], skip_ins=arch_stage2[3])

        # # arch_origin_comb : [Norm, UP, Attn]
        # arch_stage1 = arch_id[:3]
        # self.cell1.set_arch(norm_id=arch_stage1[0], up_id=arch_stage1[1], attn_id=arch_stage1[2])
        # arch_stage2 = arch_id[3:6]
        # self.cell2.set_arch(norm_id=arch_stage2[0], up_id=arch_stage2[1], attn_id=arch_stage2[2])
        # arch_stage3 = arch_id[6:]
        # self.cell3.set_arch(norm_id=arch_stage3[0], up_id=arch_stage3[1], attn_id=arch_stage3[2])

    def forward(self, input):

        feat_4 = self.init(input)

        # # skip_origin
        # skip_out1, se1 = self.cell1(feat_4, feat_4)
        # skip_out2, se2 = self.cell2(skip_out1, se1)
        # skip_out3, se3 = self.cell3((feat_4, skip_out1, se1), se2)

        # skip_search
        skip_out1, se1 = self.cell1(feat_4)
        # print("skip_out1:{}".format(skip_out1.size()))
        skip_out2, se2 = self.cell2(se1)
        se2 = self.seblock(feat_4, se2)
        skip_out3, se3 = self.cell3(se2, [skip_out1, se1])

        img_256 = self.to_big(se3)
        img_128 = self.to_128(skip_out3)

        return [img_256, img_128]

class GeneratorNoAttnskip(nn.Module):
    def __init__(self, ngf=64, nz=100, nc=3, im_size=256, cur_stage=1):
        super(GeneratorNoAttnskip, self).__init__()

        nfc_multi = {4: 16, 8: 8, 16: 4, 32: 2, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * ngf)

        self.im_size = im_size

        self.cur_stage = cur_stage

        self.init = InitLayer(nz, channel=nfc[4])
        # # skip_origin
        # self.cell1 = CellOrign(nfc[4], nfc[16], nfc[8])
        # self.cell2 = CellOrign(nfc[16], nfc[64], nfc[32])
        # self.cell3 = CellOrign(nfc[64], nfc[256], nfc[128], (nfc[4],nfc[8],nfc[16]))

        # skip_search
        self.cell1 = CellNoAttnSkip(nfc[4], nfc[8], nfc[16], shor_cuts=[nfc[4]], cur_stage=cur_stage)
        self.cell2 = CellNoAttnSkip(nfc[16], nfc[32], nfc[64], skip_ins=[nfc[8]], shor_cuts=[nfc[4],nfc[16]], cur_stage=cur_stage)
        self.cell3 = CellNoAttnSkip(nfc[64], nfc[128], nfc[256], skip_ins=[nfc[8], nfc[32]], shor_cuts=[nfc[4],nfc[16],nfc[64]], cur_stage=cur_stage)

        # self.feat_8 = UpBlockComp(nfc[4], nfc[8])
        # self.feat_16 = UpBlock(nfc[8], nfc[16])
        #
        # self.feat_32 = UpBlockComp(nfc[16], nfc[32])
        # self.feat_64 = UpBlock(nfc[32], nfc[64])
        #
        # self.feat_128 = UpBlockComp(nfc[64], nfc[128])
        # self.feat_256 = UpBlock(nfc[128], nfc[256])

        # self.se_64 = SEBlock(nfc[4], nfc[64])
        # self.se_128 = SEBlock(nfc[8], nfc[128])
        # self.se_256 = SEBlock(nfc[16], nfc[256])

        self.to_128 = conv2d(nfc[128], nc, 1, 1, 0, bias=False)
        self.to_big = conv2d(nfc[im_size], nc, 3, 1, 1, bias=False)

    def set_arch(self, arch_id):
        if not isinstance(arch_id, list):
            arch_id = arch_id.to('cpu').numpy().tolist()
        arch_id = [int(x) for x in arch_id]

        # if self.cur_stage == 1:
            # arch_comb : [SC, UP, Norm, Skips]
        arch_stage1 = arch_id[:1]
        self.cell1.set_arch(short_cuts=arch_stage1[0], skip_ins=[])
        arch_stage2 = arch_id[1:3]
        self.cell2.set_arch(short_cuts=arch_stage2[0], skip_ins=arch_stage2[1])
        arch_stage3 = arch_id[3:]
        self.cell3.set_arch(short_cuts=arch_stage3[0], skip_ins=arch_stage3[1])

        # elif self.cur_stage == 2:
        #     # arch_comb : [SC, UP, Norm, Skips]
        #
        #     arch_stage1 = arch_id[:1] + arch_id[5:7]
        #     self.cell1.set_arch(short_cut_id=arch_stage1[0], skip_ins=[], up_id=arch_stage1[1], norm_id=arch_stage1[2])
        #     arch_stage2 = arch_id[1:3] + arch_id[7:9]
        #     self.cell2.set_arch(short_cut_id=arch_stage2[0], skip_ins=arch_stage2[1], up_id=arch_stage2[2], norm_id=arch_stage2[3])
        #     arch_stage3 = arch_id[3:5] + arch_id[9:11]
        #     self.cell3.set_arch(short_cut_id=arch_stage3[0], skip_ins=arch_stage3[1], up_id=arch_stage3[2], norm_id=arch_stage3[3])

        # # arch_comb : [SC, UP, Attn, Skips]
        # arch_stage1 = arch_id[:3]
        # self.cell1.set_arch(short_cut_id=arch_stage1[0], up_id=arch_stage1[1], attn_id=arch_stage1[2], skip_ins=[])
        # arch_stage2 = arch_id[3:7]
        # self.cell2.set_arch(short_cut_id=arch_stage2[0], up_id=arch_stage2[1], attn_id=arch_stage2[2], skip_ins=arch_stage2[3])
        # arch_stage3 = arch_id[7:]
        # self.cell3.set_arch(short_cut_id=arch_stage3[0], up_id=arch_stage3[1], attn_id=arch_stage3[2], skip_ins=arch_stage2[3])

        # # arch_origin_comb : [Norm, UP, Attn]
        # arch_stage1 = arch_id[:3]
        # self.cell1.set_arch(norm_id=arch_stage1[0], up_id=arch_stage1[1], attn_id=arch_stage1[2])
        # arch_stage2 = arch_id[3:6]
        # self.cell2.set_arch(norm_id=arch_stage2[0], up_id=arch_stage2[1], attn_id=arch_stage2[2])
        # arch_stage3 = arch_id[6:]
        # self.cell3.set_arch(norm_id=arch_stage3[0], up_id=arch_stage3[1], attn_id=arch_stage3[2])

    def forward(self, input):

        feat_4 = self.init(input)

        # # skip_origin
        # skip_out1, se1 = self.cell1(feat_4, feat_4)
        # skip_out2, se2 = self.cell2(skip_out1, se1)
        # skip_out3, se3 = self.cell3((feat_4, skip_out1, se1), se2)

        # skip_search
        skip_out1, se1 = self.cell1(feat_4, sc_ft=[feat_4])
        # print("skip_out1:{}".format(skip_out1.size()))
        skip_out2, se2 = self.cell2(se1, skip_ft=[skip_out1], sc_ft=[feat_4,se1])
        skip_out3, se3 = self.cell3(se2, skip_ft=[skip_out1, skip_out2], sc_ft=[feat_4,se1,se2])

        img_256 = torch.tanh(self.to_big(se3))
        img_128 = torch.tanh(self.to_128(skip_out3))

        return [img_256, img_128]

class GeneratorAttn(nn.Module):
    def __init__(self, ngf=64, nz=100, nc=3, im_size=256, cur_stage=1):
        super(GeneratorAttn, self).__init__()

        nfc_multi = {4: 16, 8: 8, 16: 4, 32: 2, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * ngf)

        self.im_size = im_size

        self.cur_stage = cur_stage

        self.init = InitLayer(nz, channel=nfc[4])
        # # skip_origin
        # self.cell1 = CellOrign(nfc[4], nfc[16], nfc[8])
        # self.cell2 = CellOrign(nfc[16], nfc[64], nfc[32])
        # self.cell3 = CellOrign(nfc[64], nfc[256], nfc[128], (nfc[4],nfc[8],nfc[16]))

        # skip_search
        self.cell1 = CellAttn(nfc[4], nfc[8], nfc[16], cur_stage=cur_stage)
        self.cell2 = CellAttn(nfc[16], nfc[32], nfc[64], [nfc[8]], cur_stage=cur_stage)
        self.cell3 = CellAttn(nfc[64], nfc[128], nfc[256], [nfc[8], nfc[32]], cur_stage=cur_stage)

        # self.feat_8 = UpBlockComp(nfc[4], nfc[8])
        # self.feat_16 = UpBlock(nfc[8], nfc[16])
        #
        # self.feat_32 = UpBlockComp(nfc[16], nfc[32])
        # self.feat_64 = UpBlock(nfc[32], nfc[64])
        #
        # self.feat_128 = UpBlockComp(nfc[64], nfc[128])
        # self.feat_256 = UpBlock(nfc[128], nfc[256])

        # self.se_64 = SEBlock(nfc[4], nfc[64])
        # self.se_128 = SEBlock(nfc[8], nfc[128])
        # self.se_256 = SEBlock(nfc[16], nfc[256])

        self.to_128 = conv2d(nfc[128], nc, 1, 1, 0, bias=False)
        self.to_big = conv2d(nfc[im_size], nc, 3, 1, 1, bias=False)

    def set_arch(self, arch_id):
        if not isinstance(arch_id, list):
            arch_id = arch_id.to('cpu').numpy().tolist()
        arch_id = [int(x) for x in arch_id]

        if self.cur_stage == 1:
            # arch_comb : [SC, UP, Norm, Skips]
            arch_stage1 = arch_id[:1]
            self.cell1.set_arch(short_cut_id=arch_stage1[0], skip_ins=[])
            arch_stage2 = arch_id[1:3]
            self.cell2.set_arch(short_cut_id=arch_stage2[0], skip_ins=arch_stage2[1])
            arch_stage3 = arch_id[3:]
            self.cell3.set_arch(short_cut_id=arch_stage3[0], skip_ins=arch_stage3[1])
        elif self.cur_stage == 2:
            # arch_comb : [SC, UP, Norm, Attn, Skips]

            arch_stage1 = arch_id[:1] + arch_id[5:8]
            self.cell1.set_arch(short_cut_id=arch_stage1[0], skip_ins=[], up_id=arch_stage1[1], norm_id=arch_stage1[2], attn_id=arch_stage1[3])
            arch_stage2 = arch_id[1:3] + arch_id[8:11]
            self.cell2.set_arch(short_cut_id=arch_stage2[0], skip_ins=arch_stage2[1], up_id=arch_stage2[2], norm_id=arch_stage2[3], attn_id=arch_stage1[4])
            arch_stage3 = arch_id[3:5] + arch_id[11:]
            self.cell3.set_arch(short_cut_id=arch_stage3[0], skip_ins=arch_stage3[1], up_id=arch_stage3[2], norm_id=arch_stage3[3], attn_id=arch_stage1[4])

        # # arch_comb : [SC, UP, Attn, Skips]
        # arch_stage1 = arch_id[:3]
        # self.cell1.set_arch(short_cut_id=arch_stage1[0], up_id=arch_stage1[1], attn_id=arch_stage1[2], skip_ins=[])
        # arch_stage2 = arch_id[3:7]
        # self.cell2.set_arch(short_cut_id=arch_stage2[0], up_id=arch_stage2[1], attn_id=arch_stage2[2], skip_ins=arch_stage2[3])
        # arch_stage3 = arch_id[7:]
        # self.cell3.set_arch(short_cut_id=arch_stage3[0], up_id=arch_stage3[1], attn_id=arch_stage3[2], skip_ins=arch_stage2[3])

        # # arch_origin_comb : [Norm, UP, Attn]
        # arch_stage1 = arch_id[:3]
        # self.cell1.set_arch(norm_id=arch_stage1[0], up_id=arch_stage1[1], attn_id=arch_stage1[2])
        # arch_stage2 = arch_id[3:6]
        # self.cell2.set_arch(norm_id=arch_stage2[0], up_id=arch_stage2[1], attn_id=arch_stage2[2])
        # arch_stage3 = arch_id[6:]
        # self.cell3.set_arch(norm_id=arch_stage3[0], up_id=arch_stage3[1], attn_id=arch_stage3[2])

    def forward(self, input):

        feat_4 = self.init(input)

        # # skip_origin
        # skip_out1, se1 = self.cell1(feat_4, feat_4)
        # skip_out2, se2 = self.cell2(skip_out1, se1)
        # skip_out3, se3 = self.cell3((feat_4, skip_out1, se1), se2)

        # skip_search
        skip_out1, se1 = self.cell1(feat_4)
        # print("skip_out1:{}".format(skip_out1.size()))
        skip_out2, se2 = self.cell2(se1, [skip_out1])
        skip_out3, se3 = self.cell3(se2, [skip_out1, skip_out2])

        img_256 = torch.tanh(self.to_big(se3))
        img_128 = torch.tanh(self.to_128(skip_out3))

        return [img_256, img_128]


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
    def __init__(self, ndf=64, nc=3, im_size=512,cur_stage=1):
        super(Discriminator, self).__init__()

        self.ndf = ndf
        self.im_size = im_size

        nfc_multi = {4: 16, 8: 16, 16: 8, 32: 4, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * ndf)

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
                nn.LeakyReLU(0.2, inplace=True))
        elif im_size == 256:
            self.down_from_big = nn.Sequential(
                conv2d(nc, nfc[512], 3, 1, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True))

        self.down_4 = DownBlockComp(nfc[512], nfc[256])
        self.down_8 = DownBlockComp(nfc[256], nfc[128])
        self.down_16 = DownBlockComp(nfc[128], nfc[64])
        self.down_32 = DownBlockComp(nfc[64], nfc[32])
        self.down_64 = DownBlockComp(nfc[32], nfc[16])

        # self.attn_down_2 = CoordAtt(nfc[512], nfc[512])
        # self.attn_down_4 = CoordAtt(nfc[256], nfc[256])
        # self.attn_down_8 = CoordAtt(nfc[128], nfc[128])
        # self.attn_down_16 = CoordAtt(nfc[64], nfc[64])
        # self.attn_down_32 = CoordAtt(nfc[32], nfc[32])
        # self.attn_down_64 = CoordAtt(nfc[16], nfc[16])

        self.rf_big = nn.Sequential(
            conv2d(nfc[16], nfc[8], 1, 1, 0, bias=False),
            batchNorm2d(nfc[8]), nn.LeakyReLU(0.2, inplace=True),
            conv2d(nfc[8], 1, 4, 1, 0, bias=False))

        self.se_2_16 = SEBlock(nfc[512], nfc[64])
        self.se_4_32 = SEBlock(nfc[256], nfc[32])
        self.se_8_64 = SEBlock(nfc[128], nfc[16])

        self.down_from_small = nn.Sequential(
            conv2d(nc, nfc[256], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            DownBlock(nfc[256], nfc[128]),
            DownBlock(nfc[128], nfc[64]),
            DownBlock(nfc[64], nfc[32]), )

        self.rf_small = conv2d(nfc[32], 1, 4, 1, 0, bias=False)

        self.decoder_big = SimpleDecoder(nfc[16], nc)
        self.decoder_part = SimpleDecoder(nfc[32], nc)
        self.decoder_small = SimpleDecoder(nfc[32], nc)

    def forward(self, imgs, label, part=None, mapping=False):
        # for real imgs:
        if type(imgs) is not list:
            imgs = [F.interpolate(imgs, size=self.im_size), F.interpolate(imgs, size=128)]

        # origin
        feat_2 = self.down_from_big(imgs[0])
        feat_4 = self.down_4(feat_2)
        feat_8 = self.down_8(feat_4)

        feat_16 = self.down_16(feat_8)
        feat_16 = self.se_2_16(feat_2, feat_16)

        feat_32 = self.down_32(feat_16)
        feat_32 = self.se_4_32(feat_4, feat_32)

        feat_last = self.down_64(feat_32)
        feat_last = self.se_8_64(feat_8, feat_last)

        # # attn
        # feat_2 = self.attn_down_2(self.down_from_big(imgs[0]))
        # feat_4 = self.attn_down_4(self.down_4(feat_2))
        # feat_8 = self.attn_down_8(self.down_8(feat_4))
        #
        # feat_16 = self.down_16(feat_8)
        # feat_16 = self.attn_down_16(self.se_2_16(feat_2, feat_16))
        #
        # feat_32 = self.down_32(feat_16)
        # feat_32 = self.attn_down_32(self.se_4_32(feat_4, feat_32))
        #
        # feat_last = self.down_64(feat_32)
        # feat_last = self.attn_down_64(self.se_8_64(feat_8, feat_last))

        rf_0 = self.rf_big(feat_last).view(-1)

        feat_small = self.down_from_small(imgs[1])

        rf_1 = self.rf_small(feat_small).view(-1)

        if label == 'real':
            rec_img_big = self.decoder_big(feat_last)
            rec_img_small = self.decoder_small(feat_small)

            assert part is not None
            rec_img_part = None
            if part == 0:
                rec_img_part = self.decoder_part(feat_32[:, :, :8, :8])
            if part == 1:
                rec_img_part = self.decoder_part(feat_32[:, :, :8, 8:])
            if part == 2:
                rec_img_part = self.decoder_part(feat_32[:, :, 8:, :8])
            if part == 3:
                rec_img_part = self.decoder_part(feat_32[:, :, 8:, 8:])

            return torch.cat([rf_0, rf_1]), [rec_img_big, rec_img_small, rec_img_part]

        elif not mapping:
            return torch.cat([rf_0, rf_1])

        else:
            return feat_last,feat_small


class SimpleDecoder(nn.Module):
    """docstring for CAN_SimpleDecoder"""

    def __init__(self, nfc_in=64, nc=3):
        super(SimpleDecoder, self).__init__()

        nfc_multi = {4: 16, 8: 8, 16: 4, 32: 2, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * 32)

        def upBlock(in_planes, out_planes):
            block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                conv2d(in_planes, out_planes * 2, 3, 1, 1, bias=False),
                batchNorm2d(out_planes * 2), GLU())
            return block

        self.main = nn.Sequential(nn.AdaptiveAvgPool2d(8),
                                  upBlock(nfc_in, nfc[16]),
                                  upBlock(nfc[16], nfc[32]),
                                  upBlock(nfc[32], nfc[64]),
                                  upBlock(nfc[64], nfc[128]),
                                  conv2d(nfc[128], nc, 3, 1, 1, bias=False),
                                  nn.Tanh())

    def forward(self, input):
        # input shape: c x 4 x 4
        return self.main(input)


from random import randint


def random_crop(image, size):
    h, w = image.shape[2:]
    ch = randint(0, h - size - 1)
    cw = randint(0, w - size - 1)
    return image[:, :, ch:ch + size, cw:cw + size]


class TextureDiscriminator(nn.Module):
    def __init__(self, ndf=64, nc=3, im_size=512):
        super(TextureDiscriminator, self).__init__()
        self.ndf = ndf
        self.im_size = im_size

        nfc_multi = {4: 16, 8: 8, 16: 8, 32: 4, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * ndf)

        self.down_from_small = nn.Sequential(
            conv2d(nc, nfc[256], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            DownBlock(nfc[256], nfc[128]),
            DownBlock(nfc[128], nfc[64]),
            DownBlock(nfc[64], nfc[32]), )
        self.rf_small = nn.Sequential(
            conv2d(nfc[16], 1, 4, 1, 0, bias=False))

        self.decoder_small = SimpleDecoder(nfc[32], nc)

    def forward(self, img, label):
        img = random_crop(img, size=128)

        feat_small = self.down_from_small(img)
        rf = self.rf_small(feat_small).view(-1)

        if label == 'real':
            rec_img_small = self.decoder_small(feat_small)

            return rf, rec_img_small, img

        return rf