import torch as jt
import torch.nn as nn
from torch.nn.utils import spectral_norm
# jt.flags.use_cuda = 1
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

class ChannelAttn(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttn, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c).contiguous()
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SpatialAttn(nn.Module):
    def __init__(self, in_c):
        super(SpatialAttn, self).__init__()
        self.in_c = in_c
        self.convA = conv2d(in_c, in_c, kernel_size=1)
        self.convB = conv2d(in_c, in_c, kernel_size=1)
        self.convV = conv2d(in_c, in_c, kernel_size=1)

    def forward(self, input):

        feature_maps = self.convA(input)
        atten_map = self.convB(input)
        b, _, h, w = feature_maps.shape

        feature_maps = feature_maps.view(b, 1, self.in_c, h*w).contiguous()
        atten_map = atten_map.view(b, self.in_c, 1, h*w).contiguous()
        global_descriptors = jt.mean(
            (feature_maps * nn.functional.softmax(atten_map, dim=-1)), dim=-1)

        v = self.convV(input)
        atten_vectors = nn.functional.softmax(
            v.view(b, self.in_c, h*w), dim=-1)
        out = jt.bmm(atten_vectors.permute(0, 2, 1).contiguous(),
                     global_descriptors).permute(0, 2, 1).contiguous()

        return out.view(b, _, h, w)

def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


class RadixSoftmax(nn.Module):
    def __init__(self, radix, cardinality):
        super(RadixSoftmax, self).__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2).contiguous()
            x = nn.functional.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = x.sigmoid()
        return x


class BranchAttn(nn.Module):
    """Split-Attention (aka Splat)
    """

    def __init__(self, in_channels, out_channels=None, kernel_size=3, stride=1, padding=None,
                 dilation=1, groups=1, bias=False, radix=2, rd_ratio=0.25, rd_channels=None, rd_divisor=8,
                 act_layer=nn.ReLU, norm_layer=None, drop_block=None, **kwargs):
        super(BranchAttn, self).__init__()
        out_channels = out_channels or in_channels
        self.radix = radix
        self.drop_block = drop_block
        mid_chs = out_channels * radix
        if rd_channels is None:
            attn_chs = make_divisible(
                in_channels * radix * rd_ratio, min_value=32, divisor=rd_divisor)
        else:
            attn_chs = rd_channels * radix

        padding = kernel_size // 2 if padding is None else padding
        self.conv = conv2d(
            in_channels, mid_chs, kernel_size, stride, padding, dilation,
            groups=groups * radix, bias=bias, **kwargs)
        self.bn0 = norm_layer(mid_chs) if norm_layer else nn.Identity()
        self.act0 = act_layer()
        self.fc1 = conv2d(out_channels, attn_chs, 1, groups=groups)
        self.bn1 = norm_layer(attn_chs) if norm_layer else nn.Identity()
        self.act1 = act_layer()
        self.fc2 = conv2d(attn_chs, mid_chs, 1, groups=groups)
        self.rsoftmax = RadixSoftmax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn0(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act0(x)

        B, RC, H, W = x.shape
        if self.radix > 1:
            x = x.reshape((B, self.radix, RC // self.radix, H, W))
            x_gap = x.sum(dim=1)
        else:
            x_gap = x
        x_gap = x_gap.mean(2, keepdims=True).mean(3, keepdims=True)
        x_gap = self.fc1(x_gap)
        x_gap = self.bn1(x_gap)
        x_gap = self.act1(x_gap)
        x_attn = self.fc2(x_gap)

        x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1).contiguous()
        if self.radix > 1:
            out = (x * x_attn.reshape((B, self.radix,
                                       RC // self.radix, 1, 1))).sum(dim=1)
        else:
            out = x * x_attn
        return out

class SKModule(nn.Module):
    def __init__(self, features, M=2, G=16, r=8, stride=1, L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the ratio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKModule, self).__init__()
        d = max(int(features/r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                conv2d(features, features, kernel_size=3, stride=stride,
                          padding=1+i, dilation=1+i, groups=G, bias=False),
                batchNorm2d(features),
                nn.ReLU()
            ))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(conv2d(features, d, kernel_size=1, stride=1, bias=False),
                                batchNorm2d(d),
                                nn.ReLU())
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                conv2d(d, features, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        batch_size = x.shape[0]

        feats = [conv(x) for conv in self.convs]
        feats = jt.cat(feats, dim=1)
        feats = feats.view(batch_size, self.M, self.features,
                           feats.shape[2], feats.shape[3])

        feats_U = jt.sum(feats, dim=1)
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = jt.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(
            batch_size, self.M, self.features, 1, 1)
        attention_vectors = self.softmax(attention_vectors)

        feats_V = jt.sum(feats*attention_vectors, dim=1)

        return feats_V

class h_sigmoid(nn.Module):
    def __init__(self):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6()

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = batchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = jt.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = jt.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = batchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelGate(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelGate, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_avg = nn.Sequential(
            linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            linear(channel // reduction, channel, bias=False),
        )
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc_max = nn.Sequential(
            linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            linear(channel // reduction, channel, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y_avg = self.avg_pool(x).view(b, c)
        y_avg = self.fc_avg(y_avg).view(b, c, 1, 1)

        y_max = self.max_pool(x).view(b, c)
        y_max = self.fc_max(y_max).view(b, c, 1, 1)

        y = self.sigmoid(y_avg + y_avg)
        return x * y.expand_as(x)


class ChannelPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_max = jt.max(x, 1).values.unsqueeze(1)
        x_avg = jt.mean(x, 1).unsqueeze(1)
        x = jt.concat([x_max, x_avg], dim=1)
        return x


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(
            kernel_size-1) // 2, relu=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = self.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio)
        self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        return x_out