import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from utils.adain import adaptive_instance_normalization
from utils.helper import ResidualBlock, NonLocalBlock, UpSampleBlock, DownSampleBlock, GroupNorm, Swish


# Feature re-calibration layer
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ParsingShuffleRes2Block(nn.Module):

    def __init__(self, inplanes, planes, stride=1, expansion = 1, downsample=False, use_se = True, baseWidth=26, scale = 4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(ParsingShuffleRes2Block, self).__init__()

        self.expansion = expansion

        width = int(math.floor(planes * (baseWidth/64.0)))
        self.conv1 = nn.Conv2d(inplanes * 2, width*scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width*scale)

        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride = stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(nn.Conv2d(width, width, kernel_size=3, stride = stride, padding=1, bias=False))
          bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width*scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.LeakyReLU(0.2, True)
        self.stype = stype
        self.scale = scale
        self.width  = width

        self.use_se = use_se
        if self.use_se:
            self.selayer = SELayer(planes * self.expansion, 16)

        self.downsample = downsample
        if self.downsample:
            self.down = nn.Sequential(
                nn.Conv2d(inplanes * 2, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()

        channels_per_group = num_channels // groups
        
        # reshape
        x = x.view(batchsize, groups, 
            channels_per_group, height, width)

        # transpose
        # - contiguous() required if transpose() is used before view().
        #   See https://github.com/pytorch/pytorch/issues/764
        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(batchsize, -1, height, width)

        return x

    def forward(self, x, c):
        
        residual = x

        x_c = torch.cat((x, c), 1)

        out = self.conv1(x_c)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.channel_shuffle(out, self.width)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0 or self.stype=='stage':
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(self.bns[i](sp))
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype=='normal':
          out = torch.cat((out, spx[self.nums]),1)
        elif self.scale != 1 and self.stype=='stage':
          out = torch.cat((out, self.pool(spx[self.nums])),1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.use_se:
            out = self.selayer(out)

        if self.downsample:
            residual = self.down(residual)

        out += residual
        out = self.relu(out)

        return out

class SPADEResBlk(nn.Module):
    def __init__(self, fin, fout, seg_fin):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        self.conv_0 = spectral_norm(self.conv_0)
        self.conv_1 = spectral_norm(self.conv_1)
        if self.learned_shortcut:
            self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        self.norm_0 = SPADE(fin, seg_fin)
        self.norm_1 = SPADE(fmiddle, seg_fin)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, seg_fin)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class SPADE(nn.Module):
    def __init__(self, cin, seg_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(seg_dim, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.alpha = nn.Conv2d(128, cin,
            kernel_size=3, stride=1, padding=1)
        self.beta = nn.Conv2d(128, cin,
            kernel_size=3, stride=1, padding=1)
            
    @staticmethod
    def PN(x):
        '''
            positional normalization: normalize each positional vector along the channel dimension
        '''
        assert len(x.shape) == 4, 'Only works for 4D(image) tensor'
        x = x - x.mean(dim=1, keepdim=True)
        x_norm = x.norm(dim=1, keepdim=True) + 1e-6
        x = x / x_norm
        return x
        
    def DPN(self, x, s):
        h, w = x.shape[2], x.shape[3]
        s = F.interpolate(s, (h, w), mode='bilinear', align_corners = False)
        s = self.conv(s)
        a = self.alpha(s)
        b  = self.beta(s)
        return x * (1 + a) + b

    def forward(self, x, s):
        x_out = self.DPN(self.PN(x), s)
        return x_out
    

class ResAdaIN(nn.Module):
    def __init__(self, dim, norm_layer=nn.BatchNorm2d, use_bias=False):
        super(ResAdaIN, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias)
        self.norm1 = norm_layer(dim)
        self.relu = nn.ReLU(True)
        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias)
        self.norm2 = norm_layer(dim)

    def forward(self, x, style):

        residual = x
        x = adaptive_instance_normalization(x,style)
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.pad2(x)
        x = self.conv2(x)
        x = self.norm2(x)
        out = x + residual

        return nn.ReLU(True)(out)

class ResBlock(nn.Module):
    def __init__(self, dim, norm_layer=nn.BatchNorm2d, use_bias=False):
        super(ResBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias)
        self.norm1 = norm_layer(dim)
        self.relu = nn.ReLU(True)
        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias)
        self.norm2 = norm_layer(dim)

    def forward(self, x):

        residual = x
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.pad2(x)
        x = self.conv2(x)
        x = self.norm2(x)
        out = x + residual

        return nn.ReLU(True)(out)
        
# class ResBlock_attn(nn.Module):
#     def __init__(self, dim, norm_layer=nn.BatchNorm2d, use_bias=False):
#         super(ResBlock_attn, self).__init__()
#         self.attn1 = SelfAttention(dim)
#         self.norm1 = norm_layer(dim)
#         self.relu = nn.ReLU(True)
#         self.attn2 = SelfAttention(dim)
#         self.norm2 = norm_layer(dim)

#     def forward(self, x):

#         residual = x
#         x = self.attn1(x)
#         x = self.norm1(x)
#         x = self.relu(x)
#         x = self.attn2(x)
#         x = self.norm2(x)
#         out = x + residual

#         return nn.ReLU(True)(out)
        
class Codebook(nn.Module):
    def __init__(self, num_codebook_vectors, latent_dim, k=10):
        super(Codebook, self).__init__()
        self.num_codebook_vectors = num_codebook_vectors  # 1024
        self.latent_dim = latent_dim  # 256
        self.beta = 0.25
        self.k = k  # top k

        self.embedding = nn.Embedding(self.num_codebook_vectors, self.latent_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)

    def forward(self, z, modified_codebook=None, ref=None):  # [b,c=256,h,w]
        if modified_codebook is not None:
            codebook = modified_codebook
        else:
            codebook = self.embedding.weight

        z = z.permute(0, 2, 3, 1).contiguous()  # [b,h,w,256]
        z_flattened = z.view(-1, self.latent_dim)  # [b*h*w=1024,256]
        # print(self.embedding.weight.shape, z.shape)
        # print(z_flattened.shape)
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(codebook ** 2, dim=1) - \
            2 * (torch.matmul(z_flattened, codebook.t()))

        # 计算最小索引
        min_encoding_indices = torch.argmin(d, dim=1)  # [b*h*w]
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        z_q = z + (z_q - z).detach()  # 使得z_q在相量值大小不变的情况下梯度与z相同

        z_q = z_q.permute(0, 3, 1, 2)  # [b,c=256,h,w]

        return z_q, min_encoding_indices, loss

class Smooth(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = [[1, 2, 1],
                  [2, 4, 2],
                  [1, 2, 1]]
        kernel = torch.tensor([[kernel]], dtype=torch.float)
        kernel /= kernel.sum()
        self.kernel = nn.Parameter(kernel, requires_grad=False)
        self.pad = nn.ReplicationPad2d(1)
    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        x = x.view(-1, 1, h, w)
        x = self.pad(x)
        x = F.conv2d(x, self.kernel)
        return x.view(b, c, h, w)


class Encoder(nn.Module):
    def __init__(self, input_channel = 3, output_channel = 256):
        super(Encoder, self).__init__()
        channels = [128, 128, 128, 256, 256, 512]
        attn_resolutions = [16]
        num_res_blocks = 2
        resolution = 256
        layers = [nn.Conv2d(input_channel, channels[0], 3, 1, 1)]
        for i in range(len(channels)-1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))
            if i != len(channels)-2 and i != len(channels)-3:
                layers.append(DownSampleBlock(channels[i+1]))
                resolution //= 2
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(NonLocalBlock(channels[-1]))
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(GroupNorm(channels[-1]))
        layers.append(Swish())
        layers.append(nn.Conv2d(channels[-1], output_channel, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):         # [1,3,256,256]
        x1 = self.model[:3](x)    # [1,128,256,256]
        x2 = self.model[3:6](x1)  # [1,128,128,128]
        x3 = self.model[6:9](x2)  # [1,256,64,64]
        x4 = self.model[9:15](x3)   # [1,512,32,32]
        x = self.model[15:](x4)   # [1,256,32,32]

        return x, [x,x3,x2,x1]
    

import functools
def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer
    
class Decoder(nn.Module):
    def __init__(self, input_channel = 256):
        super(Decoder, self).__init__()
        channels = [512, 256, 256, 128, 128]
        attn_resolutions = [16]
        num_res_blocks = 3
        resolution = 16

        in_channels = channels[0]
        layers = [nn.Conv2d(input_channel, in_channels, 3, 1, 1),
                  ResidualBlock(in_channels, in_channels),
                  NonLocalBlock(in_channels),
                  ResidualBlock(in_channels, in_channels)]
        for i in range(len(channels)):
            out_channels = channels[i]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))
            if i != 0 and i != 1:
                layers.append(UpSampleBlock(out_channels))
                resolution *= 2

        layers.append(GroupNorm(in_channels))
        layers.append(Swish())
        layers.append(nn.Conv2d(in_channels, 3, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    # no_skip
    def forward(self, x):           # [1, 256, 32, 32]
        x1 = self.model[:23](x)      # [1, 256, 64, 64]
        x2 = self.model[23:27](x1)    # [1, 128, 128, 128]
        x3 = self.model[27:31](x2)    # [1, 128, 256, 256]
        x4 = self.model[31:](x3)      # [1, 3, 256, 256]
        return x4
    
    def multi_forward(self, x):           # [1, 256, 32, 32]
        x1 = self.model[:23](x)      # [1, 256, 64, 64]
        x2 = self.model[23:27](x1)    # [1, 128, 128, 128]
        x3 = self.model[27:31](x2)    # [1, 128, 256, 256]
        return [None, x1,x2,x3]
    


def freeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False
    

def tensor2mask(tensor):
    tensor = tensor * 0.5 + 0.5 
    weights = torch.tensor([0.299, 0.587, 0.114], device=tensor.device).view(tensor.shape[0], 3, 1, 1)

    tensor_gray = (tensor * weights).sum(dim=1, keepdim=True)

    mask = 1.0 - tensor_gray 
    return mask

    
class Decoder_skip(nn.Module):
    def __init__(self,decoder):
        super(Decoder_skip, self).__init__()
        self.decoder = decoder
        freeze(self.decoder)
        self.cat1 = Fuse_sft_block(256,256)
        self.cat2 = Fuse_sft_block(128,128)
        self.global_att1 = nn.Sequential(
            ResidualBlock(512, 256),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 256//4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(256//4,256, kernel_size=1, stride=1, padding=0),
        )
        self.global_att2 = nn.Sequential(
            ResidualBlock(256, 128),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 128//4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128//4, 128, kernel_size=1, stride=1, padding=0),
        )
        self.sigmoid = nn.Sigmoid()
        self.detail_enhance1 = nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1))
        self.detail_enhance2 = nn.Sequential(
                    nn.Conv2d(128, 128, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(128, 128, kernel_size=3, padding=1))

    def forward(self, x, mask_list, sketch_enc_list, sketch, enc_list):
        sketch_mask = tensor2mask(sketch) 
        x3 = self.decoder.model[:23](x)
        w1 = self.global_att1(torch.cat([enc_list[1],mask_list[1]],dim=1))
        w1 = self.sigmoid(w1)
        x3 = self.cat1(mask_list[1],x3,w1)
        x3 = x3 + self.detail_enhance1(sketch_enc_list[1]) * (1 - F.interpolate(sketch_mask, size=64, mode='nearest'))
        x2 = self.decoder.model[23:27](x3)
        w2 = self.global_att2(torch.cat([enc_list[2],mask_list[2]],dim=1))
        w2 = self.sigmoid(w2)
        x2 = self.cat2(mask_list[2],x2,w2)
        x2 = x2 + self.detail_enhance2(sketch_enc_list[2]) * (1 - F.interpolate(sketch_mask, size=128, mode='nearest'))
        x1 = self.decoder.model[27:](x2)
        return x1
    
    

class Fuse_sft_block(nn.Module):
    def __init__(self, in_ch, out_ch, cat_num = 2):
        super().__init__()
        self.encode_enc = ResidualBlock(cat_num*in_ch, out_ch)

        self.scale = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))

        self.shift = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))

    def forward(self, mask, dec_feat, w, res_feat=None):
        if res_feat is not None:
            enc_feat = self.encode_enc(torch.cat([mask, dec_feat, res_feat], dim=1))
        else:
            enc_feat = self.encode_enc(torch.cat([mask, dec_feat], dim=1))
        scale = self.scale(enc_feat)
        shift = self.shift(enc_feat)
        if w is None:
            residual = dec_feat * scale + shift
        else:
            residual = w * (dec_feat * scale + shift)
        out = residual
        out = dec_feat + residual
        return out

