import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from utils.modules import  ParsingShuffleRes2Block, ResAdaIN, Encoder, Decoder, Decoder_skip, SPADEResBlk
from utils.CVQ_quantise import VectorQuantiser
from utils.vgg import VGG19

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


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


# update learning rate (called once every epoch)
def update_learning_rate(scheduler, optimizer):
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print('learning rate = %.7f' % lr)

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if not m.weight.requires_grad:
                return
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None and m.bias.requires_grad:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            if not m.weight.requires_grad:
                return
            init.normal_(m.weight.data, 1.0, gain)
            if m.bias is not None and m.bias.requires_grad:
                init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids='cuda:0'):
    net.to(gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net

def define_G(stage1=False, norm='batch', init_type='normal', init_gain=0.02, gpu_ids='cuda', stage1_weights=None, num_embeddings=1024):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    if stage1: 
        net = train_codebook(norm_layer=norm_layer, num_embeddings=num_embeddings)
        net.to(gpu_ids)
    else:
        net = TransferNet_cvq(gpu_ids, norm_layer=norm_layer, stage1_weights=stage1_weights, num_embeddings=num_embeddings) # stage2
        net.to(gpu_ids)

    return net


class train_codebook(nn.Module):
    def __init__(self, input_nc=3, ngf=64, norm_layer=nn.BatchNorm2d, num_embeddings=1024):
        super(train_codebook, self).__init__()

        self.encoder = Encoder(input_channel=input_nc)
        self.decoder = Decoder()
        self.codebook = VectorQuantiser(num_embeddings, 256, 0.25)
        self.quant_conv = nn.Conv2d(256,256, 1)
        self.post_quant_conv = nn.Conv2d(256,256, 1)

        
    def forward(self, x):
        encoded_images, _ = self.encoder(x)
        quant_conv_encoded_images = self.quant_conv(encoded_images)
        codebook_mapping, _, q_loss = self.codebook(quant_conv_encoded_images)
        post_quant_conv_mapping = self.post_quant_conv(codebook_mapping)
        rec_x = self.decoder(post_quant_conv_mapping)
        
        return rec_x, q_loss


class TransferNet_cvq(nn.Module):
    def __init__(self, gpu_ids, input_nc=3, ngf=64, norm_layer=nn.BatchNorm2d, stage1_weights=None, num_embeddings=1024):
        super(TransferNet_cvq, self).__init__()
        self.encoder = Encoder(input_channel=input_nc)
        self.decoder = Decoder()
        self.quant_conv = nn.Conv2d(256,256, 1)
        self.post_quant_conv = nn.Conv2d(256,256, 1)
        self.codebook = VectorQuantiser(num_embeddings, 256, 0.25, contras_loss=False)

        if stage1_weights is not None:
            self.load_state_dict(torch.load(stage1_weights, map_location=torch.device(gpu_ids)), strict=True)
            self.post_quant_conv.requires_grad_(False)
            self.quant_conv.requires_grad_(False)
            self.codebook.requires_grad_(False)
            self.decoder.requires_grad_(False)

            sketch_weights = stage1_weights.replace('netG_a', 'netG_b')
            sketch_state_dict = torch.load(sketch_weights, map_location=torch.device(gpu_ids))
            sketch_encoder_state_dict = {k[len('encoder.'):]: v for k, v in sketch_state_dict.items() if k.startswith('encoder.')}
            self.sketch_encoder = Encoder(input_channel=3)
            self.sketch_encoder.load_state_dict(sketch_encoder_state_dict, strict=True)
        
        self.spade = SPADEResBlk(256,256,3)
        self.mask_encoder = Encoder(19)
        self.resadain1 = nn.Sequential(
                    ResAdaIN(256),
                    ResAdaIN(256),
                    ResAdaIN(256))
        self.resadain2 = nn.Sequential(
                    ResAdaIN(128),
                    ResAdaIN(128),
                    ResAdaIN(128))
        self.decoder = Decoder_skip(self.decoder)

    def forward(self, x, mask, ref):

        mask = torch.softmax(mask,1)
        _, mask_list = self.mask_encoder(mask)

        # ref encoding
        ref, ref_list = self.encoder(ref)
        encoded_x, encoded_list = self.encoder(x)
        for i in range (3):
            encoded_list[1] = self.resadain1[i](encoded_list[1], ref_list[1])
            encoded_list[2] = self.resadain2[i](encoded_list[2], ref_list[2]) 

        quant_conv_encoded_images = self.quant_conv(encoded_x)
        codebook_mapping, _, q_loss = self.codebook(quant_conv_encoded_images)
        post_quant_conv_mapping = self.post_quant_conv(codebook_mapping)
        post_quant_conv_mapping = self.spade(post_quant_conv_mapping, x)

        _, encoded_x_list = self.sketch_encoder(x)
        
        fake_sketch = self.decoder(post_quant_conv_mapping, mask_list, encoded_x_list, x, encoded_list)

        return fake_sketch, q_loss


def define_D(input_nc, ndf, netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_ids='cuda:0'):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'n_layers':
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'pixel':
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'alpha':
        net = AlphaDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'beta':
        net = BetaDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)

    return init_net(net, init_type, init_gain, gpu_ids)


class AlphaDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(AlphaDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.x_in = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        )
        self.x_d1 = ParsingShuffleRes2Block(ndf, ndf, stride=2, expansion=2, downsample=True, stype='stage')
        self.x_d2 = ParsingShuffleRes2Block(ndf * 2, ndf * 2, stride=2, expansion=2, downsample=True, stype='stage')
        self.x_d3 = ParsingShuffleRes2Block(ndf * 4, ndf * 4, stride=1, expansion=2)
        self.c_in = nn.Sequential(
            nn.Conv2d(19, ndf, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        )
        self.c_d1 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, kernel_size=3, stride=2, padding=1),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
        )
        self.c_d2 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=3, stride=2, padding=1),
            norm_layer(ndf * 4),
            nn.LeakyReLU(0.2, True),
        )

        self.x_out = nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)

        self.use_sigmoid = use_sigmoid

    def forward(self, x, c):
        x_in = self.x_in(x)
        c_in = self.c_in(c)
        x_d1 = self.x_d1(x_in, c_in)
        c_d1 = self.c_d1(c_in)
        x_d2 = self.x_d2(x_d1, c_d1)
        c_d2 = self.c_d2(c_d1)
        x_d3 = self.x_d3(x_d2, c_d2)
        out = self.x_out(x_d3)
        if self.use_sigmoid:
            out = nn.Sigmoid(out)

        return out


class BetaDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(BetaDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.x_in = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        )
        self.x_d1 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 2, kernel_size=4, stride=2, padding=1),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
        )
        self.x_d2 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 4, kernel_size=4, stride=2, padding=1),
            norm_layer(ndf * 4),
            nn.LeakyReLU(0.2, True),
        )
        self.x_d3 = nn.Sequential(
            nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=1, padding=1),
            norm_layer(ndf * 8),
            nn.LeakyReLU(0.2, True),
        )

        self.c_in = nn.Sequential(
            nn.Conv2d(19, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        )
        self.c_d1 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
        )
        self.c_d2 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            norm_layer(ndf * 4),
            nn.LeakyReLU(0.2, True),
        )

        self.x_out = nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)

        self.use_sigmoid = use_sigmoid

    def forward(self, x, c):
        x_in = self.x_in(x)
        c_in = self.c_in(c)
        ff_in = torch.cat((x_in, c_in), 1)
        x_d1 = self.x_d1(ff_in)
        c_d1 = self.c_d1(c_in)
        ffm_d1 = torch.cat((x_d1, c_d1), 1)
        x_d2 = self.x_d2(ffm_d1)
        c_d2 = self.c_d2(c_d1)
        ffm_d2 = torch.cat((x_d2, c_d2), 1)
        x_d3 = self.x_d3(ffm_d2)
        out = self.x_out(x_d3)
        if self.use_sigmoid:
            out = nn.Sigmoid(out)

        return out


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        if isinstance(input, list):
            total_loss = 0
            num_features = len(input)

            for input in input:
                target_tensor = self.get_target_tensor(input, target_is_real)
                total_loss += self.loss(input, target_tensor)

            average_loss = total_loss / num_features
            return average_loss
        else:
            target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

from utils.id_encoder.model_irse import Backbone

class IDLoss(nn.Module):
    def __init__(self):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load('pretrained_models/model_ir_se50.pth'))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

    def extract_feats(self, x):
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y):
        n_samples = y.shape[0]
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            loss += 1 - diff_target
            count += 1

        return loss / count


def freeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False

# def cross_entropy(logits, y):
#     s = torch.exp(logits*1.0)
#     logits = s / torch.sum(s, dim=1, keepdim=True)
#     c = -(y * torch.log(logits)).sum(dim=-1)
#     return torch.mean(c)

# def sigmoid(z):
#     return 1 / (1 + np.exp(-z))

# def compute_loss_v1(y_true, y_pred):
#     t_loss = y_true * np.log(sigmoid(y_pred)) + \
#              (1 - y_true) * np.log(1 - sigmoid(y_pred))  # [batch_size,num_class]
#     loss = t_loss.mean(axis=-1)  # 得到每个样本的损失值, 这里可以是
#     return -loss.mean()  # 返回整体样本的损失均值（或其他）

