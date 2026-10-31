import torch
import torch.nn as nn
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4 or 3)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

#AadIN
def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def local_adain(fp_512,fs_512,mask_p,mask_s):
    size = fp_512.size()
    for i in range(19):
        mask_p_512 = mask_p[:,i,:,:].repeat(1,512,1,1)
        mask_s_512 = mask_s[:,i,:,:].repeat(1,512,1,1)
        mask_p_512_reverse = 1-mask_p_512
        mask_s_512_reverse = 1-mask_s_512
        p = fp_512[mask_p_512==1].reshape(1,512,-1)
        s = fs_512[mask_s_512==1].reshape(1,512,-1)
        if p.size(2)==0 or s.size(2)==0 or p.size(2)==1 or s.size(2)==1:
            continue
        p_mean,p_std = calc_mean_std(p)
        s_mean,s_std = calc_mean_std(s)
        fp_512 = ((fp_512-p_mean.expand(size)*mask_p_512)/(p_std.expand(size)*mask_p_512+mask_p_512_reverse))*(s_std.expand(size)*mask_s_512+mask_s_512_reverse)+s_mean.expand(size)*mask_s_512

    return fp_512

def cal_content(content_feat):
    size = content_feat.size()
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat

class AdaAttN(nn.Module):
    def __init__(self, in_planes, max_sample=256 * 256, key_planes=None):
        super(AdaAttN, self).__init__()
        if key_planes is None:
            key_planes = in_planes
        self.f = nn.Conv2d(key_planes, key_planes, (1, 1))
        self.g = nn.Conv2d(key_planes, key_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim=-1)
        self.max_sample = max_sample

    def forward(self, content, style, content_key, style_key, seed=None):
        F = self.f(content_key)
        G = self.g(style_key)
        H = self.h(style)
        b, _, h_g, w_g = G.size()
        G = G.view(b, -1, w_g * h_g).contiguous()
        if w_g * h_g > self.max_sample:
            if seed is not None:
                torch.manual_seed(seed)
            index = torch.randperm(w_g * h_g).to(content.device)[:self.max_sample]
            G = G[:, :, index]
            style_flat = H.view(b, -1, w_g * h_g)[:, :, index].transpose(1, 2).contiguous()
        else:
            style_flat = H.view(b, -1, w_g * h_g).transpose(1, 2).contiguous()
        b, _, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        S = torch.bmm(F, G)
        # S: b, n_c, n_s
        S = self.sm(S)
        # mean: b, n_c, c
        mean = torch.bmm(S, style_flat)
        # std: b, n_c, c
        std = torch.sqrt(torch.relu(torch.bmm(S, style_flat ** 2) - mean ** 2)+1e-8)
        # mean, std: b, c, h, w
        mean = mean.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        std = std.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return std * cal_content(content) + mean

def exact_feature_distribution_matching(content, style):
    assert (content.size() == style.size()) ## content and style features should share the same shape
    B, C, W, H = content.size(0), content.size(1), content.size(2), content.size(3)
    _, index_content = torch.sort(content.view(B,C,-1))  ## sort content feature
    value_style, _ = torch.sort(style.view(B,C,-1))      ## sort style feature
    inverse_index = index_content.argsort(-1)
    transferred_content = content.view(B,C,-1) + value_style.gather(-1, inverse_index) - content.view(B,C,-1).detach()
    return transferred_content.view(B, C, W, H)