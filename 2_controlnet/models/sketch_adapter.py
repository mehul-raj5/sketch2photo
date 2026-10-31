"""SketchAdapter: frozen-codebook tokens → 16 conditioning channels at 512² (CN-L2).

(B, 256, 32, 32) z_q float32
  → L2-normalise each token (NEW-codebook-8: code norms span 1.9–21, the lookup is by direction)
  → 4 × [ConvTranspose2d(k4, s2, p1) → GroupNorm → ReLU]: 256→128 (64²) → 64 (128²) → 32 (256²) → 16 (512²)
  → Conv2d 3×3 → out_channels → sigmoid ∈ (0, 1)
GroupNorm replaces v1's BatchNorm (F18, NEW-controlnet-6): an image's conditioning no longer depends on its
batch partners or on running statistics, train and eval modes are identical, and the EMA of the parameters
is the whole state (NEW-pipeline-9). The residual variant without a sigmoid is not ported (v1 archive).
"""
import torch
import torch.nn as nn


class SketchAdapter(nn.Module):
    STAGES = (128, 64, 32, 16)

    def __init__(self, in_channels=256, out_channels=16, norm_groups=16, l2_normalise_tokens=True):
        super(SketchAdapter, self).__init__()
        for channels in self.STAGES:
            if channels % norm_groups != 0:
                raise ValueError('norm_groups=%d must divide every stage width %s' % (norm_groups, self.STAGES))
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.norm_groups = int(norm_groups)
        self.l2_normalise_tokens = bool(l2_normalise_tokens)
        layers = []
        prev = self.in_channels
        for channels in self.STAGES:
            layers += [nn.ConvTranspose2d(prev, channels, kernel_size=4, stride=2, padding=1),
                       nn.GroupNorm(self.norm_groups, channels),
                       nn.ReLU(inplace=True)]
            prev = channels
        self.upsample_layers = nn.Sequential(*layers)
        self.final_conv = nn.Conv2d(prev, self.out_channels, kernel_size=3, stride=1, padding=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z_q):
        x = z_q
        if self.l2_normalise_tokens:
            x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-6)
        return torch.sigmoid(self.final_conv(self.upsample_layers(x)))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
