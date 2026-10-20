"""Deformation network H_D and motion network H_M (Fig. 3).

Both networks follow the reference implementation:
* The SAX stack is processed by 2D convolutions with its D slices as channels.
  LAX images (one channel) first go through a small stem that maps them to D
  channels, so that all views share the same 2D encoder (weight sharing).
* 2D features are fused, decoded back to D channels at full resolution and
  "un-squeezed" into a (1, D, H, W) volume, from which 3D convolutions predict a
  dense voxel-wise displacement field (3, D, H, W) in normalised units (tanh).
* The vertex-wise displacement is then sampled from that field (geometry.py).

With width=64 and width_3d=32 the layer sizes are those of the paper.
The output layers are zero-initialised (see zero_init).
"""
import torch
from torch import nn
import torch.nn.functional as F


def conv2d(cin, cout, k=3, s=1, p=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, p, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


def deconv2d(cin, cout):
    return nn.Sequential(nn.ConvTranspose2d(cin, cout, 3, 2, 1, output_padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


def conv3d(cin, cout, s=1):
    return nn.Sequential(nn.Conv3d(cin, cout, 3, s, 1, bias=False), nn.BatchNorm3d(cout), nn.ReLU(inplace=True))


def deconv3d(cin, cout):
    return nn.Sequential(nn.ConvTranspose3d(cin, cout, 3, 2, 1, output_padding=1, bias=False),
                         nn.BatchNorm3d(cout), nn.ReLU(inplace=True))


def zero_init(layer):
    """Start from a zero displacement field: the template (H_D) or identity (H_M).

    Not in the reference code, whose random initial fields move vertices by
    several centimetres; with few training iterations this converges much faster.
    """
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)


class LaxStem(nn.Module):
    """Maps a 1-channel LAX image to `depth` channels (Mesh_2d in the reference code)."""

    def __init__(self, depth):
        super().__init__()
        self.net = nn.Sequential(conv2d(1, depth // 2), conv2d(depth // 2, depth))

    def forward(self, x):
        return self.net(x)


class Encoder2D(nn.Module):
    """VGG-like 2D encoder shared by all views and frames; returns 5 scales."""

    def __init__(self, in_ch, width):
        super().__init__()
        w = width
        spec = [(in_ch, w, 2, 1), (w, 2 * w, 2, 2), (2 * w, 4 * w, 3, 2), (4 * w, 8 * w, 3, 2), (8 * w, 8 * w, 3, 2)]
        self.blocks = nn.ModuleList(
            nn.Sequential(conv2d(cin, cout, s=stride), *[conv2d(cout, cout) for _ in range(n - 1)])
            for cin, cout, n, stride in spec)
        self.channels = [cout for _, cout, _, _ in spec]

    def forward(self, x):
        feats = []
        for block in self.blocks:
            x = block(x)
            feats.append(x)
        return feats


class _MultiView(nn.Module):
    def __init__(self, depth, width, views):
        super().__init__()
        if "sa" not in views:
            raise ValueError("the SAX view is required")
        self.views = list(views)
        self.encoder = Encoder2D(depth, width)
        if len(self.views) > 1:
            self.lax_stem = LaxStem(depth)

    def encode(self, images):
        """images: dict 'sa' (B, D, H, W), '2ch'/'4ch' (B, 1, H, W) -> list (per view) of feature lists."""
        return [self.encoder(images[v] if v == "sa" else self.lax_stem(images[v])) for v in self.views]


class DeformationNet(_MultiView):
    """H_D: ED multi-view images -> template-to-ED field Phi_{tpl->0} (Fig. 3a)."""

    def __init__(self, depth, width=64, views=("sa", "2ch", "4ch")):
        super().__init__(depth, width, views)
        w = width
        self.fuse = conv2d(8 * w * len(self.views), 16 * w)
        self.decoder = nn.Sequential(
            deconv2d(16 * w, 8 * w), conv2d(8 * w, 8 * w),
            deconv2d(8 * w, 4 * w), conv2d(4 * w, 4 * w),
            deconv2d(4 * w, 2 * w), conv2d(2 * w, 2 * w),
            deconv2d(2 * w, depth), conv2d(depth, depth))
        self.head = nn.Sequential(conv3d(1, 3), nn.Conv3d(3, 3, 1))
        zero_init(self.head[-1])

    def forward(self, images):
        deepest = [feats[-1] for feats in self.encode(images)]
        x = self.decoder(self.fuse(torch.cat(deepest, 1)))
        return torch.tanh(self.head(x.unsqueeze(1)))


class MotionNet(_MultiView):
    """H_M: ED + frame-t multi-view images -> motion field Phi_{0->t} (Fig. 3b)."""

    def __init__(self, depth, width=64, width_3d=32, views=("sa", "2ch", "4ch")):
        super().__init__(depth, width, views)
        w, w3 = width, width_3d
        self.reduce = nn.ModuleList(conv2d(2 * c, w) for c in self.encoder.channels)
        self.fuse = nn.Sequential(conv2d(w * 5 * len(self.views), depth, 1, 1, 0), conv2d(depth, depth, 1, 1, 0))
        self.down = nn.Sequential(
            conv3d(1, w3),
            conv3d(w3, 2 * w3, s=2), conv3d(2 * w3, 2 * w3),
            conv3d(2 * w3, 4 * w3, s=2), conv3d(4 * w3, 4 * w3),
            conv3d(4 * w3, 8 * w3, s=2), conv3d(8 * w3, 8 * w3))
        self.up = nn.Sequential(
            deconv3d(8 * w3, 4 * w3), conv3d(4 * w3, 4 * w3),
            deconv3d(4 * w3, 2 * w3), conv3d(2 * w3, 2 * w3),
            deconv3d(2 * w3, w3), conv3d(w3, w3))
        self.head = nn.Conv3d(w3, 3, 1)
        zero_init(self.head)

    def forward(self, images_ed, images_t):
        size = images_ed["sa"].shape[-2:]
        multiscale = []
        for feats_t, feats_ed in zip(self.encode(images_t), self.encode(images_ed)):
            for i, (ft, fe) in enumerate(zip(feats_t, feats_ed)):
                x = self.reduce[i](torch.cat([ft, fe], 1))
                if i > 0:
                    x = F.interpolate(x, size=size, mode="bilinear", align_corners=True)
                multiscale.append(x)
        x = self.fuse(torch.cat(multiscale, 1)).unsqueeze(1)
        return torch.tanh(self.head(self.up(self.down(x))))
