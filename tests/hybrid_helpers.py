from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from dust3r.memory.config import load_config
from dust3r.memory.hybrid import HybridMemory
from dust3r.memory.policies import AlwaysWritePolicy
from dust3r.memory.types import SharedNorms
from dust3r.memory.uncertainty import UncertaintyEstimator
from dust3r.online.runner import OnlineRunner


def tiny_config():
    return load_config(dict(memory=dict(input_dim=8, output_dim=8,
        short=dict(model_dim=8, num_heads=2, head_dim=4, backend='math_reference', require_flash=False),
        long=dict(model_dim=8, num_heads=2, key_dim=4, value_dim=4, backend='torch_reference'),
        fusion=dict(output_dim=8, hidden_dim=4)), uncertainty=dict(enabled=False),
        training=dict(new_memory_dtype='float32')))


class TinyEndoModel(nn.Module):
    """Actual Endo3R method contracts, without external imports/checkpoints."""
    def __init__(self):
        super().__init__()
        self.patch_size = (2, 2)
        self.feature = nn.Linear(3, 8)
        self.decoder = nn.Linear(8, 8)
        self.point_head = nn.Linear(8, 3)
        self.key_head = nn.Linear(16, 8)
        self.value_head = nn.Linear(3, 8)
        self.norm_q, self.norm_k, self.norm_v = (nn.LayerNorm(8) for _ in range(3))

    def encode_image(self, view):
        patches = F.avg_pool2d(view['img'], 2).flatten(2).transpose(1, 2)
        features = self.feature(patches)
        positions = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]]).unsqueeze(0)
        return features, positions, torch.tensor([[4, 4]])

    def encode_frames(self, left, right, feat1, feat2, pos1, pos2, shape1, shape2):
        if feat1 is None:
            feat1, pos1, shape1 = self.encode_image(left)
        else:
            feat1, pos1, shape1 = feat2, pos2, shape2
        feat2, pos2, shape2 = self.encode_image(right)
        return feat1, feat2, pos1, pos2, shape1, shape2

    def decode(self, fused, pos1, feat2, pos2):
        return [self.decoder(fused) + .2 * feat2], [self.decoder(feat2) + .2 * fused]

    def downstream_head(self, dec, shape, number):
        points = torch.sigmoid(self.point_head(dec[-1])) + 1
        points = points.transpose(1, 2).reshape(1, 3, 2, 2)
        points = F.interpolate(points, size=(4, 4), mode='nearest').permute(0, 2, 3, 1)
        return {'pts3d': points, 'conf': torch.ones(1, 4, 4)}

    def encode_feat_key(self, raw, decoded, number):
        return self.key_head(torch.cat((raw, decoded), -1))

    def encode_cur_value(self, res1, dec1, pos1, shape1):
        patches = F.avg_pool2d(res1['pts3d'].permute(0, 3, 1, 2), 2).flatten(2).transpose(1, 2)
        return self.value_head(patches)


def frames(count=5):
    return [dict(img=torch.full((1, 3, 4, 4), (i + 1) / 10),
                 origin_img=torch.full((1, 4, 4, 3), 10 * (i + 1), dtype=torch.uint8))
            for i in range(count)]


def make_runner(policy=None, uncertainty=None, training=False):
    model = TinyEndoModel()
    config = tiny_config()
    memory = HybridMemory(config.memory, SharedNorms(model.norm_q, model.norm_k, model.norm_v))
    estimator = uncertainty or UncertaintyEstimator(config.uncertainty, patch_hw=(2, 2))
    return OnlineRunner(model, memory, policy or AlwaysWritePolicy(), estimator, config, training)
