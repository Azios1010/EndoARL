# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# DUSt3R model class
# --------------------------------------------------------
from copy import deepcopy
import torch
import argparse
import os
from packaging import version
import huggingface_hub
from dust3r.core.raft import RAFT
from dust3r.post_process import estimate_focal_knowing_depth
from .utils.misc import fill_default_args, freeze_all_params, is_symmetrized, interleave, transpose_to_landscape
from .heads import head_factory
from dust3r.patch_embed import get_patch_embed
import cv2
from dust3r.utils.geometry import w2c_to_c2w, detect_dynamic_regions_3d_distance
from models.croco import CroCoNet  # noqa
import copy
import torch
import numpy as np
import torch.nn as nn
from functools import partial
from torch.nn import functional as F
from croco.models.blocks import Block
inf = float('inf')

hf_version_number = huggingface_hub.__version__
assert version.parse(hf_version_number) >= version.parse("0.22.0"), "Outdated huggingface_hub version, please reinstall requirements.txt"
class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """

    def __init__(self, dims, mode='sintel', padding_factor=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // padding_factor) + 1) * padding_factor - self.ht) % padding_factor
        pad_wd = (((self.wd // padding_factor) + 1) * padding_factor - self.wd) % padding_factor
        if mode == 'sintel':
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def load_model(model_path, device, verbose=True):
    if verbose:
        print('... loading model from', model_path)
    ckpt = torch.load(model_path, map_location='cpu')
    args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if 'landscape_only' not in args:
        args = args[:-1] + ', landscape_only=False)'
    else:
        args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    net = eval(args)
    s = net.load_state_dict(ckpt['model'], strict=False)
    if verbose:
        print(s)
    return net.to(device)


class AsymmetricCroCo3DStereo (
    CroCoNet,
    huggingface_hub.PyTorchModelHubMixin,
    library_name="dust3r",
    repo_url="https://github.com/junyi/monst3r",
    tags=["image-to-3d"],
):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).   
    """

    def __init__(self,
                 output_mode='pts3d_pose',
                 head_type='linear',
                 depth_mode=('exp', -inf, inf),
                 conf_mode=('exp', 1, inf),
                 freeze='none',
                 landscape_only=True,
                 patch_embed_cls='PatchEmbedDust3R',  # PatchEmbedDust3R or ManyAR_PatchEmbed
                 **croco_kwargs):
        self.patch_embed_cls = patch_embed_cls
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)

        self.dec_blocks2 = deepcopy(self.dec_blocks)
        self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, **croco_kwargs)
        self.set_freeze(freeze)


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu')
        else:
            return super(AsymmetricCroCo3DStereo, cls).from_pretrained(pretrained_model_name_or_path, **kw)

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            'none':     [],
            'mask':     [self.mask_token],
            'encoder':  [self.mask_token, self.patch_embed, self.enc_blocks],
            'encoder_and_decoder': [self.mask_token, self.patch_embed, self.enc_blocks, self.dec_blocks, self.dec_blocks2],
        }
        freeze_all_params(to_be_frozen[freeze])
        print(f'Freezing {freeze} parameters')

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size,
                            **kw):
        if type(img_size) is int:
            img_size = (img_size, img_size)
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        self.downstream_head2 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)

    def _encode_image(self, image, true_shape):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)
        # x (B, 576, 1024) pos (B, 576, 2); patch_size=16
        B,N,C = x.size()
        posvis = pos
        # add positional embedding without cls token
        assert self.enc_pos_embed is None
        # TODO: where to add mask for the patches
        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, posvis)

        x = self.enc_norm(x)
        return x, pos, None

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2):
        if img1.shape[-2:] == img2.shape[-2:]:
            out, pos, _ = self._encode_image(torch.cat((img1, img2), dim=0),
                                             torch.cat((true_shape1, true_shape2), dim=0))
            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
        else:
            out, pos, _ = self._encode_image(img1, true_shape1)
            out2, pos2, _ = self._encode_image(img2, true_shape2)
        return out, out2, pos, pos2

    def _encode_symmetrized(self, view1, view2):
        img1 = view1['img']
        img2 = view2['img']
        B = img1.shape[0]

        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))

        # warning! maybe the images have different portrait/landscape orientations
        if is_symmetrized(view1, view2):
            # computing half of forward pass!'
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1[::2], img2[::2], shape1[::2], shape2[::2])
            feat1, feat2 = interleave(feat1, feat2)
            pos1, pos2 = interleave(pos1, pos2)
        else:
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, img2, shape1, shape2)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)

    def _decoder(self, f1, pos1, f2, pos2):
        final_output = [(f1, f2)]  # before projection
        original_D = f1.shape[-1]

        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)

        final_output.append((f1, f2))
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            # img1 side
            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2)
            # img2 side
            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1)
            # store the result
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape)

    def forward(self, view1, view2):
        # encode the two images --> B,S,D
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2)

        # combine all ref images into object-centric representation
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)

        with torch.cuda.amp.autocast(enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)

        res2['pts3d_in_other_view'] = res2.pop('pts3d')  # predict view2's pts3d in view1's frame
        return res1, res2

class SpatialTempralMemory():
    def __init__(self, norm_q, norm_k, norm_v, mem_dropout=None, 
                 long_term_spat_size=20000, short_term_temp_size=4, 
                 attn_thresh=1e-3, sim_thresh=0.92, 
                 save_attn=False, num_patches=None):
        self.norm_q = norm_q #0.95
        self.norm_k = norm_k
        self.norm_v = norm_v
        self.mem_dropout = mem_dropout
        self.attn_thresh = attn_thresh
        self.long_mem_size = long_term_spat_size
        self.short_mem_size = short_term_temp_size
        self.top_k = long_term_spat_size
        self.save_attn = save_attn
        self.sim_thresh = sim_thresh
        self.num_patches = num_patches
        self.interval = None
        self.sf_model = torch.nn.DataParallel(RAFT(create_raft_args()))
        self.sf_model.load_state_dict(torch.load('./checkpoints/raft-things.pth'))
        self.sf_model = self.sf_model.module
        self.sf_model.requires_grad_(False)
        self.init_mem()
    
    def init_mem(self):
        self.temporal_buffer = []
        self.spatial_buffer = []
        self.mem_k = None
        self.mem_v = None
        self.mem_c = None
        self.temp_mem_k = None
        self.temp_mem_v = None
        self.temp_mem_c = None
        self.spat_mem_k = None
        self.spat_mem_v = None
        self.spat_mem_c = None
        
        self.mem_uncert = None
        self.mem_count = None
        self.mem_attn = None
        self.mem_pts = None
        self.temp_mem_pts = None
        self.spat_mem_pts = None
        self.mem_imgs = None
        self.temp_mem_imgs = None
        self.spat_mem_imgs = None
        self.lm = 0
        self.wm = 0
        if self.save_attn:
            self.attn_vis = None

    def add_mem_k(self, feat, mem_k):
        if mem_k is None:
            mem_k = feat
        else:
            mem_k = torch.cat((mem_k, feat), dim=1)

        return mem_k
    
    def add_mem_v(self, feat, mem_v):
        if mem_v is None:
            mem_v = feat
        else:
            mem_v = torch.cat((mem_v, feat), dim=1)

        return mem_v

    def add_mem_c(self, feat, mem_c):
        feat = F.avg_pool2d(feat, kernel_size=16, stride=16).reshape(1, 1, -1)
        if mem_c is None:
            mem_c = feat
        else:
            mem_c = torch.cat((mem_c, feat), dim=1)

        return mem_c
    
    def add_mem_pts(self, pts_cur, mem_pts):
        if pts_cur is not None:
            if mem_pts is None:
                mem_pts = pts_cur
            else:
                mem_pts = torch.cat((mem_pts, pts_cur), dim=1)
        return mem_pts
    
    def add_mem_img(self, img_cur, mem_imgs):
        if img_cur is not None:
            if mem_imgs is None:
                mem_imgs = img_cur
            else:
                mem_imgs = torch.cat((mem_imgs, img_cur), dim=1)
        return mem_imgs

    def add_mem_uncertainty(self, uncertainty, mem_uncert):
        uncertainty = F.avg_pool2d(uncertainty, kernel_size=16, stride=16).reshape(1, 1, -1)
        if mem_uncert is None:
            mem_uncert = uncertainty
        else:
            mem_uncert = torch.cat((mem_uncert, uncertainty), dim=2)
        return mem_uncert

    def add_temp_mem(self, feat_k, feat_v, pts_cur=None, img_cur=None):  
        if self.num_patches is None:
            self.num_patches = feat_k.shape[1]
            
        self.temp_mem_k = self.add_mem_k(feat_k, self.temp_mem_k)
        self.temp_mem_v = self.add_mem_v(feat_v, self.temp_mem_v)
        self.temp_mem_pts = self.add_mem_pts(pts_cur, self.temp_mem_pts)
        self.temp_mem_imgs = self.add_mem_img(img_cur, self.temp_mem_imgs)
        # print("adding to temporal: ", self.temp_mem_k.shape, self.temporal_buffer)
        
        del feat_k, feat_v, pts_cur, img_cur
    

    def add_spatial_mem(self, feat_k, feat_v, pts_cur=None, img_cur=None, uncertainty_cur=None, conf_cur=None):  
        if self.num_patches is None:
            self.num_patches = feat_k.shape[1]
            
        if self.mem_count is None:
            self.mem_count = torch.zeros_like(feat_k[:, :, :1])
            self.mem_attn = torch.zeros_like(feat_k[:, :, :1])
        else:
            self.mem_count += 1
            self.mem_count = torch.cat((self.mem_count, torch.zeros_like(feat_k[:, :, :1])), dim=1)
            self.mem_attn = torch.cat((self.mem_attn, torch.zeros_like(feat_k[:, :, :1])), dim=1)
        
        self.mem_k = self.add_mem_k(feat_k, self.mem_k)
        self.mem_v = self.add_mem_v(feat_v, self.mem_v)
        self.mem_pts = self.add_mem_pts(pts_cur, self.mem_pts)
        self.mem_imgs = self.add_mem_img(img_cur, self.mem_imgs)
        # use the first channel as confidence
        if uncertainty_cur is not None:
            self.mem_c = self.add_mem_c(conf_cur, self.mem_c)
            self.mem_uncert = self.add_mem_uncertainty(uncertainty_cur, self.mem_uncert)
            del uncertainty_cur, conf_cur
        del feat_k, feat_v, pts_cur, img_cur
    

    def check_sim(self, feat_k, thresh=0.7):
        # Do correlation with working memory
        if self.mem_k is None or thresh==1.0:
            return False
        
        wmem_size = self.wm * self.num_patches

        # wm: BS, T, 196, C
        wm = self.mem_k[:, -wmem_size:].reshape(self.mem_k.shape[0], -1, self.num_patches, self.mem_k.shape[-1])

        feat_k_norm = F.normalize(feat_k, p=2, dim=-1)
        wm_norm = F.normalize(wm, p=2, dim=-1)

        corr = torch.einsum('bpc,btpc->btp', feat_k_norm, wm_norm)

        mean_corr = torch.mean(corr, dim=-1)
        if mean_corr.max() > thresh:
            # print('NOT ADDING, Similarity detected:', "mean: ", mean_corr.mean(), "max: ", mean_corr.median(), "max: ", mean_corr.max(), f"then skip adding to memory with thresh {thresh}")
            return True
        else:
            # print("YES ADDING, detect sim: ", mean_corr.max(), thresh)
            return False
        
    
    def add_mem_check(self, feat_k, feat_v, pts_cur=None, img_cur=None, index=None, check_uncertainty=True, \
                      ref_view=None, cur_view=None, ref_pointmap=None, cur_pts3d=None, cur_conf=None):
        if self.num_patches is None:
            self.num_patches = feat_k.shape[1]
        self.temporal_buffer.append(index)
        
        self.add_temp_mem(feat_k, feat_v, pts_cur, img_cur)
        if len(self.temporal_buffer) > self.short_mem_size:
            
            self.temp_mem_k = self.temp_mem_k[:, -self.short_mem_size * self.num_patches:]
            self.temp_mem_v = self.temp_mem_v[:, -self.short_mem_size * self.num_patches:]
            self.temporal_buffer = self.temporal_buffer[-self.short_mem_size:]
        if self.check_sim(feat_k, thresh=self.sim_thresh): # and index - self.spatial_buffer[-1] < self.interval:
            # print('Similarity detected, skip adding to spatial memory')
            return
        if check_uncertainty:
            self.spatial_buffer.append(index)
            if index > 0:
                uncertainty = self.uncertainty_map(ref_view, cur_view, ref_pointmap, cur_pts3d)
                self.add_spatial_mem(feat_k, feat_v, pts_cur, img_cur, uncertainty, cur_conf)
            else:
                uncertainty = torch.zeros((1, cur_conf.shape[1], cur_conf.shape[2]), device=feat_k.device)
                self.add_spatial_mem(feat_k, feat_v, pts_cur, img_cur, uncertainty, cur_conf)
            
        else:
            self.spatial_buffer.append(index)
            self.add_spatial_mem(feat_k, feat_v, pts_cur, img_cur)
            
        
        
        self.wm += 1
        if self.wm > self.short_mem_size:
            self.wm -= 1
            self.lm += self.num_patches

        
        # self.lm += self.num_patches
        if self.lm > self.long_mem_size:
            self.memory_prune()
            self.lm = self.top_k - self.wm * self.num_patches

                
    def uncertainty_map(self, view1, view2, pointmap1, pointmap2):
        image1 = view1['origin_img'].permute(0, 3, 1, 2).cuda()
        image2 = view2['origin_img'].permute(0, 3, 1, 2).cuda()
        
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        flow_low, flow_up_fw = self.sf_model(image2, image1, iters=20, test_mode=True)
        flow_pr = padder.unpad(flow_up_fw).permute(0, 2, 3, 1)
        uncertainty_map = detect_dynamic_regions_3d_distance(pointmap2, pointmap1, flow_pr)
        return uncertainty_map
    

    def memory_read(self, feat, res=True):
        '''
        Params:
            - feat: [bs, p, c]
            - mem_k: [bs, t, p, c]
            - mem_v: [bs, t, p, c]
            - mem_c: [bs, t, p, 1]
        '''
        if self.mem_uncert is not None:
            mem_k = torch.cat((self.temp_mem_k, self.mem_k), dim=1) if self.temp_mem_k is not None else self.mem_k
            mem_v = torch.cat((self.temp_mem_v, self.mem_v), dim=1) if self.temp_mem_v is not None else self.mem_v
        else:
            mem_k = self.mem_k
            mem_v = self.mem_v
        affinity = torch.einsum('bpc,bxc->bpx', self.norm_q(feat), self.norm_k(mem_k.reshape(mem_k.shape[0], -1, mem_k.shape[-1])))
        affinity /= torch.sqrt(torch.tensor(feat.shape[-1]).float())
        # shape: b p c x b t*p c -> b p t*p
        if self.mem_uncert is not None:
            temp_patches = self.temp_mem_k.shape[1] * self.mem_uncert.shape[1]
            mem_uncert = torch.cat((torch.zeros(self.mem_uncert.shape[0], self.mem_uncert.shape[1], temp_patches).to(self.mem_uncert.device), self.mem_uncert), dim=2) \
                if self.temp_mem_k is not None else self.mem_uncert
            # mem_uncert = self.mem_uncert
            if mem_uncert.max() > 0:
                k = 20.0  # Adjustable decay rate
                confidence = torch.exp(-k * mem_uncert)  # Exponential decay
                confidence = torch.clamp(confidence, min=0.1, max=1.0)  # Softer minimum
                affinity = affinity * confidence
        
        attn = torch.softmax(affinity, dim=-1)

        if self.save_attn:
            if self.attn_vis is None:
                self.attn_vis = attn.reshape(-1)
            else:
                self.attn_vis = torch.cat((self.attn_vis, attn.reshape(-1)), dim=0)
        if self.mem_dropout is not None:
            attn = self.mem_dropout(attn)
        
        if self.attn_thresh > 0:
            attn[attn<self.attn_thresh] = 0
            attn = attn / attn.sum(dim=-1, keepdim=True) 
        
        out = torch.einsum('bpx,bxc->bpc', attn, self.norm_v(mem_v.reshape(mem_v.shape[0], -1, mem_v.shape[-1])))
        
        if res:
            out = out + feat
        
        
        total_attn = torch.sum(attn, dim=-2)
        if self.mem_uncert is not None:
            total_attn = total_attn[..., temp_patches:]
        # self.mem_attn += total_attn[..., None]
        
        return out
    
    def memory_prune(self):

        weights = self.mem_attn / self.mem_count
        weights[self.mem_count<self.short_mem_size+5] = 1e8

        num_mem_b = self.mem_k.shape[1]


        top_k_values, top_k_indices = torch.topk(weights, self.top_k, dim=1)
        top_k_indices_expanded = top_k_indices.expand(-1, -1, self.mem_k.size(-1))


        self.mem_k = torch.gather(self.mem_k, -2, top_k_indices_expanded)
        self.mem_v = torch.gather(self.mem_v, -2, top_k_indices_expanded)
        self.mem_attn = torch.gather(self.mem_attn, -2, top_k_indices)
        self.mem_count = torch.gather(self.mem_count, -2, top_k_indices)
        if self.mem_uncert is not None:
            self.mem_uncert = torch.gather(self.mem_uncert, -1, top_k_indices.permute(0, 2, 1))
        
        if self.mem_pts is not None:
            top_k_indices_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, self.num_patches, 3)
            self.mem_pts = torch.gather(self.mem_pts, 1, top_k_indices_expanded)
            self.mem_imgs = torch.gather(self.mem_imgs, 1, top_k_indices_expanded)

        num_mem_a = self.mem_k.shape[1]

        print('Memory pruned:', num_mem_b, '->', num_mem_a, self.mem_k.shape)

def create_raft_args():
    parser = argparse.ArgumentParser(description='RAFT options')
    parser.add_argument('--small', action='store_true', default=False, help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficient correlation implementation')
    
    # For demonstration, using an empty list; replace with sys.argv[1:] in production.
    args = parser.parse_args([])
    return args


class Endo3R(nn.Module):
    def __init__(self, 
                 dus3r_name="./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth", 
                 use_feat=False, 
                 mem_pos_enc=False, 
                 memory_dropout=0.15,
                 memory_config=None):
        super(Endo3R, self).__init__()
        # config
        self.use_feat = use_feat
        self.mem_pos_enc = mem_pos_enc

        # DUSt3R
        if memory_config is not None and not os.path.isfile(dus3r_name):
            raise FileNotFoundError(f'Configured architecture requires a local DUSt3R checkpoint: {dus3r_name}. '
                                    'No model-hub download is performed in this path.')
        self.dust3r = AsymmetricCroCo3DStereo.from_pretrained(dus3r_name, landscape_only=True)

        # Memory encoder
        self.set_memory_encoder(enc_embed_dim=768 if use_feat else 1024, memory_dropout=memory_dropout) 
        self.set_attn_head()
        # Optional architecture: default legacy construction/state_dict stays unchanged.
        # No FlashAttention/FLA import is performed for the official legacy mode.
        if memory_config is not None:
            self.configure_memory(memory_config)
        

    def configure_memory(self, memory_config):
        from dust3r.memory.config import load_config
        self.memory_config = load_config(memory_config)
        config = self.memory_config
        if config.memory.backend == 'legacy' and config.policy.long_write_policy == 'heuristic':
            return
        from dust3r.memory.policies import make_policy
        from dust3r.memory.uncertainty import UncertaintyEstimator
        actor = normalizer = None
        if config.policy.long_write_policy == 'ppo':
            from dust3r.rl.networks import Actor
            from dust3r.rl.normalization import ObservationNormalizer
            actor = Actor(config.rl.actor_state_dim, config.rl.actor_hidden)
            normalizer = ObservationNormalizer()
        self.long_write_policy = make_policy(config.policy.long_write_policy,
            threshold=config.policy.heuristic_similarity, actor=actor, normalizer=normalizer)
        if config.memory.backend == 'hybrid_flash_kda':
            from dust3r.memory.hybrid import HybridMemory
            from dust3r.memory.types import SharedNorms
            self.hybrid_memory = HybridMemory(config.memory,
                shared_norms=SharedNorms(self.norm_q, self.norm_k, self.norm_v))
            self.hybrid_uncertainty = UncertaintyEstimator(config.uncertainty, config.checkpoints.raft)
        else:
            # The legacy policy adapter retains original uncertainty/pruning behavior.
            # Disabled wrapper is only a runner dependency; it is never used to alter B0.
            from dataclasses import replace
            self.hybrid_uncertainty = UncertaintyEstimator(replace(config.uncertainty, enabled=False))

    def make_online_runner(self, training=False, policy=None):
        from dust3r.online.runner import create_online_runner
        return create_online_runner(self, self.memory_config, training=training, policy=policy)

    def set_memory_encoder(self, enc_depth=6, enc_embed_dim=1024, out_dim=1024, enc_num_heads=16, 
                           mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                           memory_dropout=0.15):
        
        self.value_encoder = nn.ModuleList([
            Block(enc_embed_dim, enc_num_heads, mlp_ratio, qkv_bias=True, 
                  norm_layer=norm_layer, rope=self.dust3r.rope if self.mem_pos_enc else None)
            for i in range(enc_depth)])
        
        self.value_norm = norm_layer(enc_embed_dim)
        self.value_out = nn.Linear(enc_embed_dim, out_dim)
        
        if not self.use_feat:
            self.pos_patch_embed = copy.deepcopy(self.dust3r.patch_embed)
            self.pos_patch_embed.load_state_dict(self.dust3r.patch_embed.state_dict())
        
        # Normalization layers
        self.norm_q = nn.LayerNorm(1024)
        self.norm_k = nn.LayerNorm(1024)
        self.norm_v = nn.LayerNorm(1024)
        self.mem_dropout = nn.Dropout(memory_dropout)
        
    def set_attn_head(self, enc_embed_dim=1024+768, out_dim=1024):
        self.attn_head_1 = nn.Sequential(
            nn.Linear(enc_embed_dim, enc_embed_dim),
            nn.GELU(),
            nn.Linear(enc_embed_dim, out_dim)
        )
        
        self.attn_head_2 = nn.Sequential(
            nn.Linear(enc_embed_dim, enc_embed_dim),
            nn.GELU(),
            nn.Linear(enc_embed_dim, out_dim)
        )

    def encode_image(self, view):
        img = view['img']
        B = img.shape[0]
        im_shape = view.get('true_shape', torch.tensor(img.shape[-2:])[None].repeat(B, 1))
        
        out, pos, _ = self.dust3r._encode_image(img, im_shape)
        
        return out, pos, im_shape
    
    def encode_image_pairs(self, view1, view2):
        img1 = view1['img']
        img2 = view2['img']

        B = img1.shape[0]

        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))
        
        
        out, pos, _ = self.dust3r._encode_image(torch.cat((img1, img2), dim=0),
                                                torch.cat((shape1, shape2), dim=0))
        out, out2 = out.chunk(2, dim=0)
        pos, pos2 = pos.chunk(2, dim=0)
        
        return out, out2, pos, pos2, shape1, shape2
    
    def encode_frames(self, view1, view2, feat1, feat2, pos1, pos2, shape1, shape2):
        if feat1 is None:
            feat1, feat2, pos1, pos2, shape1, shape2 = self.encode_image_pairs(view1, view2)
        
        else:
            feat1, pos1, shape1 = feat2, pos2, shape2
            feat2, pos2, shape2 = self.encode_image(view2)
        
        return feat1, feat2, pos1, pos2, shape1, shape2
    
    def encode_feat_key(self, feat1, feat2, num=1):
        feat = torch.cat((feat1, feat2), dim=-1)
        feat_k = getattr(self, f'attn_head_{num}')(feat)
        
        return feat_k
    
    def encode_value(self, x, pos):
        for block in self.value_encoder:
            x = block(x, pos)
        x = self.value_norm(x)
        x = self.value_out(x)
        return x
    
    def encode_cur_value(self, res1, dec1, pos1, shape1):
        if self.use_feat:
            cur_v = self.encode_value(dec1[-1], pos1)
                
        else:
            out, pos_v = self.pos_patch_embed(res1['pts3d'].permute(0, 3, 1, 2), true_shape=shape1)
            cur_v = self.encode_value(out, pos_v)
        
        return cur_v
    
    def decode(self, feat1, pos1, feat2, pos2):
        dec1, dec2 = self.dust3r._decoder(feat1, pos1, feat2, pos2)
        
        return dec1, dec2
    
    def downstream_head(self, dec, true_shape, num=1):
        with torch.cuda.amp.autocast(enabled=False):
            res = self.dust3r._downstream_head(num, [tok.float() for tok in dec], true_shape)

        return res
    
    def find_initial_pair(self, graph, n_frames):
        view1, view2, pred1, pred2 = graph['view1'], graph['view2'], graph['pred1'], graph['pred2']
        n_pairs = len(view1['idx'])

        conf_matrix = torch.zeros(n_frames, n_frames)


        for i in range(n_pairs):
            idx1, idx2 = view1['idx'][i], view2['idx'][i]

            conf1 = pred1['conf'][i]
            conf2 = pred2['conf'][i]

            conf1_sig = (conf1-1)/conf1
            conf2_sig = (conf2-1)/conf2

            conf  = conf1_sig.mean() + conf2_sig.mean()
            conf_matrix[idx1, idx2] = conf

        
        pair_idx = np.unravel_index(conf_matrix.argmax(), conf_matrix.shape)

        print(f'init pair:{pair_idx}, conf: {conf_matrix.max()}')

        return pair_idx
    
    def find_next_best_view(self, frames, idx_todo, feat_fuse, pos1, shape1):
        best_conf = 0.0
        from copy import deepcopy
        for i in idx_todo:
            view = frames[i]
            feat2, pos2, shape2 = self.encode_image(view)
            dec1, dec2 = self.decode(feat_fuse, pos1, feat2, pos2)
            res1 = self.downstream_head(dec1, shape1, 1)
            res2 = self.downstream_head(dec2, shape2, 2)

            conf1 = res1['conf']
            conf2 = res2['conf']

            conf1_sig = (conf1-1)/conf1
            conf2_sig = (conf2-1)/conf2

            
            
            total_conf_mean = conf1_sig.mean() + conf2_sig.mean()


            if total_conf_mean > best_conf:
                best_conf = total_conf_mean
                best_id = i
                best_dec1 = deepcopy(dec1)
                best_dec2 = deepcopy(dec2)
                best_res1 = deepcopy(res1)
                best_res2 = deepcopy(res2)
                best_feat2 = feat2
                best_pos2 = pos2
                best_shape2 = shape2
        

        return best_id, best_dec1, best_dec2, best_res1, best_res2, best_feat2, best_pos2, best_shape2, best_conf
    
    def forward(self, frames, return_memory=False, eval=False, uncertainty_check=True):
        config = getattr(self, 'memory_config', None)
        if config is not None and (config.memory.backend != 'legacy' or
                                   config.policy.long_write_policy != 'heuristic'):
            from dust3r.online.runner import run_episode
            return run_episode(self, frames, config, return_memory=return_memory)
        if self.training:
            sp_mem = SpatialTempralMemory(self.norm_q, self.norm_k, self.norm_v, mem_dropout=self.mem_dropout, attn_thresh=0)
        else:
            sp_mem = SpatialTempralMemory(self.norm_q, self.norm_k, self.norm_v)
        
        feat1, feat2, pos1, pos2, shape1, shape2 = None, None, None, None, None, None
        feat_k1, feat_k2 = None, None

        preds = None
        preds_all = []
        if not self.training:
            sp_mem.interval = max(10, len(frames)//10)
        for i in range(len(frames)):
            if i == len(frames)-1:
                break
            view1 = frames[i]
            view2 = frames[(i+1)]

            ##### Encode frames
            # feat1: [bs, p=196, c=1024]   
            feat1, feat2, pos1, pos2, shape1, shape2 = self.encode_frames(view1, view2, feat1, feat2, pos1, pos2, shape1, shape2)

            ##### Memory readout
            if feat_k2 is not None:
                feat_fuse = sp_mem.memory_read(feat_k2, res=True)
            
            else:
                feat_fuse = feat1
            
            ##### Decode features
            # dec1[-1]: [bs, p, c=768]
            dec1, dec2 = self.decode(feat_fuse, pos1, feat2, pos2)
            
            ##### Encode feat key
            # after decoder to perform cross attn between fusedfeat and dec last feat, build feat key from linear layer output
            feat_k1 = self.encode_feat_key(feat1, dec1[-1], 1)
            if torch.isnan(feat1).any() or torch.isnan(dec1[-1]).any():
                print("detect nan", torch.isnan(feat1).any(), torch.isnan(dec1[-1]).any(), torch.isnan(feat_fuse).any())
            feat_k2 = self.encode_feat_key(feat2, dec2[-1], 2)

            ##### Regress pointmaps
            with torch.cuda.amp.autocast(enabled=False):
                res1 = self.downstream_head(dec1, shape1, 1)
                res2 = self.downstream_head(dec2, shape2, 2)
            
            ##### Memory update
            cur_v = self.encode_cur_value(res1, dec1, pos1, shape1)

            if self.training:
                sp_mem.add_mem(feat_k1, cur_v+feat_k1)
            else:
                # print("processing id: ", i)

                if uncertainty_check and i > 0:
                    if len(sp_mem.spatial_buffer) > 0:
                        ref_index = sp_mem.spatial_buffer[-1]
                    else:
                        ref_index = i - 1
                    if ref_index > 0:
                        ref_pointmap = preds[ref_index]['pts3d_in_other_view']
                    else:
                        ref_pointmap = preds[ref_index]['pts3d']
                    sp_mem.add_mem_check(feat_k1, cur_v+feat_k1, index=i, check_uncertainty=uncertainty_check, ref_view=frames[ref_index], \
                                         cur_view = view1, ref_pointmap=ref_pointmap, cur_pts3d=res1['pts3d'], cur_conf=res1['conf'])
                else:
                    sp_mem.add_mem_check(feat_k1, cur_v+feat_k1, index=i, check_uncertainty=uncertainty_check, cur_conf=res1['conf'])
            
            res2['pts3d_in_other_view'] = res2.pop('pts3d')  
             
            if preds is None:
                preds = [res1]
                preds_all = [(res1, res2)]
            else:
                res1['pts3d_in_other_view'] = res1.pop('pts3d')
                preds.append(res1)
                preds_all.append((res1, res2))

                
                
                
        preds.append(res2)



        if return_memory:
            return preds, preds_all, sp_mem
        
        return preds, preds_all
    
    def pred_intrinsic(self, pred):
        _, H, W, _ = pred['pts3d'].shape
        pp = torch.tensor((W/2, H/2))
        focal = estimate_focal_knowing_depth(pred['pts3d'].cpu(), pp, focal_mode='weiszfeld')
        intrinsic = np.eye(3)
        intrinsic[0, 0] = focal
        intrinsic[1, 1] = focal
        intrinsic[:2, 2] = pp
        return torch.from_numpy(intrinsic).float().cuda()
    
    def pred_poses(self, pred, intrinsic):
        if 'pts3d' in pred:
            _, H, W, _ = pred['pts3d'].shape
            pts = pred['pts3d'].detach().cpu().numpy()[0]
        else:
            _, H, W, _ = pred['pts3d_in_other_view'].shape
            pts = pred['pts3d_in_other_view'].detach().cpu().numpy()[0]
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        points_2d = np.stack((u, v), axis=-1)
        dist_coeffs = np.zeros(4).astype(np.float32)
        success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
            pts.reshape(-1, 3).astype(np.float32), 
            points_2d.reshape(-1, 2).astype(np.float32), 
            intrinsic.astype(np.float32),
             dist_coeffs) #,iterationsCount=100, reprojectionError=2, flags=cv2.SOLVEPNP_SQPNP)
    
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # Extrinsic parameters (4x4 matrix)
        extrinsic_matrix = np.hstack((rotation_matrix, translation_vector.reshape(-1, 1)))
        extrinsic_matrix = np.vstack((extrinsic_matrix, [0, 0, 0, 1]))
        return torch.from_numpy(extrinsic_matrix).float().cuda(), w2c_to_c2w(torch.from_numpy(extrinsic_matrix)).float().cuda()


    