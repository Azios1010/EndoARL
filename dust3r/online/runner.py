"""Causal prepare/read/decode then commit/write execution for Endo3R.

No model, FlashAttention, FLA or RAFT dependency is imported here. A model
implementing the existing Endo3R encode/decode interfaces can be injected.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import torch
from torch.nn import functional as F

from dust3r.memory.config import load_config
from dust3r.memory.types import MemoryCandidate, PolicyDecision, ReadDiagnostics, WriteDiagnostics


class ReconstructionFailure(FloatingPointError):
    """A model/state numerical failure, distinct from missing supervision."""


@dataclass
class PairResult:
    res1: dict
    res2: dict
    feat_k1: torch.Tensor
    candidate: MemoryCandidate
    candidate_index: int
    arrival_index: int
    read_diagnostics: ReadDiagnostics
    memory_feature: torch.Tensor
    terminal: bool = False
    decision: Optional[PolicyDecision] = None
    write_diagnostics: Optional[WriteDiagnostics] = None


def _finite(value, label):
    if not torch.isfinite(value).all():
        raise ReconstructionFailure(f'Nonfinite {label}; no memory update is allowed.')


class OnlineRunner:
    def __init__(self, model, memory=None, policy=None, uncertainty=None, config=None, training=False):
        self.model = model
        self.config = load_config(config or getattr(model, 'memory_config', None))
        self.memory = memory if memory is not None else getattr(model, 'hybrid_memory', None)
        self.policy = policy if policy is not None else getattr(model, 'long_write_policy', None)
        self.uncertainty = uncertainty if uncertainty is not None else getattr(model, 'hybrid_uncertainty', None)
        if self.config.memory.backend == 'legacy' and self.memory is None:
            from dust3r.memory.legacy_adapter import LegacyMemoryAdapter
            self.memory = LegacyMemoryAdapter(model, self.config)
        if self.memory is None or self.policy is None or self.uncertainty is None:
            raise ValueError('OnlineRunner requires memory, write policy and uncertainty modules.')
        self.training = training
        for name in ('actor', 'critic', 'normalizer'):
            module = getattr(self.policy, name, None)
            if module is not None:
                parameter = next(self.model.parameters(), None)
                if parameter is not None:
                    module.to(parameter.device)
        self.patch_size = self._patch_size()
        self.reset()

    def _patch_size(self):
        embed = getattr(getattr(self.model, 'dust3r', self.model), 'patch_embed', None)
        size = getattr(embed, 'patch_size', None)
        if size is None:
            size = getattr(self.model, 'patch_size', None)
        if size is None:
            raise ValueError('The checkpoint/model must expose its actual pixel patch_size.')
        return (size, size) if isinstance(size, int) else tuple(int(v) for v in size)

    def reset(self, frames=None, episode_id='sequence', training=None, image_hw=None):
        if training is not None:
            self.training = bool(training)
        if frames is not None and len(frames) < 2:
            raise ValueError('Online reconstruction requires at least two frames.')
        self.frames = frames
        self.episode_id = episode_id
        self.state = None
        self.cache = SimpleNamespace(feat1=None, feat2=None, pos1=None, pos2=None,
                                     shape1=None, shape2=None, feat_k2=None)
        self._pair_index = 0
        self._prepared = None
        self._stream_left = None
        self.last_commit_info = {}
        self.last_decision = None
        if hasattr(self.policy, 'reset'):
            self.policy.reset()
        if hasattr(self.uncertainty, 'reset'):
            self.uncertainty.reset()
        return self

    def _memory_autocast(self, device):
        name = self.config.training.new_memory_dtype
        if name == 'float32':
            return nullcontext()
        return torch.autocast(device_type=device.type, dtype=getattr(torch, name))

    def _ensure_state(self, left, right, feat):
        h, w = left['img'].shape[-2:]
        if right['img'].shape[-2:] != (h, w):
            raise ValueError('Adjacent frames must have the same cropped image size.')
        ph, pw = self.patch_size
        if h % ph or w % pw:
            raise ValueError('Cropped images must be divisible by checkpoint patch_size.')
        grid = (h // ph, w // pw)
        if feat.shape[0] != 1 or feat.shape[1] != grid[0] * grid[1]:
            raise ValueError('v1 requires B=1 and encoder row-major tokens matching the image grid.')
        if self.state is None:
            self.state = self.memory.reset(batch_size=1, image_hw=(h, w), patch_hw=grid,
                                           device=feat.device, training=self.training,
                                           sequence_id=self.episode_id)
        elif self.state.image_hw != (h, w) or self.state.patch_hw != grid:
            raise ValueError('Resolution/patch layout changed within an episode; reset is required.')
        self.state.current_step = self._pair_index
        self.state.candidate_index = self._pair_index
        self.state.arrival_index = self._pair_index + 1

    @staticmethod
    def _rgb(view, pointmap):
        if 'origin_img' not in view:
            raise ValueError('The runner requires cropped origin_img RGB; it never guesses normalization.')
        rgb = torch.as_tensor(view['origin_img'], device=pointmap.device)
        if rgb.ndim == 3:
            rgb = rgb.unsqueeze(0)
        if rgb.shape[-1] == 3:
            rgb = rgb.permute(0, 3, 1, 2)
        if rgb.shape != (pointmap.shape[0], 3, *pointmap.shape[1:3]):
            raise ValueError('origin_img must match the cropped pointmap dimensions.')
        if not torch.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 255:
            raise ValueError('origin_img must contain finite RGB values in [0,255].')
        return rgb.to(torch.uint8).contiguous()

    def _prepare_pair(self, left, right, terminal):
        if self._prepared is not None:
            raise RuntimeError('Commit the prepared pair before preparing another pair.')
        c = self.cache
        feat1, feat2, pos1, pos2, shape1, shape2 = self.model.encode_frames(
            left, right, c.feat1, c.feat2, c.pos1, c.pos2, c.shape1, c.shape2)
        self._ensure_state(left, right, feat1)
        _finite(feat1, 'encoder features')
        _finite(feat2, 'encoder features')
        if c.feat_k2 is None:
            fused = feat1
            zeros = feat1.new_zeros(feat1.shape[0])
            diag = ReadDiagnostics(*(zeros.clone() for _ in range(5)), has_short=False, has_long=False)
        else:
            with self._memory_autocast(feat1.device):
                fused, diag = self.memory.read(c.feat_k2, self.state)
            fused = fused.to(feat1.dtype)
        _finite(fused, 'memory read')
        dec1, dec2 = self.model.decode(fused, pos1, feat2, pos2)
        res1 = self.model.downstream_head(dec1, shape1, 1)
        res2 = self.model.downstream_head(dec2, shape2, 2)
        for label, result in (('res1', res1), ('res2', res2)):
            if 'pts3d' not in result or 'conf' not in result:
                raise ValueError('Endo3R heads must return pts3d and conf before output renaming.')
            _finite(result['pts3d'], label + ' pointmap before value encoding')
            _finite(result['conf'], label + ' confidence')
        feat_k1 = self.model.encode_feat_key(feat1, dec1[-1], 1)
        feat_k2 = self.model.encode_feat_key(feat2, dec2[-1], 2)
        cur_v = self.model.encode_cur_value(res1, dec1, pos1, shape1)
        for label, value in (('key1', feat_k1), ('key2', feat_k2), ('value', cur_v)):
            _finite(value, label)
        value = cur_v + feat_k1
        _finite(value, 'residual value')
        descriptor = F.normalize(feat_k1.float().mean(1), dim=-1, eps=1e-6).detach()
        points = res1['pts3d']
        candidate = MemoryCandidate(key=feat_k1, value=value, descriptor=descriptor,
                                    image=self._rgb(left, points), pointmap=points,
                                    valid_mask=torch.isfinite(points).all(-1), confidence=res1['conf'],
                                    candidate_index=self._pair_index, arrival_index=self._pair_index + 1)
        self.cache = SimpleNamespace(feat1=feat1, feat2=feat2, pos1=pos1, pos2=pos2,
                                     shape1=shape1, shape2=shape2, feat_k2=feat_k2)
        self._prepared = PairResult(res1, res2, feat_k1, candidate, self._pair_index,
                                    self._pair_index + 1, diag, fused, bool(terminal))
        return self._prepared

    def prepare_next(self):
        if self.frames is None:
            raise RuntimeError('prepare_next requires reset(frames); use step for streaming frames.')
        if self._pair_index >= len(self.frames) - 1:
            return None
        i = self._pair_index
        return self._prepare_pair(self.frames[i], self.frames[i + 1], i == len(self.frames) - 2)

    def observation(self):
        if self._prepared is None or self._prepared.terminal:
            raise RuntimeError('No eligible prepared policy decision.')
        return self.policy.observe(self._prepared.candidate, self.state, self._prepared.read_diagnostics)

    def _reference(self):
        s = self.state
        if not s.has_long_memory:
            return None
        if s.last_long_reference_image is None or s.last_long_reference_pointmap is None:
            raise RuntimeError('Initialized long state has no accepted reference; reset required.')
        return SimpleNamespace(image=s.last_long_reference_image,
                               pointmap=s.last_long_reference_pointmap,
                               valid_mask=s.last_long_reference_valid_mask)

    def _commit_reference(self, candidate, confidence):
        s = self.state
        s.last_long_descriptor = candidate.descriptor.detach().clone()
        s.has_last_long_descriptor = True
        limit = self.config.policy.heuristic_recent_long_descriptors
        s.recent_long_descriptors = torch.cat((s.recent_long_descriptors,
                                               candidate.descriptor.detach().unsqueeze(1)), dim=1)[:, -limit:]
        s.last_long_reference_index = candidate.candidate_index
        s.last_long_reference_arrival_index = candidate.arrival_index
        s.last_long_reference_image = candidate.image.detach().clone()
        s.last_long_reference_pointmap = candidate.pointmap.detach().float().clone()
        s.last_long_reference_valid_mask = candidate.valid_mask.detach().clone()
        if confidence.raft_called:
            s.previous_uncertainty_stats = confidence.uncertainty_stats.detach().clone()
            s.has_previous_raft_quality = True
        s.last_write_step = s.current_step
        s.write_count += 1
        s.kda_update_count += 1

    def commit(self, action):
        pair = self._prepared
        if pair is None:
            raise RuntimeError('Prepare a pair before committing its memory action.')
        decision = action if isinstance(action, PolicyDecision) else None
        raw = decision.action if decision is not None else action
        if isinstance(raw, torch.Tensor):
            if raw.numel() != 1:
                raise ValueError('v1 expects one scalar action for B=1.')
            raw = int(raw.item())
        if pair.terminal:
            if raw is not None:
                raise ValueError('Terminal pair has no learned write action; pass None.')
        elif raw not in (0, 1):
            raise ValueError('Exactly TEMP_ONLY=0 or WRITE_LONG=1 is required.')
        elif not self.state.has_long_memory and raw != 1:
            raise ValueError('First eligible action is forced WRITE_LONG.')
        forced = not self.state.has_long_memory and not pair.terminal
        info = dict(requested_write=raw == 1, successful_write=False, raft_called=False,
                    kda_updated=False, geometry_called=False, forced=forced,
                    candidate_index=pair.candidate_index, arrival_index=pair.arrival_index)
        # Keep live flags for an absorbing numerical failure midway through a write.
        self.last_commit_info = info
        if self.config.memory.backend == 'legacy':
            diag = self.memory.commit(pair.candidate, raw, self.state, pair.terminal)
            info.update(self.memory.last_commit_info)
        else:
            with self._memory_autocast(pair.feat_k1.device):
                self.state = self.memory.append_short(pair.candidate.key, pair.candidate.value,
                                                      pair.candidate.descriptor, self.state)
            diag = WriteDiagnostics(False, False, 'terminal' if pair.terminal else 'TEMP_ONLY')
            if raw == 1:
                self.state.write_attempt_count += 1
                had_long = self.state.has_long_memory
                confidence = self.uncertainty.estimate(pair.candidate, self._reference(), self.patch_size)
                info['uncertainty_metric'] = self.config.uncertainty.metric
                info['uncertainty_diagnostics'] = confidence.diagnostics
                info['raft_called'] = confidence.raft_called
                info['geometry_called'] = had_long and self.uncertainty.enabled
                self.state.raft_call_count += int(confidence.raft_called)
                if confidence.has_any_positive_patch:
                    # Preserve the cost of an attempted kernel if its state becomes
                    # nonfinite and the environment enters an absorbing failure.
                    info['kda_updated'] = True
                    with self._memory_autocast(pair.feat_k1.device):
                        self.state, diag = self.memory.write_long(pair.candidate, confidence, self.state)
                    info['kda_updated'] = diag.update_called
                    _finite(self.state.kda_state, 'KDA final state')
                    if diag.success:
                        self._commit_reference(pair.candidate, confidence)
                else:
                    diag = WriteDiagnostics(False, False, 'zero_valid_correspondences')
                    if not had_long:
                        raise ReconstructionFailure('No valid patch can initialize long memory.')
                info['successful_write'] = diag.success
                info['kda_updated'] = diag.update_called
        pair.decision, pair.write_diagnostics = decision, diag
        self.last_decision, self.last_commit_info = decision, info
        self._pair_index += 1
        self._prepared = None
        return diag

    def step(self, next_frame, terminal=False):
        if self.frames is not None:
            raise RuntimeError('Use prepare_next/commit when reset with a frame sequence.')
        if self._stream_left is None:
            self._stream_left = next_frame
            return None
        pair = self._prepare_pair(self._stream_left, next_frame, terminal)
        if pair.terminal:
            self.commit(None)
        else:
            packet = self.observation()
            decision = self.policy.act(packet, action_mask=packet.action_mask,
                                       deterministic=(not self.training))
            self.commit(decision)
        self._stream_left = next_frame
        return pair

    def detach_state(self, refresh_short=True):
        if self.state is not None:
            with self._memory_autocast(self.state.short_k.device):
                self.state = self.memory.detach(self.state, reproject=refresh_short)
        for name, value in vars(self.cache).items():
            if isinstance(value, torch.Tensor):
                setattr(self.cache, name, value.detach())
        return self.state


def create_online_runner(model, config=None, training=False, policy=None):
    return OnlineRunner(model, policy=policy, config=config, training=training)


def run_episode(model, frames, config=None, return_memory=False, policy=None):
    runner = create_online_runner(model, config, training=False, policy=policy)
    runner.reset(frames)
    preds, pairs = [], []
    while True:
        pair = runner.prepare_next()
        if pair is None:
            break
        if pair.terminal:
            runner.commit(None)
        else:
            packet = runner.observation()
            decision = runner.policy.act(packet, action_mask=packet.action_mask, deterministic=True)
            runner.commit(decision)
        left, right = dict(pair.res1), dict(pair.res2)
        right['pts3d_in_other_view'] = right.pop('pts3d')
        if pair.candidate_index > 0:
            left['pts3d_in_other_view'] = left.pop('pts3d')
        preds.append(left)
        pairs.append((left, right))
    preds.append(pairs[-1][1])
    return (preds, pairs, runner.state) if return_memory else (preds, pairs)
