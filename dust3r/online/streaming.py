"""Frame-at-a-time runners; official legacy mathematics remain untouched.

The legacy adapter transcribes Endo3R.forward's *evaluation* ordering, including
its final-pair memory write and existing pruning/RAFT behavior. It only replaces
the retained input/output lists with the latest accepted reference. Numerical
conformance to the released forward is SERVER_VALIDATION_REQUIRED.
"""
from types import SimpleNamespace
from collections import OrderedDict
from collections.abc import Sequence

import torch


class DeviceSequence(Sequence):
    """Lazy frame/target view with at most two resident device items."""
    def __init__(self, sample, device, field="frame", max_cached=2):
        if field not in {"frame", "target"} or not 1 <= max_cached <= 2:
            raise ValueError("Expected frame/target field and cache size 1..2.")
        self.sample, self.device, self.field = sample, device, field
        self.max_cached = int(max_cached)
        self.cache = OrderedDict()

    def __len__(self):
        return self.sample.frame_count

    def __getitem__(self, index):
        if isinstance(index, slice):
            raise TypeError("DeviceSequence intentionally does not materialize slices.")
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        from dust3r.training.hybrid_trainer import move_tensors
        frame, target = self.sample.get_frame(index)
        value = move_tensors(frame if self.field == "frame" else target, self.device)
        self.cache[index] = value
        while len(self.cache) > self.max_cached:
            self.cache.popitem(last=False)
        return value


class LegacyOfficialStream:
    def __init__(self, model, frame_count, *, episode_id="sequence", uncertainty_check=True,
                 memory_factory=None):
        if model.training:
            raise ValueError("LEGACY_OFFICIAL streaming is inference only; call model.eval().")
        if frame_count < 2:
            raise ValueError("At least two frames are required.")
        if memory_factory is None:
            from dust3r.model import SpatialTempralMemory
            memory_factory = SpatialTempralMemory
        self.model = model
        self.frame_count = int(frame_count)
        self.episode_id = str(episode_id)
        self.uncertainty_check = bool(uncertainty_check)
        self.state = memory_factory(model.norm_q, model.norm_k, model.norm_v)
        self.state.interval = max(10, frame_count // 10)
        self.cache = (None,) * 6
        self.feat_k2 = None
        self.left = None
        self.reference_view = self.reference_points = None
        self.reference_index = -1
        self.previous_points = None
        self.index = 0
        self.last_commit_info = {}
        self.write_attempt_count = self.write_count = self.raft_call_count = 0

    @torch.no_grad()
    def step(self, next_frame, terminal=False):
        if self.left is None:
            self.left = next_frame
            return None
        i = self.index
        if i >= self.frame_count - 1 or bool(terminal) != (i == self.frame_count - 2):
            raise ValueError("Frame count/terminal marker differs from the declared sequence.")
        view1, view2 = self.left, next_frame
        feat1, feat2, pos1, pos2, shape1, shape2 = self.model.encode_frames(view1, view2, *self.cache)
        fused = self.state.memory_read(self.feat_k2, res=True) if self.feat_k2 is not None else feat1
        dec1, dec2 = self.model.decode(fused, pos1, feat2, pos2)
        feat_k1 = self.model.encode_feat_key(feat1, dec1[-1], 1)
        self.feat_k2 = self.model.encode_feat_key(feat2, dec2[-1], 2)
        # Released forward disables autocast for reconstruction heads.
        with torch.autocast(device_type=feat1.device.type, enabled=False):
            res1 = self.model.downstream_head(dec1, shape1, 1)
            res2 = self.model.downstream_head(dec2, shape2, 2)
        cur_v = self.model.encode_cur_value(res1, dec1, pos1, shape1)
        before = len(self.state.spatial_buffer)
        kwargs = dict(index=i, check_uncertainty=self.uncertainty_check, cur_conf=res1["conf"])
        if self.uncertainty_check and i > 0:
            ref_index = self.state.spatial_buffer[-1] if self.state.spatial_buffer else i - 1
            if ref_index == self.reference_index:
                ref_view, ref_points = self.reference_view, self.reference_points
            elif ref_index == i - 1:
                ref_view, ref_points = self.left_previous, self.previous_points
            else:
                raise RuntimeError("Released reference lookup could not be preserved by streaming.")
            kwargs.update(ref_view=ref_view, cur_view=view1, ref_pointmap=ref_points,
                          cur_pts3d=res1["pts3d"])
        self.state.add_mem_check(feat_k1, cur_v + feat_k1, **kwargs)
        accepted = len(self.state.spatial_buffer) > before
        raft_called = accepted and self.uncertainty_check and i > 0
        self.write_attempt_count += 1
        self.write_count += int(accepted)
        self.raft_call_count += int(raft_called)
        if accepted:
            self.reference_view, self.reference_points = view1, res1["pts3d"]
            self.reference_index = i
        self.left_previous, self.previous_points = view1, res1["pts3d"]
        self.last_commit_info = dict(requested_write=True, successful_write=accepted,
                                     raft_called=raft_called, kda_updated=False,
                                     candidate_index=i, arrival_index=i + 1,
                                     terminal_legacy_write=bool(terminal),
                                     write_semantics="legacy_similarity_checked_all_pairs")
        self.cache = feat1, feat2, pos1, pos2, shape1, shape2
        self.left = next_frame
        self.index += 1
        # Canonical pts3d aliases are intentionally retained for evaluators.
        return SimpleNamespace(res1=res1, res2=res2, candidate_index=i,
                               arrival_index=i + 1, terminal=bool(terminal))


def create_stream_runner(model, config, frame_count, episode_id="sequence"):
    if config.memory.backend == "legacy" and config.policy.long_write_policy == "heuristic":
        if config.baseline_identity != "LEGACY_OFFICIAL":
            raise ValueError("Unmodified legacy inference must be labeled LEGACY_OFFICIAL.")
        return LegacyOfficialStream(model, frame_count, episode_id=episode_id,
                                    uncertainty_check=config.uncertainty.enabled)
    runner = model.make_online_runner(training=False)
    runner.reset(episode_id=episode_id, training=False)
    return runner


def stream_predictions(runner, sample, device):
    """Yield one final prediction/GT pair, retaining no full-video GPU list."""
    from dust3r.training.hybrid_trainer import move_tensors
    for index in range(sample.frame_count):
        frame, _ = sample.get_frame(index)
        pair = runner.step(move_tensors(frame, device), terminal=index == sample.frame_count - 1)
        if pair is None:
            continue
        _, target = sample.get_frame(pair.candidate_index)
        yield pair.candidate_index, pair.res1, target
        if pair.terminal:
            _, target = sample.get_frame(pair.arrival_index)
            yield pair.arrival_index, pair.res2, target
