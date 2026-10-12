"""Exclusive component timing; actual profiling is explicitly server-only.

SERVER_VALIDATION_REQUIRED. Default command does not load models or run CUDA.
Measurements use CUDA synchronization; this tool is NOT a training reward loop.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import json
from pathlib import Path
import statistics
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class ComponentProfiler:
    def __init__(self, device):
        import torch
        self.device = torch.device(device)
        self.timings = defaultdict(list)
        self.stack = []
        self.originals = []
        self.pair_latencies = []
        self.pair_start = None

    def synchronize(self):
        if self.device.type == 'cuda':
            import torch
            torch.cuda.synchronize(self.device)

    @contextmanager
    def scope(self, name):
        self.synchronize()
        entry = [time.perf_counter(), 0.0]
        self.stack.append(entry)
        try:
            yield
        finally:
            self.synchronize()
            elapsed = time.perf_counter() - entry[0]
            self.stack.pop()
            self.timings[name].append(max(0., elapsed - entry[1]) * 1000)
            if self.stack:
                self.stack[-1][1] += elapsed

    def wrap(self, obj, method, name, start_pair=False, end_pair=False):
        if obj is None or not hasattr(obj, method):
            return
        original = getattr(obj, method)
        def instrumented(*args, **kwargs):
            if start_pair:
                self.synchronize()
                self.pair_start = time.perf_counter()
            with self.scope(name):
                result = original(*args, **kwargs)
            if end_pair and self.pair_start is not None:
                self.synchronize()
                self.pair_latencies.append((time.perf_counter() - self.pair_start) * 1000)
                self.pair_start = None
            return result
        self.originals.append((obj, method, original))
        setattr(obj, method, instrumented)

    def restore(self):
        for obj, method, original in reversed(self.originals):
            setattr(obj, method, original)

    def summary(self):
        def percentile(values, q):
            ordered = sorted(values)
            return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]
        return {name: {'calls': len(values), 'total_ms': sum(values), 'mean_ms': statistics.mean(values),
                       'p50_ms': percentile(values, .5), 'p95_ms': percentile(values, .95)}
                for name, values in self.timings.items() if values}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/hybrid_memory.yaml')
    parser.add_argument('--input')
    parser.add_argument('--hybrid-checkpoint')
    parser.add_argument('--actor-checkpoint')
    parser.add_argument('--trusted-checkpoints', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num-frames', type=int, default=8)
    parser.add_argument('--warmup-frames', type=int, default=3)
    parser.add_argument('--server-run', action='store_true')
    parser.add_argument('--output')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    if not args.server_run:
        print(json.dumps({'dry_run': True, 'SERVER_VALIDATION_REQUIRED': True,
                          'scopes': ['image_encoding', 'decoder', 'key_encoding', 'value_encoding',
                          'heads', 'short_read', 'long_read', 'fusion', 'observation', 'policy',
                          'short_update', 'long_update', 'uncertainty_geometry', 'raft'],
                          'no_downloads': True}, indent=2))
        return 0
    import torch
    from demo_hybrid import prepare_config, load_frames, load_inference_model
    from dust3r.training.hybrid_trainer import move_tensors
    config = prepare_config(args.config)
    from dust3r.training.reproducibility import seed_everything
    from dust3r.experiments.benchmark import runtime_metadata
    from dust3r.training.teacher import file_sha256
    seed_everything(args.seed)
    model, checkpoint_hashes = load_inference_model(config, device=args.device, hybrid_checkpoint=args.hybrid_checkpoint,
                                   actor_checkpoint=args.actor_checkpoint, trusted_checkpoints=args.trusted_checkpoints)
    frames = [move_tensors(frame, args.device) for frame in load_frames(args.input,
              (config.training.image_height, config.training.image_width), args.num_frames)]
    with torch.no_grad():
        model(frames[:max(2, min(args.warmup_frames, len(frames)))])
    profiler = ComponentProfiler(args.device)
    for method, name in [('encode_frames', 'image_encoding'), ('decode', 'decoder'),
                         ('encode_feat_key', 'key_encoding'), ('encode_cur_value', 'value_encoding'),
                         ('downstream_head', 'heads')]:
        profiler.wrap(model, method, name, start_pair=method == 'encode_frames')
    memory = getattr(model, 'hybrid_memory', None)
    if memory is not None:
        for obj, method, name in [(memory, 'read', 'memory_read'), (memory.short, 'read', 'short_read'),
              (memory.long, 'read', 'long_read'), (memory.fusion, 'forward', 'fusion'),
              (memory, 'append_short', 'short_update'), (memory.long, 'update', 'long_update')]:
            profiler.wrap(obj, method, name)
    uncertainty = getattr(model, 'hybrid_uncertainty', None)
    profiler.wrap(uncertainty, 'estimate', 'uncertainty_geometry')
    profiler.wrap(getattr(uncertainty, 'flow_model', None), 'forward', 'raft')
    policy = getattr(model, 'long_write_policy', None)
    profiler.wrap(policy, 'observe', 'observation')
    profiler.wrap(policy, 'act', 'policy')
    if memory is None:
        from dust3r.model import SpatialTempralMemory
        profiler.wrap(SpatialTempralMemory, 'memory_read', 'memory_read')
        profiler.wrap(SpatialTempralMemory, 'check_sim', 'similarity')
        profiler.wrap(SpatialTempralMemory, 'uncertainty_map', 'uncertainty_geometry')
        profiler.wrap(SpatialTempralMemory, 'memory_prune', 'legacy_prune')
        profiler.wrap(SpatialTempralMemory, '__init__', 'legacy_memory_construct')
        if config.policy.long_write_policy == 'heuristic':
            profiler.wrap(SpatialTempralMemory, 'add_mem_check', 'legacy_update', end_pair=True)
    if memory is not None or config.policy.long_write_policy != 'heuristic':
        from dust3r.online.runner import OnlineRunner
        profiler.wrap(OnlineRunner, 'commit', 'policy_and_memory_commit', end_pair=True)
    if profiler.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(profiler.device)
    try:
        profiler.synchronize()
        total_start = time.perf_counter()
        with torch.no_grad(), profiler.scope('total_sequence'):
            preds, pairs, state = model(frames, return_memory=True)
        profiler.synchronize()
        total_ms = (time.perf_counter() - total_start) * 1000
    finally:
        profiler.restore()
    report = {'config': config.to_dict(), 'gpu': str(profiler.device), 'input_frames': len(frames),
              'pair_count': len(pairs), 'output_count': len(preds), 'exclusive_component_timings': profiler.summary(),
              'write_count': getattr(state, 'write_count', None), 'raft_calls': getattr(state, 'raft_call_count', None),
              'kda_state_bytes': getattr(state, 'kda_state', torch.empty(0)).numel() * 4,
              'output_retention': 'all predictions on device, matched to compatibility forward'}
    report.update(schema='endo3r_component_profile_v1', hardware=runtime_metadata(args.device),
                  original_checkpoint_hashes=checkpoint_hashes, seed=args.seed,
                  hybrid_checkpoint_sha256=file_sha256(args.hybrid_checkpoint) if args.hybrid_checkpoint else None,
                  actor_checkpoint_sha256=file_sha256(args.actor_checkpoint) if args.actor_checkpoint else None,
                  actual_backends={'short': memory.short.last_backend if memory is not None else 'legacy_dense',
                                   'long': memory.long.adapter.last_backend if memory is not None else 'legacy_dense'},
                  timing_scope='synchronized_exclusive_components_not_headline_throughput',
                  component_operations={'policy': 'LongWritePolicy.act'},
                  image_hw=[config.training.image_height, config.training.image_width])
    from dust3r.training.provenance import source_provenance, runtime_backend_provenance
    report['source_provenance'] = source_provenance()
    report['runtime_backend_provenance'] = runtime_backend_provenance(model)
    latencies = sorted(profiler.pair_latencies)
    report.update(total_sequence_ms=total_ms, fps_input_frames=len(frames) * 1000 / total_ms,
                  average_pair_ms=statistics.mean(latencies) if latencies else None,
                  pair_p50_ms=statistics.median(latencies) if latencies else None,
                  pair_p95_ms=latencies[round(.95 * (len(latencies) - 1))] if latencies else None)
    if profiler.device.type == 'cuda':
        report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(profiler.device)
        report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved(profiler.device)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
