"""Capture official teacher memory features without changing its algorithm.

SERVER_VALIDATION_REQUIRED: local legacy checkpoints/dependencies and a finite
teacher are required. Default command is dry; --run-cache explicitly opts in.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/hybrid_memory.yaml')
    parser.add_argument('--manifest')
    parser.add_argument('--output', default='teacher_traces')
    parser.add_argument('--max-frames', type=int, default=16)
    parser.add_argument('--max-samples', type=int, default=1)
    parser.add_argument('--split', default='train')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--trusted-checkpoints', action='store_true')
    parser.add_argument('--run-cache', action='store_true')
    args = parser.parse_args(argv)
    from dust3r.memory.config import load_config
    raw = load_config(args.config).to_dict()
    raw['memory']['backend'] = 'legacy'
    raw['baseline_identity'] = 'LEGACY_OFFICIAL'
    raw['policy']['long_write_policy'] = 'heuristic'
    config = load_config(raw)
    if not args.run_cache:
        print(json.dumps({'dry_run': True, 'SERVER_VALIDATION_REQUIRED': True,
                          'teacher': 'LEGACY_OFFICIAL', 'no_downloads': True}, indent=2))
        return 0
    if not 3 <= args.max_frames <= 128:
        raise ValueError('Teacher extraction is bounded to 3..128 frames per sequence.')
    from dust3r.training.sequence_dataset import SequenceDataset
    from dust3r.training.hybrid_trainer import load_initialized_model, move_tensors
    from dust3r.training.teacher import TeacherTraceCache, TeacherTraceMetadata, file_sha256
    import torch
    from unittest.mock import patch
    dataset = SequenceDataset(args.manifest or config.training.data_manifest,
              split=args.split, image_hw=(config.training.image_height, config.training.image_width),
              sequence_raw_frames=config.training.sequence_raw_frames, stride=config.training.stride)
    if args.max_frames < config.training.sequence_raw_frames or args.max_samples < 1:
        raise ValueError('Cache complete training clips: --max-frames >= sequence_raw_frames and --max-samples >=1.')
    if not Path('checkpoints/raft-things.pth').is_file():
        raise FileNotFoundError('Official teacher requires checkpoints/raft-things.pth; no download is performed.')
    model, hashes = load_initialized_model(config, args.device, args.trusted_checkpoints)
    model.eval()
    from dust3r.model import SpatialTempralMemory
    original_read = SpatialTempralMemory.memory_read
    preprocessing_hash = hashlib.sha256(json.dumps({'hw': [config.training.image_height,
          config.training.image_width], 'crop': 'joint_center_crop_v1', 'normalization': 'rgb/127.5-1'},
          sort_keys=True).encode()).hexdigest()
    checkpoint_hash = hashlib.sha256(json.dumps({**hashes, 'raft': file_sha256('checkpoints/raft-things.pth')},
                                                sort_keys=True).encode()).hexdigest()
    summary = []
    for index in range(min(len(dataset), args.max_samples)):
        sample = dataset[index]
        captured = []
        def spy(memory, query, *a, **kw):
            feature = original_read(memory, query, *a, **kw)
            if not torch.isfinite(feature).all():
                raise FloatingPointError('Official teacher memory feature is nonfinite; no baseline fix applied.')
            captured.append(feature.detach().float().cpu())
            return feature
        frames = [move_tensors(frame, args.device) for frame in sample.frames[:args.max_frames]]
        with torch.no_grad(), patch.object(SpatialTempralMemory, 'memory_read', spy):
            _, _, memory = model(frames, return_memory=True)
        trajectory_hash = hashlib.sha256(json.dumps(memory.spatial_buffer).encode()).hexdigest()
        metadata = TeacherTraceMetadata(sample.episode_id, checkpoint_hash, preprocessing_hash, trajectory_hash)
        cache = TeacherTraceCache(args.output, metadata)
        for candidate_index, feature in enumerate(captured, start=1):
            cache.save(candidate_index, candidate_index + 1, feature)
        summary.append({**vars(metadata), 'trace_count': len(captured)})
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'manifest.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
