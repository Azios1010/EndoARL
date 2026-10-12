"""Local-only, bounded inference entrypoint. No installation or download paths.

SERVER_VALIDATION_REQUIRED: real pretrained checkpoints, Flash/FLA and RAFT.
Use --dry-run for configuration/readiness; inference requires --run-inference.
"""
import argparse
import json
from pathlib import Path


def prepare_config(path, reference=False):
    from dust3r.memory.config import load_config
    config = load_config(path)
    if reference:
        values = config.to_dict()
        values['memory']['short'].update(backend='math_reference', require_flash=False)
        values['memory']['long']['backend'] = 'torch_reference'
        values['uncertainty']['enabled'] = False
        values['training']['new_memory_dtype'] = 'float32'
        config = load_config(values)
    return config


def load_frames(directory, image_hw=(256, 320), max_frames=8):
    """Explicit center crop/resize, matching GT loader's pixel footprint."""
    import numpy as np
    import torch
    from PIL import Image
    from torch.nn import functional as F
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f'Local RGB directory required: {directory}')
    files = sorted(path for path in directory.iterdir() if path.suffix.lower() in {'.png', '.jpg', '.jpeg'})
    if max_frames < 2 or max_frames > 128:
        raise ValueError('Bounded demo supports 2..128 frames per invocation.')
    files = files[:max_frames]
    if len(files) < 2:
        raise ValueError('At least two local RGB frames are required.')
    h, w = image_hw
    frames = []
    for index, path in enumerate(files):
        with Image.open(path) as image:
            rgb = torch.from_numpy(np.asarray(image.convert('RGB')).copy())
        ih, iw = rgb.shape[:2]
        ratio = w / h
        cw, ch = (max(1, round(ih * ratio)), ih) if iw / ih > ratio else (iw, max(1, round(iw / ratio)))
        left, top = (iw - cw) // 2, (ih - ch) // 2
        crop = rgb[top:top + ch, left:left + cw].permute(2, 0, 1)[None].float()
        crop = F.interpolate(crop, (h, w), mode='bilinear', align_corners=False).round().clamp(0, 255).byte()
        frames.append({'img': crop.float() / 127.5 - 1,
                       'origin_img': crop.permute(0, 2, 3, 1).contiguous(),
                       'true_shape': torch.tensor([[h, w]]), 'idx': index, 'instance': str(path)})
    return frames


def load_inference_model(config, *, device='cpu', hybrid_checkpoint=None,
                         actor_checkpoint=None, trusted_checkpoints=False, allow_untrained=False):
    import torch
    from dust3r.training.hybrid_trainer import load_initialized_model, load_training_checkpoint
    from dust3r.memory.config import load_config
    effective = config.to_dict()
    requested_policy = config.policy.long_write_policy
    if requested_policy == 'ppo':
        if actor_checkpoint is None or not Path(actor_checkpoint).is_file():
            raise FileNotFoundError('PPO inference requires a local --actor-checkpoint; no random policy fallback.')
        # Legacy + heuristic intentionally registers no new runner modules.
        # Construct the policy adapter's disabled uncertainty wrapper first.
        effective['policy']['long_write_policy'] = ('always' if config.memory.backend == 'legacy' else 'heuristic')
    if config.memory.backend == 'legacy':
        required = Path('checkpoints/raft-things.pth')
        if not required.is_file():
            raise FileNotFoundError(f'Legacy memory always loads its original local RAFT checkpoint: {required}')
    if config.memory.backend == 'hybrid_flash_kda' and not hybrid_checkpoint and not allow_untrained:
        raise FileNotFoundError('Supply --hybrid-checkpoint; --allow-untrained-memory is explicit initialization debugging only.')
    if hybrid_checkpoint and not Path(hybrid_checkpoint).is_file():
        raise FileNotFoundError(f'Local hybrid checkpoint missing: {hybrid_checkpoint}')
    model, hashes = load_initialized_model(load_config(effective), device, trusted_checkpoints)
    if hybrid_checkpoint:
        load_training_checkpoint(hybrid_checkpoint, model)
    if requested_policy == 'ppo':
        from dust3r.memory.policies import PPOWritePolicy
        from dust3r.rl.networks import Actor
        from dust3r.rl.normalization import ObservationNormalizer
        from dust3r.training.teacher import file_sha256
        from dust3r.training.provenance import verify_actor_environment
        payload = torch.load(actor_checkpoint, map_location='cpu', weights_only=True)
        if payload.get('schema') != 'endo3r_ppo_write_v1' or payload.get('observation_schema') != 'actor30_v1':
            raise ValueError('Actor checkpoint schema/observation interface is incompatible.')
        if config.memory.backend == 'legacy' and payload.get('original_checkpoint_hashes', {}).get('endo3r') != hashes.get('endo3r'):
            raise ValueError('Legacy actor was trained with a different released Endo3R checkpoint.')
        contract = verify_actor_environment(payload, config, hashes,
                    file_sha256(hybrid_checkpoint) if hybrid_checkpoint else None)
        actor = Actor(30, config.rl.actor_hidden).to(device)
        actor.load_state_dict(payload['actor'], strict=True)
        normalizer = ObservationNormalizer().to(device)
        normalizer.load_state_dict(payload['normalizer'], strict=True)
        normalizer.freeze()
        actor.eval()
        model.long_write_policy = PPOWritePolicy(actor, normalizer=normalizer)
        model.actor_environment_contract_sha256 = contract
        model.actor_source_provenance = payload.get('source_provenance')
        # Reuse registered modules, change dispatch config only (no reinitialization).
        model.memory_config = config
    model.eval()
    return model, hashes


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/hybrid_memory.yaml')
    parser.add_argument('--input')
    parser.add_argument('--max-frames', type=int, default=8)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--hybrid-checkpoint')
    parser.add_argument('--actor-checkpoint')
    parser.add_argument('--trusted-checkpoints', action='store_true')
    parser.add_argument('--allow-untrained-memory', action='store_true')
    parser.add_argument('--reference', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--run-inference', action='store_true')
    parser.add_argument('--output', help='Optional local prediction .pt file')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = prepare_config(args.config, args.reference)
    from dust3r.training.hybrid_trainer import validate_required_paths
    readiness = [item for item in validate_required_paths(config) if not item.startswith('Verified GT')]
    report = {'backend': config.memory.backend, 'short_backend': config.memory.short.backend,
              'long_backend': config.memory.long.backend, 'policy': config.policy.long_write_policy,
              'reference_mode': args.reference, 'blocking_requirements': readiness,
              'SERVER_VALIDATION_REQUIRED': True, 'no_downloads': True}
    if args.dry_run or not args.run_inference:
        print(json.dumps(report, indent=2))
        return 0
    if not args.input:
        raise ValueError('--input is required for explicitly requested inference.')
    if readiness:
        raise FileNotFoundError('\n'.join(readiness))
    import torch
    from dust3r.training.hybrid_trainer import move_tensors
    model, hashes = load_inference_model(config, device=args.device, hybrid_checkpoint=args.hybrid_checkpoint,
                      actor_checkpoint=args.actor_checkpoint, trusted_checkpoints=args.trusted_checkpoints,
                      allow_untrained=args.allow_untrained_memory)
    frames = [move_tensors(frame, args.device) for frame in load_frames(args.input,
              (config.training.image_height, config.training.image_width), args.max_frames)]
    with torch.no_grad():
        preds, pairs, state = model(frames, return_memory=True, uncertainty_check=config.uncertainty.enabled)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'predictions': move_tensors(preds, 'cpu'), 'config': config.to_dict(),
                    'checkpoint_hashes': hashes}, output)
    print(json.dumps({**report, 'output_frames': len(preds), 'pair_count': len(pairs),
                      'write_count': getattr(state, 'write_count', None),
                      'raft_calls': getattr(state, 'raft_call_count', None)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
