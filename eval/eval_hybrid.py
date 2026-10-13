"""Validated common-frame reconstruction evaluation; no fabricated GT fallback.

SERVER_VALIDATION_REQUIRED: genuine calibrated sequence manifest and identified
local checkpoints. Pose evaluation is explicit and uses lazy local OpenCV.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def evaluate_predictions(predictions, targets, pose=False):
    import torch
    from dust3r.training.losses import EpisodeScale, pointmap_from_output
    if len(predictions) != len(targets) or not predictions:
        raise ValueError('Predictions must cover exactly the GT sequence frames.')
    if not all(target.get('provenance_verified') is True for target in targets):
        raise ValueError('Evaluation requires genuine calibrated GT provenance.')
    first = pointmap_from_output(predictions[0])
    scale = EpisodeScale.fit(first, targets[0]['pointmap'], targets[0]['valid_mask'])
    first_pose = targets[0]['camera_to_world'].float()
    frames = []
    for index, (output, target) in enumerate(zip(predictions, targets)):
        pred = pointmap_from_output(output).detach().float()
        gt = target['pointmap'].to(pred.device).float()
        mask = target['valid_mask'].to(pred.device).bool()
        if pred.shape != gt.shape or mask.shape != pred.shape[:-1] or not mask.any():
            raise ValueError('Evaluation targets must be aligned nonempty true GT masks.')
        if not torch.isfinite(pred).all() or not torch.isfinite(gt[mask]).all():
            raise FloatingPointError('Evaluation counts nonfinite predictions as failure, never masks them out.')
        aligned = pred * scale.scale.to(pred.device)
        error = (aligned[mask] - gt[mask]).abs().mean() / scale.radius.to(pred.device)
        entry = {'frame': index, 'point_l1_normalized': error.item(), 'valid_ratio': mask.float().mean().item()}
        pose_j = target['camera_to_world'].to(pred.device).float()
        common_to_camera = torch.linalg.solve(pose_j, first_pose.to(pred.device))
        pred_camera = torch.einsum('bhwc,bdc->bhwd', aligned, common_to_camera[:, :3, :3]) + common_to_camera[:, None, None, :3, 3]
        gt_camera = torch.einsum('bhwc,bdc->bhwd', gt, common_to_camera[:, :3, :3]) + common_to_camera[:, None, None, :3, 3]
        gt_depth = target.get('depth')
        gt_depth = gt_depth.to(pred.device).float() if gt_depth is not None else gt_camera[..., 2]
        dm = mask & torch.isfinite(gt_depth) & (gt_depth > 0)
        if not dm.any():
            raise ValueError('No real positive GT depth coverage for evaluation.')
        pd, gd = pred_camera[..., 2][dm], gt_depth[dm]
        diff = pd - gd
        ratio = torch.maximum(gd / pd.clamp_min(1e-6), pd / gd).clamp_min(1)
        entry.update(depth_abs_rel=(diff.abs() / gd).mean().item(),
                     depth_rmse=diff.square().mean().sqrt().item(),
                     depth_log_rmse=(pd.clamp_min(1e-6).log() - gd.log()).square().mean().sqrt().item(),
                     depth_delta1=(ratio < 1.25).float().mean().item(),
                     nonpositive_predicted_depth_ratio=(pd <= 0).float().mean().item())
        if pose:
            entry.update(evaluate_pose(aligned, target['intrinsics'], pose_j, first_pose))
        frames.append(entry)
    keys = [key for key in frames[0] if key != 'frame']
    aggregate = {}
    for key in keys:
        values = [item[key] for item in frames if isinstance(item.get(key), (int, float))]
        aggregate[key] = sum(values) / len(values) if values else None
    return {'scale_protocol': 'one_detached_first_frame_scale', 'first_scale': scale.scale.item(),
            'frame_metrics': frames, 'mean_metrics': aggregate, 'pose_uses_verified_intrinsics': bool(pose)}


def evaluate_pose(points, intrinsics, pose_j, first_pose):
    import numpy as np
    import torch
    try:
        import cv2
    except ImportError as exc:
        raise ImportError('Explicit pose evaluation requires local OpenCV; no package is installed automatically.') from exc
    _, h, w, _ = points.shape
    yy, xx = torch.meshgrid(torch.linspace(0, h - 1, min(16, h)).long(),
                            torch.linspace(0, w - 1, min(16, w)).long(), indexing='ij')
    cloud = points.detach().cpu()[0, yy, xx].reshape(-1, 3).numpy().astype(np.float64)
    pixels = torch.stack((xx, yy), -1).reshape(-1, 2).numpy().astype(np.float64)
    if cloud.shape[0] < 6:
        raise ValueError('Pose evaluation needs at least six spatial samples.')
    success, rotation, translation, _ = cv2.solvePnPRansac(cloud, pixels,
              intrinsics.detach().cpu()[0].numpy().astype(np.float64), np.zeros(4), iterationsCount=100)
    if not success:
        return {'pose_success': 0.0, 'pose_rotation_degrees': None, 'pose_translation_m': None}
    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(rotation)[0]
    transform[:3, 3] = translation[:, 0]
    predicted = torch.from_numpy(np.linalg.inv(transform)).float()
    gt = torch.linalg.solve(first_pose.cpu(), pose_j.cpu())[0]
    cosine = ((torch.trace(predicted[:3, :3].T @ gt[:3, :3]) - 1) / 2).clamp(-1, 1)
    return {'pose_success': 1.0, 'pose_rotation_degrees': torch.rad2deg(torch.acos(cosine)).item(),
            'pose_translation_m': (predicted[:3, 3] - gt[:3, 3]).norm().item()}


def _stream_frame_metrics(index, output, target, scale, first_pose, pose=False):
    """Same metric equations as evaluate_predictions, for one resident frame."""
    import torch
    from dust3r.training.losses import pointmap_from_output
    if target.get('provenance_verified') is not True:
        raise ValueError('Evaluation requires genuine calibrated GT provenance.')
    pred = pointmap_from_output(output).detach().float()
    gt = target['pointmap'].to(pred.device).float()
    mask = target['valid_mask'].to(pred.device).bool()
    if pred.shape != gt.shape or mask.shape != pred.shape[:-1] or not mask.any():
        raise ValueError('Evaluation targets must be aligned nonempty true GT masks.')
    if not torch.isfinite(pred).all() or not torch.isfinite(gt[mask]).all():
        raise FloatingPointError('Evaluation counts nonfinite predictions as failure, never masks them out.')
    aligned = pred * scale.scale.to(pred.device)
    error = (aligned[mask] - gt[mask]).abs().mean() / scale.radius.to(pred.device)
    entry = {'frame': index, 'point_l1_normalized': error.item(), 'valid_ratio': mask.float().mean().item()}
    pose_j = target['camera_to_world'].to(pred.device).float()
    common_to_camera = torch.linalg.solve(pose_j, first_pose.to(pred.device))
    pred_camera = torch.einsum('bhwc,bdc->bhwd', aligned, common_to_camera[:, :3, :3]) + common_to_camera[:, None, None, :3, 3]
    gt_camera = torch.einsum('bhwc,bdc->bhwd', gt, common_to_camera[:, :3, :3]) + common_to_camera[:, None, None, :3, 3]
    gt_depth = target.get('depth')
    gt_depth = gt_depth.to(pred.device).float() if gt_depth is not None else gt_camera[..., 2]
    dm = mask & torch.isfinite(gt_depth) & (gt_depth > 0)
    if not dm.any():
        raise ValueError('No real positive GT depth coverage for evaluation.')
    pd, gd = pred_camera[..., 2][dm], gt_depth[dm]
    diff = pd - gd
    ratio = torch.maximum(gd / pd.clamp_min(1e-6), pd / gd).clamp_min(1)
    entry.update(depth_abs_rel=(diff.abs() / gd).mean().item(),
                 depth_rmse=diff.square().mean().sqrt().item(),
                 depth_log_rmse=(pd.clamp_min(1e-6).log() - gd.log()).square().mean().sqrt().item(),
                 depth_delta1=(ratio < 1.25).float().mean().item(),
                 nonpositive_predicted_depth_ratio=(pd <= 0).float().mean().item())
    if pose:
        entry.update(evaluate_pose(aligned, target['intrinsics'], pose_j, first_pose))
    return entry


def evaluate_stream(sample, runner, device='cpu', pose=False):
    """GT-geometry certification with bounded frame/prediction tensor residency.

    Retains first-frame scale/pose and scalar per-frame metrics. Ground truth is
    supplied only to this evaluator, never to reconstruction or policy inputs.
    This remains the GT-pose/first-scale geometry protocol, distinct from the
    paper protocol. ``evaluate_predictions`` keeps its original list API.
    """
    import torch
    from dust3r.online.streaming import stream_predictions
    from dust3r.training.losses import EpisodeScale, pointmap_from_output
    frames, scale, first_pose = [], None, None
    with torch.no_grad():
        for index, output, target in stream_predictions(runner, sample, device):
            if index != len(frames):
                raise ValueError('Streaming predictions must cover every GT frame exactly once in order.')
            if target.get('provenance_verified') is not True:
                raise ValueError('Evaluation requires genuine calibrated GT provenance.')
            if scale is None:
                first = pointmap_from_output(output)
                scale = EpisodeScale.fit(first, target['pointmap'], target['valid_mask'])
                first_pose = target['camera_to_world'].detach().float().clone()
            frames.append(_stream_frame_metrics(index, output, target, scale, first_pose, pose))
    if not frames or len(frames) != sample.frame_count:
        raise ValueError('Predictions must cover exactly the GT sequence frames.')
    keys = [key for key in frames[0] if key != 'frame']
    aggregate = {}
    for key in keys:
        values = [item[key] for item in frames if isinstance(item.get(key), (int, float))]
        aggregate[key] = sum(values) / len(values) if values else None
    return {'scale_protocol': 'one_detached_first_frame_scale', 'first_scale': scale.scale.item(),
            'frame_metrics': frames, 'mean_metrics': aggregate, 'pose_uses_verified_intrinsics': bool(pose)}


def subset_fingerprint(coverage):
    """Bind episode ordering and exact selected source frame IDs, without GT."""
    encoded = json.dumps(coverage, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/hybrid_memory.yaml')
    parser.add_argument('--manifest')
    parser.add_argument('--split', default='validation')
    parser.add_argument('--hybrid-checkpoint')
    parser.add_argument('--actor-checkpoint')
    parser.add_argument('--trusted-checkpoints', action='store_true')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--reference', action='store_true')
    parser.add_argument('--pose', action='store_true')
    parser.add_argument('--max-frames', type=int, help='Optional >=2-frame prefix per sequence; recorded in coverage hash')
    parser.add_argument('--run-evaluation', action='store_true')
    parser.add_argument('--output')
    parser.add_argument('--baseline-report', help='Matching held-out LEGACY_OFFICIAL evaluation JSON')
    parser.add_argument('--certified-checkpoint-output', help='Explicit NEW checkpoint path; enables held-out PPO quality gate')
    args = parser.parse_args(argv)
    if args.max_frames is not None and args.max_frames < 2:
        parser.error('--max-frames must be at least 2')
    from demo_hybrid import prepare_config, load_inference_model
    config = prepare_config(args.config, args.reference)
    if not args.run_evaluation:
        print(json.dumps({'dry_run': True, 'SERVER_VALIDATION_REQUIRED': True,
                          'manifest': args.manifest or config.training.data_manifest,
                          'no_downloads': True}, indent=2))
        return 0
    import torch
    from dust3r.training.sequence_dataset import SequenceDataset
    from dust3r.online.streaming import create_stream_runner
    dataset = SequenceDataset(args.manifest or config.training.data_manifest, split=args.split,
                              image_hw=(config.training.image_height, config.training.image_width),
                              stride=config.training.stride)
    model, hashes = load_inference_model(config, device=args.device, hybrid_checkpoint=args.hybrid_checkpoint,
                                   actor_checkpoint=args.actor_checkpoint, trusted_checkpoints=args.trusted_checkpoints)
    results, coverage = [], []
    for index in range(len(dataset)):
        sample = dataset.stream(index)
        if args.max_frames is not None and sample.frame_count > args.max_frames:
            sample = sample.prefix(args.max_frames)
        runner = create_stream_runner(model, config, sample.frame_count, sample.episode_id)
        sequence_result = evaluate_stream(sample, runner, device=args.device, pose=args.pose)
        for frame in sequence_result['frame_metrics']:
            frame['source_frame_id'] = sample.source_frame_id(frame['frame'])
        results.append({'episode_id': sample.episode_id, **sequence_result})
        coverage.append({'episode_id': sample.episode_id,
                         'source_frame_ids': [sample.source_frame_id(i) for i in range(sample.frame_count)]})
        sample.clear_cache()
    from dust3r.training.teacher import file_sha256
    from dust3r.training.provenance import source_provenance
    memory = getattr(model, 'hybrid_memory', None)
    report = {'baseline_identity': config.baseline_identity, 'config': config.to_dict(), 'sequences': results,
              'split': args.split, 'manifest_sha256': file_sha256(args.manifest or config.training.data_manifest),
              'image_hw': [config.training.image_height, config.training.image_width],
              'max_frames': args.max_frames, 'subset_sha256': subset_fingerprint(coverage),
              'coverage': coverage, 'stride': config.training.stride,
              'source_provenance': source_provenance(),
              'evaluation_streaming': True,
              'scale_protocol': 'one_detached_first_frame_scale', 'original_checkpoint_hashes': hashes,
              'hybrid_checkpoint_sha256': file_sha256(args.hybrid_checkpoint) if args.hybrid_checkpoint else None,
              'actual_backends': {'short': memory.short.last_backend if memory is not None else 'legacy_dense',
                                  'long': memory.long.adapter.last_backend if memory is not None else 'legacy_dense'}}
    if args.certified_checkpoint_output:
        if not args.hybrid_checkpoint or not args.baseline_report:
            raise ValueError('Certification requires --hybrid-checkpoint and --baseline-report')
        from dust3r.training.validation import certify_for_ppo
        baseline = json.loads(Path(args.baseline_report).read_text(encoding='utf-8'))
        report['ppo_certification'] = certify_for_ppo(args.hybrid_checkpoint, report, baseline,
                                                     args.certified_checkpoint_output,
                                                     config.evaluation.quality_relative_tolerance)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
