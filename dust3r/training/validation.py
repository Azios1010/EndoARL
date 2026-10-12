"""Explicit held-out quality certification; never alters source/checkpoint input.

SERVER_VALIDATION_REQUIRED: reports must come from identified real checkpoints
and verified patient-disjoint GT, not CPU toy/reference conformance tests.
"""
import hashlib
import json
import math
from pathlib import Path

import torch

from .hybrid_trainer import CHECKPOINT_SCHEMA
from .teacher import file_sha256


def validate_legacy_baseline_report(report, endo3r_hash, manifest_hash):
    """Require a finite identified held-out baseline before legacy-policy PPO."""
    if (report.get('baseline_identity') != 'LEGACY_OFFICIAL' or
            report.get('split') not in {'validation', 'test'} or
            report.get('original_checkpoint_hashes', {}).get('endo3r') != endo3r_hash or
            report.get('manifest_sha256') != manifest_hash):
        raise ValueError('Legacy PPO requires a matching identified held-out LEGACY_OFFICIAL report')
    if not report.get('sequences'):
        raise ValueError('Legacy baseline has no held-out evaluation coverage')
    for sequence in report['sequences']:
        if not sequence.get('frame_metrics'):
            raise ValueError('Legacy baseline must cover nonempty GT frames')
        for frame in sequence['frame_metrics']:
            for key in ('point_l1_normalized', 'depth_abs_rel'):
                if not isinstance(frame.get(key), (int, float)) or not math.isfinite(frame[key]) or frame[key] < 0:
                    raise ValueError('Legacy baseline quality is invalid; do not silently repair it for PPO')


def certify_for_ppo(checkpoint, hybrid_report, baseline_report, output, tolerance=0.05):
    """Save a NEW checkpoint after matching held-out official-baseline metrics.

    Reports bind source weights and GT manifest hashes. Gate normalized pointmap
    error and depth absolute-relative error per sequence; no failed poses/NaNs
    are discarded. This gate certifies reconstruction, not PPO policy quality.
    """
    checkpoint, output = Path(checkpoint), Path(output)
    if output.resolve() == checkpoint.resolve():
        raise ValueError('Certification output must be a different checkpoint path')
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Quality tolerance must be finite and nonnegative')
    if hybrid_report.get('baseline_identity') != 'NEW_HYBRID':
        raise ValueError('Certification requires NEW_HYBRID evaluation')
    if baseline_report.get('baseline_identity') != 'LEGACY_OFFICIAL':
        raise ValueError('Comparison must use the unchanged LEGACY_OFFICIAL baseline')
    if hybrid_report.get('split') not in {'validation', 'test'}:
        raise ValueError('Certification requires an explicitly held-out validation/test split')
    for field in ('split', 'manifest_sha256', 'image_hw', 'scale_protocol'):
        if not hybrid_report.get(field) or hybrid_report.get(field) != baseline_report.get(field):
            raise ValueError(f'Evaluation protocol mismatch or missing {field}')
    for field in ('max_frames', 'subset_sha256'):
        if field in hybrid_report or field in baseline_report:
            if field not in hybrid_report or field not in baseline_report or hybrid_report[field] != baseline_report[field]:
                raise ValueError(f'Evaluation coverage mismatch: {field}')
    if hybrid_report.get('hybrid_checkpoint_sha256') != file_sha256(checkpoint):
        raise ValueError('Hybrid report was not produced by this checkpoint')
    if not baseline_report.get('original_checkpoint_hashes', {}).get('endo3r'):
        raise ValueError('Official baseline must identify its released checkpoint')
    if hybrid_report.get('original_checkpoint_hashes'):
        for name in ('dust3r', 'endo3r'):
            current = hybrid_report['original_checkpoint_hashes'].get(name)
            reference = baseline_report['original_checkpoint_hashes'].get(name)
            if not current or current != reference:
                raise ValueError(f'Initial released checkpoint mismatch: {name}')
    actual = hybrid_report.get('actual_backends', {})
    if actual.get('short') not in {'sdpa_flash', 'flash_attn'} or not str(actual.get('long', '')).startswith('fla_kda_'):
        raise ValueError('SERVER_VALIDATION_REQUIRED: actual Flash and FLA kernels must be validated before PPO')
    hybrid = {item['episode_id']: item for item in hybrid_report.get('sequences', [])}
    baseline = {item['episode_id']: item for item in baseline_report.get('sequences', [])}
    if not hybrid or set(hybrid) != set(baseline):
        raise ValueError('Reports must cover exactly the same nonempty held-out sequences')
    comparisons = []
    for episode in sorted(hybrid):
        current, reference = hybrid[episode], baseline[episode]
        if len(current.get('frame_metrics', [])) != len(reference.get('frame_metrics', [])) or not current.get('frame_metrics'):
            raise ValueError('Matching nonempty per-frame evaluation coverage is required')
        for metric in ('point_l1_normalized', 'depth_abs_rel'):
            for report in (current, reference):
                if any(not isinstance(frame.get(metric), (int, float)) or
                       not math.isfinite(frame[metric]) or frame[metric] < 0
                       for frame in report['frame_metrics']):
                    raise ValueError(f'Invalid/missing per-frame metric: {metric}')
            value = sum(frame[metric] for frame in current['frame_metrics']) / len(current['frame_metrics'])
            base = sum(frame[metric] for frame in reference['frame_metrics']) / len(reference['frame_metrics'])
            limit = base * (1 + tolerance) + 1e-8
            if value > limit:
                raise ValueError(f'Held-out quality gate failed: {episode}/{metric}={value} > {limit}')
            comparisons.append({'episode_id': episode, 'metric': metric, 'hybrid': value, 'baseline': base})
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if payload.get('schema') != CHECKPOINT_SCHEMA or payload.get('config', {}).get('memory', {}).get('backend') != 'hybrid_flash_kda':
        raise ValueError('Only explicit hybrid training checkpoints can be certified')
    report_hash = hashlib.sha256(json.dumps(hybrid_report, sort_keys=True, allow_nan=False).encode()).hexdigest()
    baseline_hash = hashlib.sha256(json.dumps(baseline_report, sort_keys=True, allow_nan=False).encode()).hexdigest()
    payload['validation_summary'] = {'validated_for_ppo': True, 'quality_tolerance': tolerance,
                                    'source_checkpoint_sha256': file_sha256(checkpoint),
                                    'held_out_report_sha256': report_hash,
                                    'baseline_report_sha256': baseline_hash,
                                    'manifest_sha256': hybrid_report['manifest_sha256'],
                                    'comparisons': comparisons}
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return payload['validation_summary']
