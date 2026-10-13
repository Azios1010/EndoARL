"""Read-only held-out checkpoint ranking and explicit PPO certification.

No model construction, optimization, CUDA execution, or network access occurs
here. Scores are recomputed from reviewed validation reports: an arithmetic
mean over all evaluated frames, so longer sequences contribute more frames.
The existing per-sequence quality gate is also enforced before ranking.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import pickle

from .hybrid_trainer import CHECKPOINT_SCHEMA
from .teacher import file_sha256
from .validation import certify_for_ppo, validate_legacy_baseline_report


SELECTION_SCHEMA = 'endo3r_validation_checkpoint_selection_v1'
SCALE_PROTOCOL = 'one_detached_first_frame_scale'
METRICS = ('point_l1_normalized', 'depth_abs_rel')


def _is_sha256(value):
    return (isinstance(value, str) and len(value) == 64 and
            all(char in '0123456789abcdef' for char in value))


def _read_report(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Local validation report missing: {path}')
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError('Validation report must be a JSON object')
    return path, data, file_sha256(path)


def _finite_nonnegative(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


def _failure_fields(value, location):
    for key in ('errors', 'failures', 'failed_sequences', 'failed_frames'):
        if value.get(key):
            raise ValueError(f'Reported evaluation failure at {location}: {key}')
    if value.get('status') in {'failed', 'incomplete', 'error', 'blocked'}:
        raise ValueError(f'Incomplete/failed evaluation at {location}')
    if value.get('coverage_complete') is False:
        raise ValueError(f'Incomplete evaluation coverage at {location}')


def _report_contract(report, *, allow_bounded_validation, require_source_provenance):
    if report.get('split') != 'validation':
        raise ValueError('Only validation reports may select a checkpoint; train/test is forbidden')
    if report.get('scale_protocol') != SCALE_PROTOCOL:
        raise ValueError('Selection requires the declared common-first-camera/first-frame-scale certification protocol')
    if not _is_sha256(report.get('manifest_sha256')):
        raise ValueError('Missing/invalid validation GT manifest SHA256')
    image_hw = report.get('image_hw')
    if (not isinstance(image_hw, list) or len(image_hw) != 2 or
            any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in image_hw)):
        raise ValueError('Missing/invalid evaluation image_hw')
    if 'max_frames' not in report or not _is_sha256(report.get('subset_sha256')):
        raise ValueError('Exact max_frames/subset_sha256 evaluation coverage is required')
    bounded = report['max_frames']
    if bounded is not None:
        if not isinstance(bounded, int) or isinstance(bounded, bool) or bounded < 2:
            raise ValueError('Invalid max_frames evaluation limit')
        if not allow_bounded_validation:
            raise ValueError('Bounded-prefix validation cannot select by default; evaluate full validation or explicitly allow bounded validation')
    provenance = report.get('source_provenance', {})
    if not isinstance(provenance, dict):
        raise ValueError('Invalid evaluation source_provenance object')
    source = provenance.get('source_tree_sha256')
    if source is not None and not _is_sha256(source):
        raise ValueError('Invalid evaluation source tree SHA256')
    if require_source_provenance and source is None:
        raise ValueError('Missing source_provenance.source_tree_sha256; regenerate identified validation reports')
    originals = report.get('original_checkpoint_hashes', {})
    if not isinstance(originals, dict):
        raise ValueError('Invalid released original_checkpoint_hashes object')
    for key in ('dust3r', 'endo3r'):
        if not _is_sha256(originals.get(key)):
            raise ValueError(f'Missing/invalid released {key} checkpoint SHA256')
    _failure_fields(report, 'report')
    sequences = report.get('sequences')
    if not isinstance(sequences, list) or not sequences:
        raise ValueError('Validation coverage must contain nonempty held-out sequences')
    episodes, total_frames = {}, 0
    for item in sequences:
        if not isinstance(item, dict):
            raise ValueError('Validation sequence must be an object')
        episode_id = item.get('episode_id')
        if not isinstance(episode_id, str) or not episode_id or episode_id in episodes:
            raise ValueError('Missing/duplicate validation episode_id')
        _failure_fields(item, episode_id)
        frames = item.get('frame_metrics')
        if not isinstance(frames, list) or not frames:
            raise ValueError(f'Missing nonempty frame metrics: {episode_id}')
        if any(not isinstance(frame, dict) for frame in frames):
            raise ValueError(f'Per-frame metrics must be objects: {episode_id}')
        if [frame.get('frame') for frame in frames] != list(range(len(frames))):
            raise ValueError(f'Frame IDs must cover the sequence once in order: {episode_id}')
        for frame in frames:
            _failure_fields(frame, f'{episode_id}/{frame["frame"]}')
            for key in METRICS:
                if not _finite_nonnegative(frame.get(key)):
                    raise ValueError(f'Invalid/missing finite {key}: {episode_id}/{frame["frame"]}')
            if 'pose_success' in frame and frame['pose_success'] != 1:
                raise ValueError(f'Pose failure may not be discarded: {episode_id}/{frame["frame"]}')
            if 'valid_ratio' in frame and (not _finite_nonnegative(frame['valid_ratio']) or not 0 < frame['valid_ratio'] <= 1):
                raise ValueError(f'Invalid real GT coverage: {episode_id}/{frame["frame"]}')
        source_ids = item.get('source_frame_ids')
        if source_ids is not None and (not isinstance(source_ids, list) or len(source_ids) != len(frames)):
            raise ValueError(f'Source frame coverage length mismatch: {episode_id}')
        episodes[episode_id] = item
        total_frames += len(frames)
    return episodes, total_frames


def _same_validation_contract(current, baseline):
    for key in ('split', 'manifest_sha256', 'image_hw', 'scale_protocol', 'max_frames', 'subset_sha256'):
        if current.get(key) != baseline.get(key):
            raise ValueError(f'Validation protocol/coverage mismatch: {key}')
    for key in ('dust3r', 'endo3r'):
        if current['original_checkpoint_hashes'][key] != baseline['original_checkpoint_hashes'][key]:
            raise ValueError(f'Released checkpoint mismatch: {key}')
    a = current.get('source_provenance', {}).get('source_tree_sha256')
    b = baseline.get('source_provenance', {}).get('source_tree_sha256')
    if a != b:
        raise ValueError('Evaluation source_tree_sha256 mismatch')
    # Extra protocol fields, if emitted, may not silently differ or disappear.
    for key in ('protocol', 'preprocessing', 'pointmap_coordinate_system', 'evaluation_streaming', 'config_stride'):
        if key in current or key in baseline:
            if key not in current or key not in baseline or current[key] != baseline[key]:
                raise ValueError(f'Validation protocol mismatch: {key}')


def _candidate_quality(current, baseline, tolerance):
    hybrid = {item['episode_id']: item for item in current['sequences']}
    official = {item['episode_id']: item for item in baseline['sequences']}
    if set(hybrid) != set(official):
        raise ValueError('Validation episode coverage differs from official baseline')
    totals = {key: 0.0 for key in METRICS}
    count, comparisons = 0, []
    for episode in sorted(hybrid):
        frames, base_frames = hybrid[episode]['frame_metrics'], official[episode]['frame_metrics']
        if len(frames) != len(base_frames):
            raise ValueError(f'Validation frame coverage differs: {episode}')
        if hybrid[episode].get('source_frame_ids') != official[episode].get('source_frame_ids'):
            raise ValueError(f'Validation source frame IDs differ: {episode}')
        for a, b in zip(frames, base_frames):
            if a.get('valid_ratio') != b.get('valid_ratio'):
                raise ValueError(f'Validation GT coverage differs: {episode}/{a["frame"]}')
        count += len(frames)
        for metric in METRICS:
            value = sum(frame[metric] for frame in frames) / len(frames)
            base = sum(frame[metric] for frame in base_frames) / len(base_frames)
            limit = base * (1 + tolerance) + 1e-8  # existing certify_for_ppo gate
            comparisons.append({'episode_id': episode, 'metric': metric,
                                'hybrid': value, 'baseline': base, 'limit': limit})
            if value > limit:
                raise ValueError(f'Held-out quality gate failed: {episode}/{metric}={value} > {limit}')
            totals[metric] += sum(frame[metric] for frame in frames)
    return {key: totals[key] / count for key in METRICS}, comparisons, count


def select_hybrid_checkpoint(candidates, baseline_report, *, tolerance=0.05,
                             min_training_steps=None, ppo_manifest=None,
                             allow_bounded_validation=False,
                             require_source_provenance=True):
    """Inspect/rank ``{'checkpoint': path, 'report': path, 'label': optional}``.

    No file is written. Invalid candidates remain visible with errors. The
    optional minimum step budget is an operator requirement, not a convergence
    claim. Full validation is required unless bounded selection is explicitly
    acknowledged. Certification still uses the unchanged existing gate.

    ``ppo_manifest`` binds the exact manifest expected by the current PPO
    trainer. A train-only filtered manifest differs from a full validation
    manifest; the utility reports the mismatch instead of relabeling GT.
    """
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Quality tolerance must be finite and nonnegative')
    if min_training_steps is not None and (not isinstance(min_training_steps, int) or isinstance(min_training_steps, bool) or min_training_steps < 1):
        raise ValueError('min_training_steps must be a positive explicit budget')
    if not candidates:
        raise ValueError('At least one explicitly identified validation candidate is required')
    baseline_path, baseline, baseline_sha = _read_report(baseline_report)
    result = {'schema': SELECTION_SCHEMA, 'status': 'BLOCKED',
              'selection_split': 'validation', 'quality_tolerance': tolerance,
              'score_protocol': 'frame_uniform_arithmetic_mean_point_l1_normalized_then_depth_abs_rel',
              'minimum_training_steps': min_training_steps,
              'minimum_steps_is_not_convergence_proof': True,
              'baseline_report': str(baseline_path), 'baseline_report_sha256': baseline_sha,
              'source_tree_sha256': (baseline.get('source_provenance') or {}).get('source_tree_sha256')
              if isinstance(baseline.get('source_provenance', {}), dict) else None,
              'coverage_scope': 'full_declared_validation' if baseline.get('max_frames') is None else 'explicit_bounded_prefix',
              'max_frames': baseline.get('max_frames'), 'subset_sha256': baseline.get('subset_sha256'),
              'manifest_sha256': baseline.get('manifest_sha256'),
              'ppo_manifest_sha256': None, 'baseline_errors': [], 'candidates': [],
              'selected_candidate': None, 'writes_performed': False,
              'notes': ['This selects among supplied reviewed reports; it does not prove training convergence.',
                        'Certification/PPO currently bind the exact GT manifest SHA256. A separately filtered training manifest cannot be substituted.',
                        'Scores are certification geometry metrics, not a claim of paper-exact depth/pose evaluation.']}
    try:
        _report_contract(baseline, allow_bounded_validation=allow_bounded_validation,
                         require_source_provenance=require_source_provenance)
        validate_legacy_baseline_report(baseline, baseline['original_checkpoint_hashes']['endo3r'], baseline['manifest_sha256'])
        if ppo_manifest is not None:
            expected = file_sha256(ppo_manifest)
            result['ppo_manifest_sha256'] = expected
            if expected != baseline['manifest_sha256']:
                raise ValueError('PPO manifest SHA256 differs from validation manifest: current trainer cannot accept this certification; preserve/report the manifest split contract')
    except (ValueError, KeyError, TypeError, OSError) as error:
        result['baseline_errors'].append(str(error))
    seen = set()
    for index, candidate in enumerate(candidates):
        record = {'label': candidate.get('label', f'candidate_{index}'),
                  'checkpoint': str(Path(candidate['checkpoint']).resolve()),
                  'report': str(Path(candidate['report']).resolve()),
                  'eligible': False, 'errors': [], 'checkpoint_sha256': None,
                  'report_sha256': None, 'global_step': None, 'score': None,
                  'frame_count': None, 'comparisons': []}
        result['candidates'].append(record)
        try:
            if result['baseline_errors']:
                raise ValueError('Baseline/manifest contract failed; candidate cannot be selected')
            path = Path(record['checkpoint'])
            if not path.is_file():
                raise FileNotFoundError(f'Local hybrid checkpoint missing: {path}')
            record['checkpoint_sha256'] = file_sha256(path)
            if record['checkpoint_sha256'] in seen:
                raise ValueError('Duplicate checkpoint candidate SHA256')
            seen.add(record['checkpoint_sha256'])
            _, report, record['report_sha256'] = _read_report(record['report'])
            _report_contract(report, allow_bounded_validation=allow_bounded_validation,
                             require_source_provenance=require_source_provenance)
            if report.get('baseline_identity') != 'NEW_HYBRID':
                raise ValueError('Candidate evaluation must identify NEW_HYBRID')
            _same_validation_contract(report, baseline)
            if report.get('hybrid_checkpoint_sha256') != record['checkpoint_sha256']:
                raise ValueError('Missing/unmatched hybrid checkpoint SHA256 in validation report')
            actual = report.get('actual_backends', {})
            if (actual.get('short') not in {'sdpa_flash', 'flash_attn'} or
                    not str(actual.get('long', '')).startswith('fla_kda_')):
                raise ValueError('SERVER_VALIDATION_REQUIRED: actual Flash/FLA kernels required; reference/fallback reports cannot certify')
            import torch
            payload = torch.load(path, map_location='cpu', weights_only=True)
            if not isinstance(payload, dict):
                raise ValueError('Hybrid checkpoint payload must be a dictionary')
            if (payload.get('schema') != CHECKPOINT_SCHEMA or
                    payload.get('config', {}).get('memory', {}).get('backend') != 'hybrid_flash_kda'):
                raise ValueError('Candidate is not an explicit hybrid training checkpoint')
            steps = payload.get('global_step')
            record['global_step'] = steps
            if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
                raise ValueError('Candidate must identify positive completed training steps')
            if min_training_steps is not None and steps < min_training_steps:
                raise ValueError(f'Completed steps {steps} below declared budget {min_training_steps}; smoke checkpoint excluded')
            originals = payload.get('original_checkpoint_hashes', {})
            for key in ('dust3r', 'endo3r'):
                if originals.get(key) != report['original_checkpoint_hashes'][key]:
                    raise ValueError(f'Checkpoint/report initialization source mismatch: {key}')
            checkpoint_source = payload.get('source_provenance', {}).get('source_tree_sha256')
            if checkpoint_source is not None and checkpoint_source != result['source_tree_sha256']:
                raise ValueError('Training checkpoint/evaluation source tree SHA256 mismatch; review/re-evaluate source version explicitly')
            record['score'], record['comparisons'], record['frame_count'] = _candidate_quality(report, baseline, tolerance)
            record['eligible'] = True
        except (ValueError, KeyError, TypeError, OSError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
            record['errors'].append(str(error))
    eligible = sorted((item for item in result['candidates'] if item['eligible']),
                      key=lambda item: (item['score']['point_l1_normalized'],
                                        item['score']['depth_abs_rel'], item['checkpoint_sha256']))
    for rank, record in enumerate(eligible, start=1):
        record['rank'] = rank
    if eligible:
        result['status'] = 'SELECTED_NOT_CERTIFIED'
        result['selected_candidate'] = dict(eligible[0])
    return result


def write_checkpoint_selection(selection, output_checkpoint, output_report):
    """Explicitly write two NEW artifacts; refuses overwrite and stale inputs."""
    selected = selection.get('selected_candidate')
    if selection.get('status') != 'SELECTED_NOT_CERTIFIED' or not selected:
        raise ValueError('No eligible validation-selected checkpoint to certify')
    checkpoint, report_path = Path(output_checkpoint).resolve(), Path(output_report).resolve()
    if checkpoint == report_path:
        raise ValueError('Checkpoint and selection JSON output paths must be different')
    for path in (checkpoint, report_path):
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite existing artifact: {path}')
    protected = {Path(selection['baseline_report']).resolve()}
    for candidate in selection['candidates']:
        protected.update((Path(candidate['checkpoint']).resolve(), Path(candidate['report']).resolve()))
    if checkpoint in protected or report_path in protected:
        raise ValueError('Output must not alter any source report/checkpoint')
    if file_sha256(selected['checkpoint']) != selected['checkpoint_sha256']:
        raise ValueError('Selected checkpoint changed after inspection')
    _, hybrid, report_hash = _read_report(selected['report'])
    _, baseline, baseline_hash = _read_report(selection['baseline_report'])
    if report_hash != selected['report_sha256'] or baseline_hash != selection['baseline_report_sha256']:
        raise ValueError('Validation reports changed after inspection')
    summary = certify_for_ppo(selected['checkpoint'], hybrid, baseline, checkpoint,
                              selection['quality_tolerance'])
    # Retain the selected evaluation JSON contract consumed by study PPO
    # gates. Its hybrid_checkpoint_sha256 remains the evaluated SOURCE hash;
    # the separately certified artifact has its own explicit output hash.
    saved = dict(hybrid)
    saved.update(selection)
    saved.update(status='CERTIFIED', writes_performed=True,
                 certified_checkpoint=str(checkpoint), certified_checkpoint_sha256=file_sha256(checkpoint),
                 certification=summary, ppo_certification=summary,
                 checkpoint_selection=dict(selection))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects pre-existing reports, including a race.
    with report_path.open('x', encoding='utf-8') as handle:
        json.dump(saved, handle, indent=2, allow_nan=False)
        handle.write('\n')
    return saved
