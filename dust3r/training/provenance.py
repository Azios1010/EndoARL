"""Frozen-environment binding, local source identity and safe RNG checkpoints.

No package installation, network access, model import or optional CUDA kernel
import occurs here. Source probes use read-only git commands. RNG restore
supports a fresh-episode resume, not a saved video/memory/rollout cursor.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, distribution, version
import json
from pathlib import Path
import random
import subprocess
import sys
from typing import Mapping


ENVIRONMENT_SCHEMA = 'endo3r_frozen_reconstruction_environment_v1'
RNG_SCHEMA = 'endo3r_rng_v1'


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def _plain_config(config):
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, Mapping):
        return {key: _plain_config(value) for key, value in config.items()}
    if isinstance(config, (list, tuple)):
        return [_plain_config(value) for value in config]
    return config


def environment_contract(config, original_checkpoint_hashes, hybrid_checkpoint_sha256=None):
    """Configuration/weight identity shared by heuristic and PPO reconstruction.

    Excludes output/manifest paths, experiment seed, optimizer/reward settings
    and actor/critic parameters. Those describe an experiment rather than the
    frozen reconstruction transition. Observation semantics are included;
    heuristic thresholds and policy type are deliberately excluded.
    """
    cfg = _plain_config(config)
    training = cfg.get('training', {})
    policy = cfg.get('policy', {})
    return {
        'schema': ENVIRONMENT_SCHEMA,
        'input': {'image_height': training.get('image_height', 256),
                  'image_width': training.get('image_width', 320),
                  'frame_stride': training.get('stride', 1),
                  'batch_size': training.get('batch_size', 1),
                  'preprocessing': 'center_crop_rgb_bilinear_gt_nearest_pixel_center_k_v1',
                  'patch_ordering': 'encoder_row_major_v1',
                  'pointmap_coordinate_system': 'common_first_camera_v1'},
        'memory': cfg.get('memory', {}),
        'uncertainty': cfg.get('uncertainty', {}),
        'precision': {'new_memory_dtype': training.get('new_memory_dtype', 'bfloat16'),
                      'pretrained_compute_dtype': training.get('pretrained_compute_dtype', 'float32')},
        'policy_interface': {
            'observation_schema': policy.get('observation_schema', 'actor30_v1'),
            'actions': policy.get('actions', ['temp_only', 'write_long']),
            'force_first_write': policy.get('force_first_write', True),
            'terminal_pair_action': policy.get('terminal_pair_action', 'none'),
            'recent_long_descriptors': policy.get('heuristic_recent_long_descriptors', 4)},
        'original_checkpoint_hashes': dict(original_checkpoint_hashes),
        'hybrid_checkpoint_sha256': hybrid_checkpoint_sha256,
    }


def environment_contract_sha256(config, original_checkpoint_hashes, hybrid_checkpoint_sha256=None):
    return canonical_sha256(environment_contract(config, original_checkpoint_hashes,
                                                hybrid_checkpoint_sha256))


def verify_actor_environment(payload, config, original_checkpoint_hashes,
                             hybrid_checkpoint_sha256=None):
    """Reject policies trained with different geometry/memory/weights at load."""
    stored = payload.get('environment_contract_sha256')
    if not isinstance(stored, str) or len(stored) != 64:
        raise ValueError('Actor checkpoint lacks a valid environment_contract_sha256. '
                         'Retrain or explicitly re-export a verified policy with the new '
                         'frozen-environment contract; old actors cannot certify Sampson compatibility.')
    expected = environment_contract(config, original_checkpoint_hashes, hybrid_checkpoint_sha256)
    expected_hash = canonical_sha256(expected)
    source = payload.get('environment_contract')
    if source is not None and canonical_sha256(source) != stored:
        raise ValueError('Actor checkpoint environment contract/hash is internally inconsistent.')
    if stored != expected_hash:
        detail = ''
        if isinstance(source, Mapping):
            previous = source.get('uncertainty', {}).get('metric', 'geometry_3d')
            current = expected['uncertainty'].get('metric', 'geometry_3d')
            if previous != current:
                detail = f' uncertainty.metric changed from {previous!r} to {current!r}.'
        raise ValueError('Actor frozen environment contract mismatch: memory configuration, '
                         'input/precision, uncertainty or reconstruction weights changed.' + detail +
                         ' Retrain the policy for this environment; no automatic compatibility override.')
    return expected_hash


def _source_file(path):
    source_suffixes = {'.py', '.yaml', '.yml', '.toml', '.cfg', '.ini', '.txt', '.md',
                       '.sh', '.ps1', '.bat', '.cu', '.cuh', '.cpp', '.h', '.hpp', '.json'}
    if path.suffix.lower() not in source_suffixes:
        return False
    parts = path.parts
    source_roots = {'dust3r', 'croco', 'eval', 'tools', 'configs', 'tests'}
    if len(parts) == 1:
        return path.suffix.lower() not in {'.json', '.txt'} or path.name.startswith('requirements')
    return parts[0] in source_roots


def source_provenance(root=None):
    """Actual HEAD and deterministic current source fingerprint, including edits.

    Hashes source/config/docs files rather than datasets/checkpoints/results.
    In a source archive without git, HEAD is explicitly unknown and the same
    bounded source directories provide the tree fingerprint. No git mutation.
    """
    root = Path(root or Path(__file__).resolve().parents[2]).resolve()
    commit, dirty, tracked_tree, submodule_status = None, None, None, None
    paths = None
    try:
        def git(*args):
            return subprocess.run(['git', *args], cwd=root, capture_output=True,
                                  check=True, timeout=10).stdout
        commit = git('rev-parse', 'HEAD').decode('utf-8').strip()
        tracked_tree = git('rev-parse', 'HEAD^{tree}').decode('utf-8').strip()
        dirty = bool(git('status', '--porcelain', '-z'))
        paths = [Path(item.decode('utf-8', errors='surrogateescape'))
                 for item in git('ls-files', '-z', '--cached', '--others', '--exclude-standard').split(b'\0')
                 if item]
        try:
            submodule_status = git('submodule', 'status', '--recursive').decode('utf-8').strip()
        except subprocess.SubprocessError:
            submodule_status = None
    except (OSError, subprocess.SubprocessError):
        paths = None
    if paths is None:
        paths = [path.relative_to(root) for path in root.iterdir() if path.is_file()]
        for name in ('dust3r', 'croco', 'eval', 'tools', 'configs', 'tests'):
            directory = root / name
            if directory.is_dir():
                paths.extend(path.relative_to(root) for path in directory.rglob('*') if path.is_file())
    else:
        # A checked-out CroCo submodule is a gitlink in the parent index. Hash
        # its actual sources too, including locally edited submodule files.
        croco = root / 'croco'
        if croco.is_dir():
            paths.extend(path.relative_to(root) for path in croco.rglob('*') if path.is_file())
    records = []
    for relative in sorted(set(paths), key=lambda value: value.as_posix()):
        if not _source_file(relative):
            continue
        path = root / relative
        try:
            path.resolve().relative_to(root)
        except ValueError:
            continue  # External symlinks are not part of this checkout.
        if not path.is_file():
            digest = 'deleted'
        else:
            sha = hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                    sha.update(chunk)
            digest = sha.hexdigest()
        records.append((relative.as_posix(), digest))
    return {'git_commit': commit, 'git_head_tree': tracked_tree, 'git_dirty': dirty,
            'source_tree_sha256': canonical_sha256({'files': records, 'submodules': submodule_status}),
            'source_file_count': len(records), 'git_submodule_status': submodule_status,
            'fingerprint_scope': 'source_configs_tests_and_top_level_docs_excludes_data_checkpoints_results'}


def runtime_backend_provenance(model=None):
    """Installed versions and observed backends, without importing FLA/Flash."""
    import torch
    result = {'torch': str(torch.__version__), 'torch_cuda_build': torch.version.cuda}
    for package in ('flash-attn', 'flash-linear-attention', 'triton', 'numpy',
                    'opencv-python', 'opencv-python-headless'):
        try:
            result[package] = str(version(package))
        except PackageNotFoundError:
            result[package] = None
    try:
        raw = distribution('flash-linear-attention').read_text('direct_url.json')
        result['fla_installed_vcs_revision'] = json.loads(raw).get('vcs_info', {}).get('commit_id') if raw else None
    except (PackageNotFoundError, ValueError, TypeError):
        result['fla_installed_vcs_revision'] = None
    memory = getattr(model, 'hybrid_memory', None)
    result['actual_backends'] = {
        'short': getattr(getattr(memory, 'short', None), 'last_backend', 'legacy_dense' if model else 'not_observed'),
        'long': getattr(getattr(getattr(memory, 'long', None), 'adapter', None),
                        'last_backend', 'legacy_dense' if model else 'not_observed')}
    return result


def capture_rng_state():
    """weights_only-safe torch states and Python/NumPy states as primitives."""
    import numpy as np
    import torch
    py = random.getstate()
    numpy = np.random.get_state()
    return {'schema': RNG_SCHEMA,
            'python': [int(py[0]), list(py[1]), py[2]],
            'numpy': [str(numpy[0]), numpy[1].tolist(), int(numpy[2]), int(numpy[3]), float(numpy[4])],
            'torch_cpu': torch.get_rng_state().cpu(),
            'torch_cuda': [state.cpu() for state in torch.cuda.get_rng_state_all()]
                          if torch.cuda.is_available() else []}


def restore_rng_state(state):
    """Restore generators only; video/KDA/rollout cursors must start fresh."""
    import numpy as np
    import torch
    if not isinstance(state, Mapping) or state.get('schema') != RNG_SCHEMA:
        raise ValueError('Missing compatible complete RNG state; old PPO checkpoints cannot resume reproducibly.')
    required = {'python', 'numpy', 'torch_cpu', 'torch_cuda'}
    if not required <= set(state):
        raise ValueError('RNG state must contain Python, NumPy, torch CPU and CUDA generators.')
    cuda = state['torch_cuda']
    if cuda:
        if not torch.cuda.is_available() or len(cuda) != torch.cuda.device_count():
            raise ValueError('Saved CUDA RNG device count differs; exact RNG restore requires the same CUDA devices.')
    py, numpy = state['python'], state['numpy']
    random.setstate((int(py[0]), tuple(py[1]), py[2]))
    np.random.set_state((str(numpy[0]), np.asarray(numpy[1], dtype=np.uint32),
                         int(numpy[2]), int(numpy[3]), float(numpy[4])))
    torch.set_rng_state(state['torch_cpu'].cpu())
    if cuda:
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda])
    return {'resume_semantics': 'fresh_episode_complete_rng_restore_no_video_or_rollout_cursor',
            'cuda_rng_devices_restored': len(cuda),
            'opencv_rng': 'not_serializable_reseed_at_fresh_episode'}


def seed_loaded_opencv(seed):
    """OpenCV has no exported RNG state getter; reseed only after it is loaded."""
    cv2 = sys.modules.get('cv2')
    if cv2 is not None:
        cv2.setRNGSeed(int(seed) % (2 ** 31 - 1))
