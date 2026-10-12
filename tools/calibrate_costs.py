"""Convert an existing server component profile into measured PPO cost proxies.

No CUDA, models, data decoding, installations or training are performed here.
These synchronized component means are reward proxies, not headline FPS.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


BASE_COMPONENTS = ('image_encoding', 'decoder', 'key_encoding', 'value_encoding',
                   'heads', 'memory_read', 'short_read', 'long_read', 'fusion',
                   'short_update', 'observation', 'policy')


def calibration_from_profile(profile):
    if profile.get('schema') != 'endo3r_component_profile_v1':
        raise ValueError('An identified endo3r_component_profile_v1 server artifact is required')
    backends = profile.get('actual_backends', {})
    if backends.get('short') not in {'sdpa_flash', 'flash_attn'} or not str(backends.get('long', '')).startswith('fla_kda_'):
        raise ValueError('SERVER_VALIDATION_REQUIRED: cost calibration requires measured Flash/FLA kernels')
    hardware = profile.get('hardware', {}).get('hardware_id')
    image_hw = profile.get('image_hw')
    pairs = profile.get('pair_count', 0)
    if not hardware or not isinstance(image_hw, list) or len(image_hw) != 2 or any(v <= 0 for v in image_hw) or pairs <= 0:
        raise ValueError('Profile must identify hardware, image dimensions and completed pairs')
    timings = profile.get('exclusive_component_timings', {})

    def number(component, field):
        record = timings.get(component, {})
        value = record.get(field)
        if not record.get('calls') or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'Measured nonnegative {component}/{field} with calls is required; rerun a sufficiently long profile')
        return float(value)

    baseline = sum(number(name, 'total_ms') for name in BASE_COMPONENTS if name in timings) / pairs
    if baseline <= 0:
        raise ValueError('Measured action-independent per-pair baseline must be positive')
    uncertainty_enabled = profile.get('config', {}).get('uncertainty', {}).get('enabled', True)
    result = {'baseline': baseline, 'kda': number('long_update', 'mean_ms'),
              'raft': number('raft', 'mean_ms') if uncertainty_enabled else 0.,
              'geometry': number('uncertainty_geometry', 'mean_ms') if uncertainty_enabled else 0.,
              'hardware_id': hardware, 'image_hw': image_hw}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    source, output = Path(args.profile), Path(args.output)
    if source.resolve() == output.resolve() or output.exists():
        raise FileExistsError('Choose a fresh calibration output; source artifacts are never overwritten')
    raw = source.read_bytes()
    profile = json.loads(raw)
    result = calibration_from_profile(profile)
    result.update(schema='endo3r_cost_calibration_v1', units='milliseconds',
                  source_profile_sha256=hashlib.sha256(raw).hexdigest(),
                  timing_scope=profile.get('timing_scope'),
                  uncertainty_metric=profile.get('config', {}).get('uncertainty', {}).get('metric'),
                  uncertainty_enabled=profile.get('config', {}).get('uncertainty', {}).get('enabled', True),
                  memory_config=profile.get('config', {}).get('memory'),
                  precision_config={key: profile.get('config', {}).get('training', {}).get(key)
                                    for key in ('new_memory_dtype', 'pretrained_compute_dtype')},
                  checkpoint_hashes=profile.get('original_checkpoint_hashes', {}),
                  hybrid_checkpoint_sha256=profile.get('hybrid_checkpoint_sha256'),
                  disclosure='Exclusive synchronized means are frozen PPO cost proxies; use benchmark_experiment.py for headline FPS.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
