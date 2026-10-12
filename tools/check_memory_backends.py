"""Bounded backend conformance; external kernels require explicit --server-kernels.

SERVER_VALIDATION_REQUIRED: no server kernel is imported/built by the default
readiness command. --reference exercises pure PyTorch on CPU only.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def kernel_config_sha256(config):
    """Kernel math/layout/dtype contract; actor, RAFT, paths and seed excluded.

    Hybrid heuristic and PPO can share this proof because the probe exercises
    their identical memory modules, not reconstruction/uncertainty/policy code.
    """
    raw = config.to_dict() if hasattr(config, 'to_dict') else config
    training = raw.get('training', {})
    contract = {'memory': raw.get('memory', {}),
                'precision': {key: training.get(key) for key in
                              ('new_memory_dtype', 'pretrained_compute_dtype')}}
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def probe(config, *, reference=False, server_kernels=False, device='cpu'):
    import torch
    from dust3r.memory.config import FLA_REVISION, load_config
    requested = config.to_dict()
    requested_hash = hashlib.sha256(json.dumps(requested, sort_keys=True, separators=(',', ':'),
                                               allow_nan=False).encode('utf-8')).hexdigest()
    report = {'torch': str(torch.__version__), 'cuda_build': torch.version.cuda,
              'fla_present': importlib.util.find_spec('fla') is not None,
              'flash_attn_present': importlib.util.find_spec('flash_attn') is not None,
              'triton_present': importlib.util.find_spec('triton') is not None,
              'expected_fla_revision': FLA_REVISION, 'SERVER_VALIDATION_REQUIRED': True,
              'requested_config_sha256': requested_hash,
              'kernel_config_sha256': kernel_config_sha256(requested),
              'kernel_contract_scope': 'memory_config_and_memory_pretrained_activation_dtypes_only',
              'requested_device': str(device), 'device': str(device)}
    if reference and server_kernels:
        raise ValueError('Choose explicit CPU reference or server kernels, not both.')
    if not (reference or server_kernels):
        return report
    raw = config.to_dict()
    raw['uncertainty']['enabled'] = False
    if reference:
        raw['memory']['short'].update(backend='math_reference', require_flash=False)
        raw['memory']['long']['backend'] = 'torch_reference'
        raw['training']['new_memory_dtype'] = 'float32'
        device = 'cpu'
    elif not device.startswith('cuda'):
        raise ValueError('--server-kernels requires an explicit CUDA device.')
    config = load_config(raw)
    report['device'] = str(device)
    report['effective_probe_config_sha256'] = hashlib.sha256(json.dumps(
        config.to_dict(), sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()
    dtype = torch.float32 if reference else getattr(torch, config.training.new_memory_dtype)
    if server_kernels and dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError('Flash server conformance requires configured new_memory_dtype=float16 or bfloat16; '
                         'FP32 math reference does not prove the requested efficient backend.')
    from dust3r.memory.hybrid import HybridMemory
    from dust3r.memory.types import MemoryCandidate, WriteConfidence
    memory = HybridMemory(config.memory).to(device)
    d = config.memory.input_dim
    state = memory.reset(image_hw=(32, 32), patch_hw=(2, 2), device=device, training=True)
    key = torch.randn(1, 4, d, device=device)
    value = torch.randn_like(key)
    query = torch.randn_like(key)
    with torch.autocast(device_type=torch.device(device).type, dtype=dtype, enabled=not reference):
        state = memory.append_short(key, value, None, state)
        confidence = WriteConfidence(torch.ones(1, 4, 1, device=device), torch.zeros(1, 3, device=device))
        state, diagnostics = memory.write_long(MemoryCandidate(key, value), confidence, state)
        output, _ = memory.read(query, state)
        output.square().mean().backward()
    gradients = {name: parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                 for name, parameter in memory.named_parameters() if parameter.requires_grad}
    if not all(gradients.values()) or not torch.isfinite(output).all():
        raise RuntimeError('Backend forward/final-state gradient conformance failed: ' + str(gradients))
    report.update(output_shape=list(output.shape), state_shape=list(state.kda_state.shape),
                  finite_gradients=gradients, actual_short_backend=memory.short.last_backend,
                  actual_long_backend=memory.long.adapter.last_backend, reference_mode=reference,
                  passed=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/hybrid_memory.yaml')
    parser.add_argument('--reference', action='store_true')
    parser.add_argument('--server-kernels', action='store_true')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    from dust3r.memory.config import load_config
    print(json.dumps(probe(load_config(args.config), reference=args.reference,
                           server_kernels=args.server_kernels, device=args.device), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
