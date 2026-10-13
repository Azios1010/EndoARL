"""Separate training supervision selection from the full evaluation manifest.

No filtering occurs here. A derived manifest must be explicitly produced by the
coverage audit and bound to the unchanged original source. Evaluation and PPO
certification continue to identify the original manifest, including held-out
frames. These helpers perform only local metadata/file reads.
"""
from pathlib import Path

from .teacher import file_sha256


def training_data_contract(config):
    training = config.training
    source = training.data_manifest
    if not source or not Path(source).is_file():
        raise FileNotFoundError(f'Full verified source manifest required: {source!r}')
    selected = training.supervision_manifest or source
    if not Path(selected).is_file():
        raise FileNotFoundError(f'Training supervision manifest required: {selected!r}')
    contract = {
        'schema': 'endo3r_training_data_contract_v1',
        'source_manifest_sha256': file_sha256(source),
        'supervision_manifest_sha256': file_sha256(selected),
        'derived_supervision': bool(training.supervision_manifest),
        'image_hw': [training.image_height, training.image_width],
        'stride': training.stride,
        'sequence_raw_frames': training.sequence_raw_frames,
        'tbptt_pairs': training.tbptt_pairs,
    }
    if training.supervision_manifest:
        from dust3r.data.supervision import validate_supervision_manifest
        contract['selection'] = validate_supervision_manifest(
            selected, source, contract['image_hw'], stride=training.stride,
            clip_frames=training.sequence_raw_frames)
    return contract


def make_training_dataset(config, *, sequence_raw_frames=None):
    """Return only TRAIN data; optional clip length, full valid runs for PPO."""
    from .sequence_dataset import SequenceDataset
    contract = training_data_contract(config)
    training = config.training
    dataset = SequenceDataset(training.supervision_manifest or training.data_manifest,
                              split='train',
                              image_hw=(training.image_height, training.image_width),
                              sequence_raw_frames=sequence_raw_frames,
                              stride=training.stride)
    return dataset, contract
