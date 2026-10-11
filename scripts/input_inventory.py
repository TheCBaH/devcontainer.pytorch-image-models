#!/usr/bin/env python3
"""Measure the release images and resolved recipes without fetching weights."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from PIL import Image
from timm.data import create_transform, resolve_data_config
from timm.models import get_pretrained_cfg
import PIL
import timm
import torch
import torchvision

from exportlib import MAX_RES, resolved_input_size
from pt2_export_core.archive import load_manifest, release_names


def inventory(data_dir, manifest, max_res):
    images = {}
    for path in sorted((Path(data_dir) / 'images').glob('*.jpg')):
        with Image.open(path) as source:
            rgb = source.convert('RGB')
            images[path.name] = {'shape': [3, rgb.height, rgb.width],
                                 'jpeg_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    recipes = {}
    for name in release_names(load_manifest(manifest)):
        cfg = get_pretrained_cfg(name).to_dict()
        config = resolve_data_config({'input_size': tuple(resolved_input_size(cfg, max_res))},
                                     pretrained_cfg=cfg)
        key = json.dumps(config, sort_keys=True)
        recipes.setdefault(key, {'config': config, 'pipeline': repr(create_transform(
            **config, is_training=False)), 'models': []})['models'].append(name)
    return {'producer_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'timm_commit': subprocess.check_output(
                ['git', '-C', 'modules/pytorch-image-models', 'rev-parse', 'HEAD'], text=True).strip(),
            'versions': {'timm': timm.__version__, 'torch': torch.__version__,
                         'torchvision': torchvision.__version__, 'Pillow': PIL.__version__},
            'images': images, 'decoded_raw_bytes': sum(
                3 * x['shape'][1] * x['shape'][2] for x in images.values()),
            'recipes': list(recipes.values())}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--manifest', default='models-selected.yaml')
    parser.add_argument('--max-res', type=int, default=MAX_RES)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    Path(args.output).write_text(json.dumps(inventory(args.data_dir, args.manifest, args.max_res),
                                          indent=2, sort_keys=True) + '\n')
