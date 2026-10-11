"""Publish small patterned input references for independent consumers."""
import argparse
from pathlib import Path

import torch
from timm.data import create_transform

from input_assets import json_bytes, preprocess, recipe, tensor_bytes, tensor_descriptor


def generate(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    images = {}
    for name, h, w in [('portrait', 29, 17), ('landscape', 19, 31), ('small', 7, 9)]:
        images[name] = torch.tensor([
            [[(x * 37 + y * 61 + c * 83 + x * y * 11) % 256 for x in range(w)]
             for y in range(h)] for c in range(3)], dtype=torch.uint8)
    (destination / 'images.pt').write_bytes(tensor_bytes(images))
    fixtures = []
    for mode in ['border', 'center', 'squash']:
        for interpolation in ['bicubic', 'bilinear']:
            for size in [[3, 17, 17], [3, 13, 21]]:
                name = f'{mode}-{interpolation}-{size[1]}x{size[2]}'
                config = {'input_size': size, 'crop_pct': 0.875, 'crop_mode': mode,
                          'interpolation': interpolation, 'mean': [0.485, 0.456, 0.406],
                          'std': [0.229, 0.224, 0.225]}
                transform = create_transform(**config, is_training=False)
                inputs, cases = {}, []
                for key, image in images.items():
                    inputs[key], geometry = preprocess(image, transform)
                    cases.append({'image': key, 'geometry': geometry,
                                  'tensor': tensor_descriptor(inputs[key], layout='CHW')})
                (destination / f'{name}-inputs.pt').write_bytes(tensor_bytes(inputs))
                fixtures.append({'name': name, 'preprocessing': {'recipe': recipe(config, transform)},
                                 'cases': cases})
    (destination / 'fixtures.json').write_bytes(json_bytes({'schema_version': 1, 'fixtures': fixtures}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    generate(parser.parse_args().output)
