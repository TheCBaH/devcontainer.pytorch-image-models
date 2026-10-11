"""Decoded RGB assets, canonical tensor bytes and the pinned PIL input pipeline."""
import hashlib
import io
import json
import math
import platform
import subprocess
from pathlib import Path
from unittest.mock import patch

from pt2_export_core import opgraph, schema_validate
from pt2_export_core.archive import (IMAGES_ARCHIVE_PROFILE, EVALUATION_ARCHIVE_PROFILE,
                                     TENSOR_ARCHIVE_PROFILE, safe_zip_open)
from pt2_export_core.manifest import sha256_file

SCHEMAS = str(Path(__file__).resolve().parents[1] / 'schemas')
TENSOR_PROFILE = TENSOR_ARCHIVE_PROFILE
EVALUATION_PROFILE = EVALUATION_ARCHIVE_PROFILE
EVALUATION_MEMBERS = ['cases.json', 'contract-v2.json', 'expected.json',
                      'outputs.pt', 'preprocessing.json']


def json_bytes(document):
    return (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def canonical_array_bytes(array):
    if array.dtype.kind == 'u' and array.dtype.itemsize == 1:
        return array.tobytes(order='C')
    if array.dtype.kind == 'f' and array.dtype.itemsize == 4:
        return array.astype('<f4', copy=False).tobytes(order='C')
    raise ValueError(f'unsupported tensor dtype: {array.dtype}')


def tensor_descriptor(tensor, *, layout=None):
    import torch
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu' or tensor.layout != torch.strided:
        raise ValueError('expected a strided CPU tensor')
    if tensor.dtype not in (torch.uint8, torch.float32):
        raise ValueError(f'unsupported tensor dtype: {tensor.dtype}')
    if tensor.numel() == 0 or tensor.numel() * tensor.element_size() > TENSOR_PROFILE.max_member_bytes:
        raise ValueError('tensor size outside supported limits')
    if tensor.dtype == torch.float32 and not torch.isfinite(tensor).all():
        raise ValueError('nonfinite tensor values')
    raw = canonical_array_bytes(tensor.detach().contiguous().numpy())
    descriptor = {'dtype': str(tensor.dtype).removeprefix('torch.'), 'shape': list(tensor.shape),
                  'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
                  'byte_order': 'little', 'layout': layout or 'row_major'}
    return descriptor


def tensor_bytes(tensors):
    import torch
    for tensor in tensors.values():
        tensor_descriptor(tensor)
    buffer = io.BytesIO()
    torch.save(tensors, buffer)
    return buffer.getvalue()


def load_tensors(source, *, expected_sha256=None, descriptors=None):
    import torch
    if not isinstance(source, bytes) and Path(source).stat().st_size > TENSOR_PROFILE.max_archive_bytes:
        raise ValueError('tensor file exceeds archive byte limit')
    data = source if isinstance(source, bytes) else Path(source).read_bytes()
    if expected_sha256 is not None and hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError('tensor file sha256 mismatch')
    with safe_zip_open(data, profile=TENSOR_PROFILE) as archive:
        archive._with_validated_source(lambda handle, _: None)
    tensors = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
    if (not isinstance(tensors, dict) or not tensors or len(tensors) > 512
            or not all(isinstance(k, str) and k for k in tensors)):
        raise ValueError('expected a nonempty dictionary of named tensors')
    if descriptors is not None and set(tensors) != set(descriptors):
        raise ValueError('tensor keys disagree with descriptors')
    total = 0
    for key, tensor in tensors.items():
        desc = tensor_descriptor(tensor, layout=descriptors[key]['layout'] if descriptors else None)
        total += desc['bytes']
        if descriptors is not None and desc != descriptors[key]:
            raise ValueError(f'{key}: tensor descriptor disagreement')
    if total > TENSOR_PROFILE.max_total_uncompressed:
        raise ValueError('tensor dictionary exceeds byte limit')
    return tensors


def decode_images(source_zip, output, metadata_path, expected_sha256):
    import numpy as np
    import torch
    import PIL
    from PIL import Image, features
    if sha256_file(source_zip) != expected_sha256:
        raise ValueError('source images.zip sha256 mismatch')
    images, samples, attribution, labels = {}, {}, {}, {}
    with safe_zip_open(source_zip, profile=IMAGES_ARCHIVE_PROFILE) as archive:
        for member in sorted(archive.names()):
            if member.startswith('images/') and member.lower().endswith(('.jpg', '.jpeg', '.png')):
                key = member.removeprefix('images/')
                if '/' in key:
                    continue
                data = archive.read_member(member)
                with Image.open(io.BytesIO(data)) as source:
                    if source.width * source.height * 3 > TENSOR_PROFILE.max_member_bytes:
                        raise ValueError(f'{key}: decoded image exceeds pixel limit')
                    rgb = source.convert('RGB')
                    tensor = torch.from_numpy(np.array(rgb, copy=True)).permute(2, 0, 1).contiguous()
                images[key] = tensor
                samples[key] = {'source_member': member, 'source_sha256': hashlib.sha256(data).hexdigest(),
                                'tensor': tensor_descriptor(tensor, layout='CHW')}
            elif member == 'SOURCES.md':
                attribution[member] = archive.read_member(member).decode('utf-8')
            elif member.startswith('labels/') and not member.endswith('/'):
                labels[member] = archive.read_member(member).decode('utf-8')
    if not images:
        raise ValueError('images.zip contains no samples')
    Path(output).write_bytes(tensor_bytes(images))
    metadata = {'schema_version': 1, 'source_archive_sha256': expected_sha256,
                'images_sha256': sha256_file(output), 'sample_order': sorted(images),
                'decoding': {'color_mode': 'RGB', 'exif_orientation': 'not_applied',
                             'Pillow': PIL.__version__, 'jpeg_decoder': features.version('jpg'),
                             'libjpeg_turbo': bool(features.check_feature('libjpeg_turbo'))},
                'samples': samples, 'attribution': attribution, 'labels': labels}
    schema_validate.validate_document(metadata, 'images-metadata', SCHEMAS)
    Path(metadata_path).write_bytes(json_bytes(metadata))
    return load_images(output, metadata_path, expected_sha256=metadata['images_sha256'])


def load_images(path, metadata_path, *, expected_sha256):
    metadata = opgraph.strict_json_loads(Path(metadata_path).read_bytes())
    schema_validate.validate_document(metadata, 'images-metadata', SCHEMAS)
    if metadata['images_sha256'] != expected_sha256:
        raise ValueError('decoded images sha256 disagrees with metadata')
    descriptors = {k: v['tensor'] for k, v in metadata['samples'].items()}
    if metadata['sample_order'] != sorted(descriptors):
        raise ValueError('decoded image sample ordering disagreement')
    images = load_tensors(path, expected_sha256=expected_sha256, descriptors=descriptors)
    for key, tensor in images.items():
        if str(tensor.dtype) != 'torch.uint8' or tensor.ndim != 3 or tensor.shape[0] != 3:
            raise ValueError(f'{key}: expected RGB uint8 [3,H,W]')
    return images


def rgb_image(tensor):
    from PIL import Image
    if tensor_descriptor(tensor)['dtype'] != 'uint8' or tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError('RGB reconstruction requires uint8 [3,H,W]')
    return Image.fromarray(tensor.permute(1, 2, 0).contiguous().numpy())


def recipe(config, transform):
    import PIL
    import timm
    import torch
    import torchvision
    root = Path(__file__).resolve().parents[1]
    config = opgraph.strict_json_loads(json_bytes(config))
    mode = config.get('crop_mode') or 'center'
    if mode not in ('center', 'squash', 'border') or config['interpolation'] not in ('bilinear', 'bicubic'):
        raise ValueError('unsupported preprocessing recipe')
    stages = [type(t).__name__ for t in transform.transforms]
    if stages not in (['Resize', 'CenterCrop', 'MaybeToTensor', 'Normalize'],
                      ['ResizeKeepRatio', 'CenterCrop', 'MaybeToTensor', 'Normalize'],
                      ['ResizeKeepRatio', 'CenterCropOrPad', 'MaybeToTensor', 'Normalize']):
        raise ValueError(f'unsupported preprocessing pipeline: {stages}')
    effective_interpolation = transform.transforms[0].interpolation
    effective_interpolation = getattr(effective_interpolation, 'value', effective_interpolation)
    return {'schema_version': 1, 'backend': 'PIL', 'config': config,
            'versions': {'timm': timm.__version__, 'torch': torch.__version__,
                         'torchvision': torchvision.__version__, 'Pillow': PIL.__version__},
            'timm_commit': subprocess.check_output(['git', '-C', str(root / 'modules/pytorch-image-models'),
                                                   'rev-parse', 'HEAD'], text=True).strip(),
            'helper_sha256': sha256_file(__file__),
            'platform': {'architecture': platform.machine(), 'system': platform.system()},
            'pipeline': stages, 'crop_mode_resolved': mode,
            'scale_size': [math.floor(x / (config.get('crop_pct') or 0.875)) for x in config['input_size'][1:]],
            'resize_rounding': ('truncate_secondary_dimension' if mode == 'center' and
                                config['input_size'][1] == config['input_size'][2] else 'ties_to_even'),
            'crop_rounding': 'ties_to_even', 'padding': 'extra_pixel_right_bottom',
            'padding_fill': [round(255 * x) for x in config['mean']] if mode == 'border' else [0, 0, 0],
            'interpolation': effective_interpolation, 'antialias': 'PIL_always',
            'operations': ['RGB_PIL_reconstruction', 'resize_uint8', 'center_crop_or_pad_uint8',
                           'CHW_float32', 'divide_float32_255', 'subtract_float32_mean',
                           'divide_float32_std']}


def preprocess(tensor, transform):
    """Record geometry at the actual torchvision calls, including timm's wrappers."""
    from torchvision.transforms import functional as F
    events = []

    def record(op, original):
        def invoke(image, *args, **kwargs):
            before = [image.height, image.width]
            result = original(image, *args, **kwargs)
            event = {'op': op, 'before': before, 'after': [result.height, result.width]}
            if op == 'crop':
                event['box'] = list(args[:4])
            elif op == 'pad':
                event['padding'] = list(args[0])
                event['fill'] = kwargs.get('fill', args[1] if len(args) > 1 else 0)
            events.append(event)
            return result
        return invoke

    with patch.object(F, 'resize', record('resize', F.resize)), \
         patch.object(F, 'pad', record('pad', F.pad)), \
         patch.object(F, 'crop', record('crop', F.crop)):
        result = transform(rgb_image(tensor))
    tensor_descriptor(result, layout='CHW')
    return result, events


def reproduce(images, preprocessing):
    from timm.data import create_transform
    declared = preprocessing['recipe']
    transform = create_transform(**declared['config'], is_training=False)
    resolved = recipe(declared['config'], transform)
    if resolved != declared:
        raise ValueError('recipe provenance or executable pipeline differs')
    tensors, geometry = {}, {}
    for key, image in sorted(images.items()):
        tensors[key], geometry[key] = preprocess(image, transform)
    return tensors, geometry


def compare_inputs(model, actual, reference):
    import torch
    if set(actual) != set(reference):
        raise ValueError(f'{model}: missing or unexpected input samples')
    for key in sorted(reference):
        if tensor_descriptor(actual[key]) != tensor_descriptor(reference[key]):
            if actual[key].shape != reference[key].shape:
                raise ValueError(f'{model}/{key}: input shape differs')
            a = actual[key].contiguous().view(torch.uint8).flatten()
            b = reference[key].contiguous().view(torch.uint8).flatten()
            index = int(torch.nonzero(a != b)[0])
            element = index // 4
            raise ValueError(f'{model}/{key}: first differing element {element}: '
                             f'{actual[key].flatten()[element].item()} vs '
                             f'{reference[key].flatten()[element].item()} (byte {index})')
