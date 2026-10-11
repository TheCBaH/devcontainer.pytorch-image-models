"""Signature-driven evaluation bindings and cross-document tensor checks."""
import hashlib

from pt2_export_core import opgraph, schema_validate
from input_assets import SCHEMAS, tensor_descriptor


def graph_signature(graph):
    module = graph['graph_module']
    tensors = module['graph']['tensor_values']

    def user_specs(side):
        tag = 'user_input' if side == 'input' else 'user_output'
        values = []
        for spec in module['signature'][side + '_specs']:
            if tag not in spec:
                continue
            arg = spec[tag]['arg']
            if set(arg) != {'as_tensor'}:
                raise ValueError('only static tensor user bindings are supported')
            name = arg['as_tensor']['name']
            value = tensors[name]
            if value['dtype'] != 7 or any(set(x) != {'as_int'} for x in value['sizes']):
                raise ValueError(f'{name}: only static float32 graph tensors are supported')
            shape = [x['as_int'] for x in value['sizes']]
            if any(x < 1 for x in shape):
                raise ValueError(f'{name}: invalid static shape')
            values.append({'name': name, 'dtype': 'float32', 'shape': shape})
        if not values or len({x['name'] for x in values}) != len(values):
            raise ValueError(f'graph has missing or duplicate user {side}s')
        return values

    return user_specs('input'), user_specs('output')


def input_value(sample, transform):
    if transform == {'op': 'identity'}:
        return sample
    if transform == {'op': 'unsqueeze', 'axis': 0}:
        return sample.unsqueeze(0)
    raise ValueError(f'unsupported input transform: {transform}')


def output_value(value, transforms):
    if transforms not in ([], [{'op': 'select', 'axis': 0, 'index': 0},
                              {'op': 'cast', 'dtype': 'float32'}]):
        raise ValueError(f'unsupported output transforms: {transforms}')
    return value if not transforms else value[0].float()


def _sample_metadata(tensor, layout='row_major'):
    desc = tensor_descriptor(tensor, layout=layout)
    return {k: desc[k] for k in ('dtype', 'shape', 'layout')}


def build_contract(name, graph_bytes, inputs, outputs, *, decoded_sha256,
                   geometry, input_specs=None, output_specs=None, case_mappings=None):
    graph = opgraph.strict_json_loads(graph_bytes)
    user_inputs, user_outputs = graph_signature(graph)
    input_layout = 'CHW' if input_specs is None else 'row_major'
    if input_specs is None:
        if len(user_inputs) != 1 or len(user_outputs) != 1:
            raise ValueError('classification packing requires one user input and output')
        input_specs = {user_inputs[0]['name']: ('image', {'op': 'unsqueeze', 'axis': 0})}
        output_specs = {user_outputs[0]['name']: ('logits', [
            {'op': 'select', 'axis': 0, 'index': 0}, {'op': 'cast', 'dtype': 'float32'}])}
    if case_mappings is None:
        if set(inputs) != set(outputs) or set(inputs) != set(geometry):
            raise ValueError('case sample sets disagree')
        case_mappings = [{'id': key, 'decoded_image_key': key,
                          'inputs': {v[0]: key for v in input_specs.values()},
                          'outputs': {v[0]: key for v in output_specs.values()}}
                         for key in sorted(inputs)]
    if not case_mappings:
        raise ValueError('evaluation has no cases')

    def bindings(users, specs, tensors, side):
        if set(specs) != {x['name'] for x in users}:
            raise ValueError(f'{side} binding coverage disagrees with graph signature')
        result = []
        for user in users:
            sample_name, transform = specs[user['name']]
            key = case_mappings[0][side][sample_name]
            item = {f'graph_{side[:-1]}': user['name'], f'sample_{side[:-1]}': sample_name,
                    'graph_dtype': user['dtype'], 'graph_shape': user['shape'],
                    'sample': _sample_metadata(tensors[key], input_layout if side == 'inputs' else 'row_major'),
                    'transform': transform}
            result.append(item)
        return result

    contract = {'schema_version': 2, 'model': name,
                'graph': {'sha256': hashlib.sha256(graph_bytes).hexdigest(),
                          'path': f'models/{name}/models/model.json',
                          'user_inputs': user_inputs, 'user_outputs': user_outputs},
                'decoded_images': {'asset': 'decoded_images', 'sha256': decoded_sha256,
                                   'metadata_asset': 'image_metadata'},
                'inputs': {'asset': f'{name}/inputs',
                           'bindings': bindings(user_inputs, input_specs, inputs, 'inputs')},
                'outputs': {'member': 'outputs.pt',
                            'bindings': bindings(user_outputs, output_specs, outputs, 'outputs')},
                'cases': {'member': 'cases.json'}, 'preprocessing': {'member': 'preprocessing.json'},
                'comparison': {'atol': 1e-4, 'rtol': 1e-3, 'finite': 'required',
                               'rule': 'abs(actual-reference)<=atol+rtol*abs(reference)',
                               'preprocessing': 'exact_bits',
                               'tie_break': '(-logit_value, class_index)'}}
    cases = {'schema_version': 1, 'cases': []}
    for mapping in case_mappings:
        case = {'id': mapping['id'], 'decoded_image_key': mapping['decoded_image_key'],
                'geometry': geometry[mapping['decoded_image_key']]}
        for side, tensors in (('inputs', inputs), ('outputs', outputs)):
            case[side] = {sample: {'tensor_key': key, 'tensor': tensor_descriptor(
                tensors[key], layout=input_layout if side == 'inputs' else 'row_major')}
                          for sample, key in mapping[side].items()}
        cases['cases'].append(case)
    validate_contract(contract, cases, graph_bytes, inputs, outputs)
    return contract, cases


def validate_contract(contract, cases, graph_bytes, inputs, outputs, *, decoded_keys=None):
    import torch
    schema_validate.validate_document(contract, 'contract-v2', SCHEMAS)
    schema_validate.validate_document(cases, 'cases', SCHEMAS)
    user_inputs, user_outputs = graph_signature(opgraph.strict_json_loads(graph_bytes))
    if hashlib.sha256(graph_bytes).hexdigest() != contract['graph']['sha256']:
        raise ValueError('stale graph digest')
    if (contract['graph']['user_inputs'] != user_inputs or contract['graph']['user_outputs'] != user_outputs):
        raise ValueError('graph signature metadata disagreement')
    seen = set()
    used = {'inputs': set(), 'outputs': set()}
    for side, users, tensors in (('inputs', user_inputs, inputs), ('outputs', user_outputs, outputs)):
        bindings = contract[side]['bindings']
        graph_field, sample_field = f'graph_{side[:-1]}', f'sample_{side[:-1]}'
        if [b[graph_field] for b in bindings] != [u['name'] for u in users]:
            raise ValueError(f'{side}: omitted, duplicate or unordered graph bindings')
        samples = [b[sample_field] for b in bindings]
        if len(set(samples)) != len(samples):
            raise ValueError(f'{side}: duplicate sample binding')
        for binding, user in zip(bindings, users):
            if binding['graph_shape'] != user['shape'] or binding['graph_dtype'] != user['dtype']:
                raise ValueError(f'{side}: wrong graph shape/dtype')
        for case in cases['cases']:
            if side == 'inputs':
                if case['id'] in seen:
                    raise ValueError('duplicate case id')
                seen.add(case['id'])
                if decoded_keys is not None and case['decoded_image_key'] not in decoded_keys:
                    raise ValueError('missing decoded image sample')
            if set(case[side]) != set(samples):
                raise ValueError(f'{side}: incomplete case bindings')
            for binding in bindings:
                stored = case[side][binding[sample_field]]
                key = stored['tensor_key']
                if key not in tensors:
                    raise ValueError(f'{case["id"]}: missing tensor key {key}')
                used[side].add(key)
                tensor = tensors[key]
                layout = stored['tensor']['layout']
                if tensor_descriptor(tensor, layout=layout) != stored['tensor'] or _sample_metadata(tensor, layout) != binding['sample']:
                    raise ValueError(f'{case["id"]}/{key}: tensor descriptor disagreement')
                if side == 'inputs':
                    value = input_value(tensor, binding['transform'])
                    if list(value.shape) != binding['graph_shape'] or str(value.dtype) != 'torch.' + binding['graph_dtype']:
                        raise ValueError(f'{key}: input transform shape/dtype disagreement')
                else:
                    value = output_value(torch.zeros(binding['graph_shape'], dtype=torch.float32),
                                         binding['transform'])
                    if list(value.shape) != binding['sample']['shape'] or str(value.dtype) != 'torch.' + binding['sample']['dtype']:
                        raise ValueError(f'{key}: output transform shape/dtype disagreement')
        if used[side] != set(tensors):
            raise ValueError(f'{side}: unused tensor keys')
