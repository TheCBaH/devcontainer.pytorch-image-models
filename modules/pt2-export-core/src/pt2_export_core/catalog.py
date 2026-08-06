"""Render/parse the operator cross-reference: a models x operations matrix over the (op, call
configuration) units `opgraph.collect_ops` produces for each exported model.

One cross-reference is written per *dialect* -- the ATen graph `torch.export` hands back, and
the core ATen graph `run_decompositions()` produces from it -- because the two describe
genuinely different operator sets rather than one being a subset of the other. The core one is
additionally written per *backend*: decomposition runs after dispatch, so which kernel a
composite operator expands into depends on the device the model was traced on, and a file that
merged two backends would be describing no single lowering at all.
"""
from typing import NamedTuple

import yaml

from .markdown import heading_anchor
from .opgraph import canonical_config

# Backends a core ATen cross-reference is always written for, so an empty one states that
# nothing needed that backend rather than leaving a missing file to interpret. Other hardware
# adds to this -- ops-core-cuda.yaml sits alongside rather than replacing anything.
CORE_BACKENDS = ('meta', 'cpu')

_ATEN_PROSE = (
    'Operators are read from the graph `torch.export.export()` hands back, before any '
    'decomposition -- the ATen dialect, with `conv2d`, `linear`, `layer_norm` and '
    '`scaled_dot_product_attention` still whole. This is also the graph published under '
    '`models/`. It is not functionalized, so in-place forms (`relu_`, `add_`, `silu_`) and '
    'eval-time `dropout` appear as themselves and a consumer has to handle mutation. What it '
    'does not depend on is the device it was traced on; the decomposed graph does, which is why '
    'its cross-references are named per backend (`ops-core-<backend>.md`).'
)

_CORE_PROSE = (
    'Operators are read from the graph `run_decompositions()` produces on the **{backend}** '
    'backend -- core ATen, what a PT2 backend actually lowers, rather than the '
    '`conv2d`/`batch_norm`/`linear` the exporter hands back first ([`ops-aten.md`](ops-aten.md)). '
    'That decomposition runs *after* dispatch, so the lowering is specific to {backend}: a '
    'composite operator expands into whichever kernel the dispatcher selected, and one '
    '`scaled_dot_product_attention` call becomes 20 nodes on meta against 22 on cpu. Hence the '
    'backend in the file name -- the same zoo lowered on other hardware belongs in its own file.'
)


class Dialect(NamedTuple):
    """Which graph a cross-reference describes, and how to say so.

    Keeping the prose here makes adding a dialect (or a backend) data rather than a second copy
    of the renderers.
    """

    key: str            # 'aten' | 'core'
    label: str          # how the dialect is named in prose
    backend: str | None  # None when the graph does not depend on one, i.e. for ATen
    ir: str             # the `# ir:` line of the YAML header
    prose: str          # the markdown paragraph saying where this graph comes from

    @property
    def stem(self):
        return f'ops-{self.key}' + (f'-{self.backend}' if self.backend else '')

    @property
    def yaml_name(self):
        return f'{self.stem}.yaml'

    @property
    def md_name(self):
        return f'{self.stem}.md'


ATEN = Dialect('aten', 'ATen', None, 'ATen (torch.export.export)', _ATEN_PROSE)


def core(backend):
    """The core ATen dialect as lowered on one backend. See CORE_BACKENDS."""
    return Dialect('core', 'core ATen', backend,
                   f'core ATen (torch.export + run_decompositions)   backend: {backend}',
                   _CORE_PROSE.format(backend=backend))


def short_op(op):
    """`aten.convolution.default` -> `convolution.default`. The overload suffix stays:
    different overloads are different operators to whoever has to implement them."""
    return op[len('aten.'):] if op.startswith('aten.') else op


def full_op(op):
    """Inverse of short_op(). An operator name is `namespace.name.overload`, so a key with
    a namespace still on it (anything but aten's) keeps exactly two dots and is left alone."""
    return op if op.count('.') >= 2 else f'aten.{op}'


def build_op_index(ops_by_model):
    """({op: {canonical config: id}}, {op: [(id, config)]}) over every model in the run.

    Ids are per-op and positional: 1..N over that op's configurations in canonical sort
    order, which makes them deterministic for a given run but not stable across runs that
    discover a new configuration sorting ahead of an existing one.
    """
    by_op = {}
    for ops in ops_by_model.values():
        for op, config, _count in ops:
            by_op.setdefault(op, {}).setdefault(canonical_config(config), config)

    index, catalog = {}, {}
    for op, configs in by_op.items():
        ordered = sorted(configs)
        index[op] = {canonical: i for i, canonical in enumerate(ordered, 1)}
        catalog[op] = [(i, configs[canonical]) for i, canonical in enumerate(ordered, 1)]
    return index, catalog


def op_usage(ops_by_model, families):
    """{op: {config id: {'models': set, 'families': set, 'nodes': n, 'example': name}}}.

    Models and families are kept as sets rather than counters because the digest reports both
    per configuration and rolled up per operator, and a variant using four configurations
    of `convolution` is still one variant at the operator level.
    """
    index, _ = build_op_index(ops_by_model)
    usage = {}
    for name in sorted(ops_by_model):
        family = families.get(name, 'unknown')
        for op, config, count in ops_by_model[name]:
            cell = usage.setdefault(op, {}).setdefault(
                index[op][canonical_config(config)],
                {'models': set(), 'families': set(), 'nodes': 0, 'example': name},
            )
            cell['models'].add(name)
            cell['families'].add(family)
            cell['nodes'] += count
    return usage


def _literal_symint_ops(catalog):
    """The SYMINT_LITERAL_OPS this catalog actually contains, short-named and sorted.

    The set covers both dialects (`convolution` at core level, `conv2d` at ATen level), so each
    file documents only its own half of it.
    """
    from .opgraph import SYMINT_LITERAL_OPS

    return sorted(short_op(op) for op in SYMINT_LITERAL_OPS if op in catalog)


class _FlowMap(dict):
    """A mapping rendered inline (`{stride: [1, 1], groups: 1}`) rather than as a block."""


class _OpsDumper(yaml.SafeDumper):
    pass


_OpsDumper.add_representer(
    _FlowMap,
    lambda dumper, data: dumper.represent_mapping('tag:yaml.org,2002:map', data, flow_style=True),
)


def render_ops_yaml(ops_by_model, op_schemas, skipped, dialect, zoo_name, zoo_version,
                     torch_version, script='scripts/export_report.py'):
    """The cross-reference itself: an operator catalog plus a sparse models × operations
    matrix whose cells are {configuration id: node count}.

    Flow-style mappings and an effectively unlimited line width keep one model-operation
    pair (and one configuration) to a line, which is what makes a large zoo greppable and
    diffable; PyYAML's default block style and 80-column wrapping do neither.
    """
    index, catalog = build_op_index(ops_by_model)

    ops = {}
    for op in sorted(catalog):
        ops[short_op(op)] = {
            'schema': op_schemas.get(op, ''),
            'configs': {i: _FlowMap(config) for i, config in catalog[op]},
        }

    models = {}
    for name in sorted(ops_by_model):
        cells = {}
        for op, config, count in ops_by_model[name]:
            cells.setdefault(short_op(op), {})[index[op][canonical_config(config)]] = count
        models[name] = {op: _FlowMap(sorted(cell.items())) for op, cell in sorted(cells.items())}

    document = {'ops': ops, 'models': models}
    if skipped:
        document['skipped'] = {name: skipped[name] for name in sorted(skipped)}

    header = [
        f'# {zoo_name} x aten cross-reference -- generated by {script}, do not edit.',
        '#',
        f'# ir: {dialect.ir}   {zoo_name}: {zoo_version}   torch: {torch_version}',
        '#',
        '# ops.<op>.configs[<id>] -- one distinct configuration of that operator seen across the',
        '#   zoo: its non-Tensor schema arguments, plus out_dtype/out_rank read from the graph\'s',
        '#   own shape metadata (and kernel/depthwise for convolution, which the argument list',
        '#   does not carry but a backend very much cares about).',
        '# models.<variant>.<op> -- {configuration id: number of nodes using it}.',
        '# skipped -- models that exported but whose operators could not be collected.',
        '#',
    ]
    if dialect.backend:
        header += [
            f'# This is the lowering produced on the {dialect.backend} backend: decomposition runs',
            '# after dispatch, so the same zoo lowered on other hardware is a different file, not',
            f'# extra rows here. The undecomposed graph, which does not vary, is in {ATEN.yaml_name}.',
            '#',
        ]
    header += [
        '# SymInt arguments are tensor extents that scale with the input resolution, so they are',
        '# recorded as arity ("[*4]", or "*" for a scalar) rather than verbatim, except on',
        # Only the exceptions this dialect contains: the set spans both, and naming absent ops
        # would send a reader looking for them here.
        '# ' + ', '.join(_literal_symint_ops(catalog)) + ',',
        '# where they are architectural knobs. Configuration ids are positional within an op, so',
        '# they can shift between runs when a newly seen configuration sorts ahead of an old one.',
        '',
    ]
    body = yaml.dump(document, Dumper=_OpsDumper, sort_keys=False, width=10 ** 6, default_flow_style=False)
    return '\n'.join(header) + body


def fmt_config(config):
    """A configuration as compact one-line text for the markdown digest."""
    parts = []
    for key, value in config.items():
        if isinstance(value, list):
            rendered = '[' + ','.join(str(v) for v in value) + ']'
        elif isinstance(value, bool):
            rendered = 'true' if value else 'false'
        elif value is None:
            rendered = 'none'
        else:
            rendered = str(value)
        parts.append(f'{key}={rendered}')
    return (' '.join(parts)).replace('|', '/').replace('\n', ' ').strip() or '(no arguments)'


def render_ops_md(ops_by_model, op_schemas, families, skipped, dialect, zoo_name, zoo_version,
                   torch_version, script='scripts/export_report.py'):
    """Op-major digest of the cross-reference: for each operator, which configurations the
    zoo needs and how many variants need each. The model axis is collapsed to counts --
    the per-variant detail lives in the YAML, which is the machine-readable artifact."""
    _, catalog = build_op_index(ops_by_model)
    usage = op_usage(ops_by_model, families)

    total_configs = sum(len(configs) for configs in catalog.values())

    def op_models(op):
        return set().union(*(c['models'] for c in usage[op].values()))

    ordered_ops = sorted(catalog, key=lambda op: (-len(op_models(op)), op))

    # Anchors are minted in the order the headings below are emitted, since that is what
    # decides GitHub's numeric suffix on any repeated slug. The summary table doubles as
    # this file's table of contents, so every operator in it links to its own section.
    anchors = {}
    summary_anchor = heading_anchor('summary', anchors)
    heading_anchor('configurations', anchors)
    op_anchor = {op: heading_anchor(short_op(op), anchors) for op in ordered_ops}

    title = f'{zoo_name} {dialect.label} operator cross-reference'
    if dialect.backend:
        title += f' ({dialect.backend})'
    lines = [f'# {title}', '']
    lines.append(f'Generated by `{script}` against {zoo_name} {zoo_version} and torch '
                 f'{torch_version}. {len(ops_by_model)} exported variants use {len(catalog)} distinct '
                 f'{dialect.label} operators in {total_configs} distinct configurations.')
    lines.append('')
    lines.append(dialect.prose)
    lines.append('')
    lines.append('A *configuration* is an operator\'s non-Tensor schema arguments plus `out_dtype` '
                 'and `out_rank` taken from the graph\'s shape metadata, and for convolutions the '
                 '`kernel` size and whether it is `depthwise`. SymInt arguments are tensor extents that scale '
                 'with input resolution, so they are recorded as arity (`size=[*4]`) rather than '
                 'verbatim, except on ' +
                 ', '.join(f'`{op}`' for op in _literal_symint_ops(catalog)) +
                 ', where they are architectural knobs. `models` counts variants using that '
                 'configuration, `nodes` the total call sites across them. The full per-variant '
                 f'matrix is in [`{dialect.yaml_name}`]({dialect.yaml_name}); ids here are that '
                 'file\'s ids.')
    lines.append('')

    lines.append('## summary')
    lines.append('')
    lines.append('Every operator in the file, most widely used first; each name links to its '
                 'configurations below.')
    lines.append('')
    lines.append('| op | configs | models | families | nodes |')
    lines.append('|---|---|---|---|---|')
    for op in ordered_ops:
        cells = usage[op].values()
        lines.append(f'| [{short_op(op)}](#{op_anchor[op]}) | {len(catalog[op])} | '
                     f'{len(op_models(op))} | '
                     f'{len(set().union(*(c["families"] for c in cells)))} | '
                     f'{sum(c["nodes"] for c in cells)} |')
    lines.append('')

    lines.append('## configurations')
    lines.append('')
    for op in ordered_ops:
        lines.append(f'### {short_op(op)}')
        lines.append('')
        lines.append(f'[↑ summary](#{summary_anchor})')
        lines.append('')
        schema = op_schemas.get(op)
        if schema:
            lines.append(f'`{schema}`')
            lines.append('')
        lines.append('| id | configuration | models | families | nodes | e.g. |')
        lines.append('|---|---|---|---|---|---|')
        for i, config in sorted(catalog[op], key=lambda c: (-len(usage[op][c[0]]['models']), c[0])):
            cell = usage[op][i]
            lines.append(f'| {i} | {fmt_config(config)} | {len(cell["models"])} | {len(cell["families"])} | '
                         f'{cell["nodes"]} | {cell["example"]} |')
        lines.append('')

    if skipped:
        lines.append('## skipped')
        lines.append('')
        lines.append('Variants that exported but whose operators could not be collected in this '
                     f'dialect, so they contribute to neither this file nor `{dialect.yaml_name}`.')
        lines.append('')
        lines.append('| variant | reason |')
        lines.append('|---|---|')
        for name in sorted(skipped):
            reason = skipped[name].replace('|', '/').replace('\n', ' ').strip()
            lines.append(f'| {name} | {reason} |')
        lines.append('')

    return '\n'.join(lines)


def parse_existing_ops(path):
    """Parse a previously generated cross-reference YAML back into
    ({name: [[op, config, count], ...]}, {op: schema}, {name: reason}) for --resume."""
    import os

    if not os.path.exists(path):
        return {}, {}, {}
    with open(path) as f:
        document = yaml.load(f, Loader=getattr(yaml, 'CSafeLoader', yaml.SafeLoader)) or {}

    catalog = document.get('ops') or {}
    schemas = {}
    configs = {}  # short op -> {id: config}
    for op, entry in catalog.items():
        schemas[full_op(op)] = entry.get('schema', '')
        configs[op] = entry.get('configs') or {}

    ops_by_model = {}
    for name, cells in (document.get('models') or {}).items():
        ops = []
        for op, counts in cells.items():
            for config_id, count in counts.items():
                config = configs.get(op, {}).get(config_id)
                if config is None:
                    continue  # catalog/matrix disagree -- treat the model as incomplete
                ops.append([full_op(op), dict(config), count])
        ops_by_model[name] = ops

    return ops_by_model, schemas, dict(document.get('skipped') or {})
