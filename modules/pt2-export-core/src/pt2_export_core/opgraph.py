"""Analysis of an already-exported `torch.export.ExportedProgram`: which ATen operators it
uses, in which call configurations, and which dims stayed dynamic.

Dialect-agnostic: the same walk describes the graph `torch.export.export()` hands back (ATen:
`conv2d`, `linear`, `layer_norm`, `scaled_dot_product_attention`) and the one
`run_decompositions()` produces from it (core ATen: `convolution`, `addmm`,
`native_layer_norm`). Nothing here decides which of the two it is looking at.

Everything here is duck-typed on the exported graph itself (node schema,
`node.meta['val']`) -- it never constructs a model or an example input, so it has no
opinion about which zoo (timm, transformers, ...) produced the program.
"""
import contextlib
import json
import re

# torch.export retains SymInt arguments for two structurally different things: tensor
# extents (a `view` size, a `slice` bound), whose values are a function of the input
# resolution, and genuine architectural knobs that merely happen to be SymInt-typed. Only
# the latter belong in an operator's configuration -- recording the former verbatim
# explodes the catalog (measured: `view.size` alone contributes 533 distinct values across
# 30 timm models, vs 6 once abstracted), so SymInt args are recorded as arity except for the
# ops listed here. An op not listed defaults to abstraction, which is the safe direction:
# a new op can never blow the catalog up, only under-describe itself.
SYMINT_LITERAL_OPS = {
    'aten.convolution.default',
    'aten.constant_pad_nd.default',
    # The ATen-dialect counterparts of the two above: at that level a convolution's stride,
    # padding, dilation and groups are all SymInt-typed, and abstracting them would describe
    # convolution in less detail than the core catalog does. `pad` stands to `constant_pad_nd`
    # as `conv2d` does to `convolution`.
    'aten.conv1d.default',
    'aten.conv2d.default',
    'aten.conv3d.default',
    'aten.conv_transpose2d.input',
    'aten._convolution.default',
    'aten.pad.default',
}

# `device` differs between the meta and CPU export paths a driver may take per model, so
# recording it would make a model's configuration depend on which path it happened to take
# rather than on the architecture. `layout`/`pin_memory` are invariant noise.
DROPPED_ARGS = {'device', 'layout', 'pin_memory'}

# An export bookkeeping node, not computation a backend has to implement.
DROPPED_OPS = {'aten._assert_tensor_metadata.default'}

# Namespaces whose members carry no schema, so `collect_ops` skips them: higher-order ops are
# graph structure (autocast regions, control flow), not operators. Named rather than left to
# the `_schema` check because `archive.graph_op_counts` reads serialized graphs, where only the
# target string survives, and has to apply the same rule.
DROPPED_NAMESPACES = {'higher_order'}

_DTYPE_NAMES = {
    'torch.float32': 'f32', 'torch.float64': 'f64', 'torch.float16': 'f16',
    'torch.bfloat16': 'bf16', 'torch.int64': 'i64', 'torch.int32': 'i32',
    'torch.int16': 'i16', 'torch.int8': 'i8', 'torch.uint8': 'u8', 'torch.bool': 'bool',
}

_SYMINT_ARGS_CACHE = {}

# One `name: type` pair of a schema's argument list, e.g. `SymInt[] stride` -- the split
# below has already isolated it, so the type is everything up to the last whitespace.
_SCHEMA_ARG_RE = re.compile(r'^(?P<type>.+?)\s+(?P<name>[A-Za-z_][A-Za-z_0-9]*)$')


def describe_dynamic_shapes(ep, axis_names=None):
    """Render every free symbol torch.export retained in `ep`, verbatim from the graph's own
    metadata: no invented labels, just each dynamic dim's axis (substituted in place of the
    graph's internal symbol name, e.g. 's53' -> 'H', via `axis_names`, a `{dim index: label}`
    mapping the caller supplies for its own input layout -- e.g. `{2: 'H', 3: 'W'}` for a
    BCHW image tensor) and its real lower bound from ep.range_constraints. Dims that
    collapsed onto the same symbol (e.g. an architecture that only accepts square input)
    naturally report once, combined, since they *are* the same symbol -- no special-casing
    needed. Any symbol not traceable to one of the named input dims (unrelated
    data-dependent shape elsewhere in the model), or any dim the caller left unnamed, is
    still reported, by its raw graph symbol name / `dimN`, rather than silently dropped.
    """
    axis_names = axis_names or {}
    input_names = set(ep.graph_signature.user_inputs)
    sym_to_axes = {}
    for node in ep.graph_module.graph.nodes:
        if node.op != 'placeholder' or node.name not in input_names:
            continue
        val = node.meta.get('val')
        if not hasattr(val, 'shape'):
            continue
        for dim, size in enumerate(val.shape):
            expr = getattr(size, 'node', None) and size.node.expr
            if expr is None or not expr.free_symbols:
                continue  # concrete dim, e.g. batch=1 -- not a torch.export.Dim, no symbol
            for sym in expr.free_symbols:
                sym_to_axes.setdefault(sym, []).append(axis_names.get(dim, f'dim{dim}'))

    if not ep.range_constraints:
        return None

    parts = []
    for sym in sorted(ep.range_constraints, key=str):
        rc = ep.range_constraints[sym]
        axes = sym_to_axes.get(sym)
        label = '='.join(axes) if axes else str(sym)
        # ValueRanges is always a genuine (lower, upper) pair -- represent both consistently
        # as an interval rather than special-casing the (common) unbounded-upper case.
        upper = '∞' if str(rc.upper) == 'int_oo' else str(rc.upper)
        parts.append(f'{label}∈[{rc.lower},{upper}]')
    return ' '.join(parts) if parts else None


def symint_arg_names(schema):
    """Names of `schema`'s SymInt-typed arguments, read out of the schema *text*.

    The JIT type system erases SymInt -- `str(arg.type)` reports `List[int]` for both
    `int[] dims` (permutations: a real configuration) and `SymInt[] size` (a tensor
    extent), so the distinction only survives in the declaration string itself, e.g.
    `aten::view(Tensor(a) self, SymInt[] size) -> Tensor(a)`.
    """
    text = str(schema)
    cached = _SYMINT_ARGS_CACHE.get(text)
    if cached is not None:
        return cached

    body = text[text.index('(') + 1:text.rindex(') ->')]
    names = set()
    # Split on commas that are not inside a bracketed type/default, e.g. `int[2] stride=[]`.
    for part in re.split(r',(?![^\[]*\])', body):
        part = part.strip().lstrip('*').strip()
        part = part.split('=', 1)[0].strip()  # drop the default value
        m = _SCHEMA_ARG_RE.match(part)
        if m and 'SymInt' in m.group('type'):
            names.add(m.group('name'))
    _SYMINT_ARGS_CACHE[text] = names
    return names


def _plain(value):
    """Coerce a schema argument value to something JSON/YAML can round-trip."""
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)  # torch.contiguous_format, torch.float32, ...


def _meta_val(arg):
    """The fake tensor an argument carries, for arguments that are graph nodes at all
    (a node's args are just as often plain ints or lists)."""
    meta = getattr(arg, 'meta', None)
    return meta.get('val') if meta else None


def _concrete_shape(val):
    """`val`'s shape as plain ints, or None if it is absent or symbolic."""
    if val is None or not hasattr(val, 'shape'):
        return None
    try:
        return [int(d) for d in val.shape]
    except Exception:
        return None  # data-dependent/dynamic dim -- no honest concrete answer


def op_config(node):
    """(op name, configuration) for one call_function node of an exported graph.

    The configuration is the node's non-Tensor schema arguments (the knobs a backend has
    to honour: stride, eps, dim, keepdim, ...) with tensor-extent SymInts abstracted per
    SYMINT_LITERAL_OPS, plus a few facts derived from the graph's own shape metadata that
    the argument list does not carry -- most importantly a convolution's kernel size and
    whether it is depthwise, which is the difference between two very different kernels
    wearing the same `aten.convolution.default` name.
    """
    target = str(node.target)
    schema = node.target._schema
    symints = symint_arg_names(schema)
    abstract_symints = target not in SYMINT_LITERAL_OPS

    config = {}
    for i, arg in enumerate(schema.arguments):
        if 'Tensor' in str(arg.type) or arg.name in DROPPED_ARGS:
            continue
        value = node.args[i] if i < len(node.args) else node.kwargs.get(arg.name, arg.default_value)
        if abstract_symints and arg.name in symints:
            value = f'[*{len(value)}]' if isinstance(value, (list, tuple)) else '*'
        config[arg.name] = _plain(value)

    # Derived facts are prefixed `out_` rather than named `dtype`/`rank`: several schemas
    # (mean.dim, arange, to.dtype) already carry a `dtype` argument of their own, and the
    # requested dtype and the produced one are different facts.
    out = node.meta.get('val')
    if isinstance(out, (list, tuple)) and out:
        out = out[0]  # multi-output op (batch_norm, layer_norm) -- describe its primary result
    dtype = getattr(out, 'dtype', None)
    if dtype is not None:
        config['out_dtype'] = _DTYPE_NAMES.get(str(dtype), str(dtype))
    out_shape = _concrete_shape(out)
    if out_shape is not None:
        config['out_rank'] = len(out_shape)

    if 'convolution' in target or target.startswith('aten.conv'):
        weight = _concrete_shape(_meta_val(node.args[1])) if len(node.args) > 1 else None
        if weight and len(weight) > 2:
            config['kernel'] = weight[2:]
        inputs = _concrete_shape(_meta_val(node.args[0])) if node.args else None
        groups = config.get('groups')
        if inputs and len(inputs) > 1 and isinstance(groups, int):
            config['depthwise'] = groups > 1 and groups == inputs[1]

    return target, config


def canonical_config(config):
    """Order-independent identity of a configuration, for deduplication and id assignment.

    Configurations themselves keep their schema argument order (`stride` before `padding`
    before `dilation`), which is how a human reads them; this is only the key.
    """
    return json.dumps(config, sort_keys=True)


def collect_ops(ep):
    """([[op name, configuration, node count], ...], {op name: schema}) for an exported program.

    call_function nodes without a schema (`operator.getitem`, higher-order ops) are
    structural graph plumbing rather than operators a backend lowers, so they are skipped.
    """
    counts = {}
    configs = {}  # keyed the same, but keeping the schema-ordered dict for output
    schemas = {}
    for node in ep.graph_module.graph.nodes:
        if node.op != 'call_function' or not hasattr(node.target, '_schema'):
            continue
        target, config = op_config(node)
        if target in DROPPED_OPS:
            continue
        schemas[target] = str(node.target._schema).replace('aten::', '', 1)
        key = (target, canonical_config(config))
        counts[key] = counts.get(key, 0) + 1
        configs.setdefault(key, config)
    ops = [[target, configs[(target, config)], count] for (target, config), count in sorted(counts.items())]
    return ops, schemas


@contextlib.contextmanager
def time_budget(seconds):
    """Bound a block on its own SIGALRM budget, raising TimeoutError when it overruns.

    Post-export sub-checks (dynamic shapes, op collection) can run far longer than the
    export itself on a minority of architectures, and neither is worth losing the
    already-computed results over -- so each gets a budget independent of the caller's own
    subprocess timeout.
    """
    import signal

    def _alarm(signum, frame):
        raise TimeoutError()

    old_handler = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
