"""timm-specific helpers shared by the export drivers in this directory.

The domain-agnostic parts of what used to live here -- the subprocess isolation harness
(`cpu_count`, `iso_env`, `run_worker`, `run_pool`, `globs`) -- moved to the
`pt2_export_core.harness` package, which every driver now imports directly. What is left
here reads timm's own registry (`get_pretrained_cfg`, `default_cfg`), so it stays specific
to this zoo rather than the shared core.
"""

# The resolution every driver caps its tracing at. Shared because `make models.verify`
# compares graphs the two drivers produced, and that comparison only means anything if both
# traced the same architecture at the same size.
MAX_RES = 224


def pretrained_info(name):
    """(tag, hf_hub_id) for a model's default pretrained weights, or (None, None).

    A model can carry a pretrained *tag* while having nowhere to fetch it from -- timm's
    `test_*` architectures are the obvious case -- so the presence of a real source, not of
    a tag, is what decides whether a variant can ship as a runnable .pt2.
    """
    from timm.models import get_pretrained_cfg
    try:
        cfg = get_pretrained_cfg(name)
    except Exception:
        return None, None
    if not (cfg.hf_hub_id or cfg.url or cfg.file):
        return None, None
    return cfg.tag or None, cfg.hf_hub_id or None


def resolved_input_size(default_cfg, max_res):
    """The (C, H, W) a model is traced at: its own default, capped at `max_res`.

    Uncapped, the handful of architectures configured for 512x512 or 1024x1024 dominate
    the run's cost for no extra information. A model that declares `min_input_size` is
    dropped to it rather than to `max_res`, because that floor is a real architectural
    constraint (patch/window divisibility) and an arbitrary cap would violate it.
    """
    input_size = default_cfg['input_size']
    if default_cfg.get('fixed_input_size') or max(input_size) <= max_res:
        return input_size
    return default_cfg.get('min_input_size') or tuple(min(x, max_res) for x in input_size)
