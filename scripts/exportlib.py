"""Helpers shared by the export drivers in this directory.

Both drivers -- ``export_report.py`` (measures every timm model) and ``export_pt2.py``
(writes .pt2 archives for the selected subset) -- run one model per subprocess in a
scratch directory. The isolation, the input-size rule and the CLI glob parsing have to
agree between them, so they live here rather than being reimplemented per driver.
"""
import concurrent.futures
import json
import os
import subprocess
import sys
import tempfile
import time

# The resolution every driver caps its tracing at. Shared because `make models.verify`
# compares graphs the two drivers produced, and that comparison only means anything if both
# traced the same architecture at the same size.
MAX_RES = 224


def cpu_count():
    """Cores this process may actually run on -- sched_getaffinity, not os.cpu_count(),
    which reports the host's cores and so over-counts inside a cpu-limited container."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def iso_env(tmpdir, hf_home=None):
    """Environment for one worker: caches redirected into `tmpdir`, threads pinned to one.

    `hf_home` opts a run out of the redirect for the HuggingFace cache alone, pointing it
    at a shared directory that outlives the worker. Weights are the one thing worth
    keeping between runs (they are large, immutable and remotely fetched), and a
    per-worker HF_HOME would re-download them for every model and leave CI nothing to
    cache. Everything else stays per-worker: inductor/triton caches are derived data, and
    HOME is redirected so nothing a model does on import can touch the real one.

    HF_HUB_OFFLINE is set either way. Downloading is a separate, explicit step
    (`export_pt2.py fetch`); by the time a worker runs, whatever it needs is already in
    the cache, and a worker that silently reaches the network would make the export
    dependent on hub availability.
    """
    env = os.environ.copy()
    env['HF_HUB_OFFLINE'] = '1'
    env['TRANSFORMERS_OFFLINE'] = '1'
    # One worker per core, so each worker gets one thread. Left at torch's default, every
    # worker would size its intra-op pool to the whole machine (24 threads x 24 workers
    # here) and they would spend the CPU-path exports fighting each other; the meta-path
    # ones never touch the pool at all.
    env['OMP_NUM_THREADS'] = '1'
    env['MKL_NUM_THREADS'] = '1'
    env['HOME'] = tmpdir
    env['XDG_CACHE_HOME'] = os.path.join(tmpdir, 'cache')
    env['HF_HOME'] = os.path.abspath(hf_home) if hf_home else os.path.join(tmpdir, 'cache', 'huggingface')
    env['TORCHINDUCTOR_CACHE_DIR'] = os.path.join(tmpdir, 'cache', 'inductor')
    env['TRITON_CACHE_DIR'] = os.path.join(tmpdir, 'cache', 'triton')
    return env


def run_worker(script, argv, name, timeout, hf_home=None):
    """Run `script` in a subprocess and return the JSON object it printed on its last line.

    A model that hangs, segfaults or exhausts memory takes down only its own worker; the
    caller gets a status dict either way, so one bad architecture never costs the results
    of the ~1300 around it.

    The worker runs in the scratch directory, so anything it writes relative to the working
    directory is deleted with it; output the caller keeps is named by absolute path.
    """
    with tempfile.TemporaryDirectory(prefix='timm_export_') as tmpdir:
        env = iso_env(tmpdir, hf_home)
        cmd = [sys.executable, os.path.abspath(script)] + [str(a) for a in argv]
        try:
            proc = subprocess.run(
                cmd, env=env, cwd=tmpdir, timeout=timeout,
                capture_output=True, text=True,
            )
        except subprocess.TimeoutExpired:
            return {'name': name, 'status': 'timeout', 'error': f'exceeded {timeout}s'}

        if proc.returncode != 0:
            stderr_tail = '\n'.join(proc.stderr.strip().splitlines()[-5:])
            return {'name': name, 'status': 'crashed', 'error': stderr_tail or f'exit {proc.returncode}'}

        line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ''
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return {'name': name, 'status': 'crashed', 'error': 'unparseable worker output'}


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


def globs(value):
    """timm.list_models takes either a glob or a list of them; accepting a comma-separated
    set here is what makes regenerating a handful of named variants practical."""
    parts = [p for p in value.split(',') if p]
    return parts if len(parts) > 1 else value


def run_pool(names, work, workers, progress):
    """Run `work(name)` over `names` in a thread pool, returning {name: error} for failures.

    Every caller here drives subprocesses, so the pool is waiting on them rather than
    computing; and every caller wants the same failure semantics -- one model that raises is
    one recorded failure, not an aborted run over the other fifty-nine.
    """
    failures = {}
    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, name): name for name in names}
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            name = futures[future]
            try:
                result = future.result()
            except Exception as e:
                result = {'name': name, 'status': 'crashed', 'error': str(e)[:300]}
            if result.get('status') != 'ok':
                failures[name] = result.get('error') or result.get('status')
            print(f'[{index}/{len(names)}] {name}: {result.get("status")}'
                  f'{progress(result)} ({time.monotonic() - start:.0f}s elapsed)', file=sys.stderr)

    if failures:
        print(f'\n{len(failures)} model(s) failed:', file=sys.stderr)
        for name, error in sorted(failures.items()):
            print(f'  {name}: {error}', file=sys.stderr)
    return failures
