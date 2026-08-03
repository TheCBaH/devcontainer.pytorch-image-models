"""Subprocess isolation harness shared by every export driver.

Every driver runs one model per subprocess in a scratch directory: the
isolation, the env redirection and the CLI glob parsing have to agree
between drivers, so they live here rather than being reimplemented per zoo.
"""
import concurrent.futures
import json
import os
import subprocess
import sys
import tempfile
import time


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

    HF_HUB_OFFLINE is set either way. Downloading is a separate, explicit step; by the
    time a worker runs, whatever it needs is already in the cache, and a worker that
    silently reaches the network would make the export dependent on hub availability.
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
    of the many around it.

    The worker runs in the scratch directory, so anything it writes relative to the working
    directory is deleted with it; output the caller keeps is named by absolute path.
    """
    with tempfile.TemporaryDirectory(prefix='pt2_export_') as tmpdir:
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


def globs(value):
    """A model registry's `list_models` typically takes either a glob or a list of them;
    accepting a comma-separated set here is what makes regenerating a handful of named
    variants practical."""
    parts = [p for p in value.split(',') if p]
    return parts if len(parts) > 1 else value


def run_pool(names, work, workers, progress):
    """Run `work(name)` over `names` in a thread pool, returning {name: error} for failures.

    Every caller here drives subprocesses, so the pool is waiting on them rather than
    computing; and every caller wants the same failure semantics -- one model that raises is
    one recorded failure, not an aborted run over the rest.
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
