"""Check code for GPUs we do not have: compile every Triton kernel a run launches for other architectures as well,
and optionally make SGLang / XTier choose their code paths as on another GPU.

  python tools/arch_check.py sm89 [sm86 ...] [--emulate sm89] -- <script.py | -m module | launch.sh> [args ...]

Compile check: the run itself is unchanged (kernels launch for the local GPU); each launch also compiles the same
specialization for the listed architectures (compile only, nothing loaded), in this process and in every child process
(a sitecustomize on PYTHONPATH installs the hook where XTIER_ARCH_CHECK is set). A kernel that uses an instruction or
type the target lacks (e.g. cvt.rn.f16x2.e2m1x2 below SM100, FP8 types below SM89) fails here the way it would fail on
that GPU, and one that needs more shared memory per block than the target has is reported too.

--emulate smXY: torch.cuda.get_device_capability / get_device_properties answer XY to callers in the sglang, xtier
and tilelang packages (their dispatch: attention backends, GEMM runners, kernel choices, TileLang's target) and the
truth to everything else (Triton, FlashInfer's JIT, torch itself), so those kernels still build for the local GPU.
SGLang's CUDA JIT and TileLang build the target's code as PTX (compute_XY, which the driver compiles for the local GPU
when the module loads): SGLang's sources assert SGL_CUDA_ARCH == __CUDA_ARCH__, and TileLang lowers for its target,
so the target's code paths cannot be built as the local GPU's binary. FlashInfer's JIT builds what SGLang sets in
FLASHINFER_CUDA_ARCH_LIST (the emulated capability) as PTX too. A path the target GPU would take then runs
here; what it would lack shows up as a missing binary (FlashMLA, DeepGEMM have none for SM8x either), a TileLang or
SGLang JIT build error, or in the compile check.

The compile check covers Triton kernels (TileLang and SGLang JIT kernels are built for the emulated target itself):
CUDA extensions (sgl-kernel, FlashInfer AOT, DeepGEMM, FlashMLA) ship their own per-architecture binaries (list them
with cuobjdump --list-elf).
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
from collections import defaultdict
from pathlib import Path

_local = threading.local()
# largest dynamic shared memory per block (opt-in), bytes
SMEM = {80: 166912, 86: 101376, 87: 166912, 89: 101376, 90: 232448}


def install(archs, report_dir):
    """Patch triton.runtime.jit.JITFunction.run in this process; results go to report_dir as they come."""
    from triton.backends.compiler import GPUTarget
    from triton.compiler import make_backend
    from triton.runtime.jit import JITFunction, create_function_from_signature

    original = JITFunction.run
    results = {}
    out = Path(report_dir, f'{os.getpid()}.jsonl')

    def record(key, status):
        if results.get(key) == status or (key in results and status == 'ok'):
            return
        results[key] = status
        with out.open('a') as f:   # as it happens: a server is stopped, not exited
            f.write(json.dumps([key, status]) + '\n')

    def binder(fn, capability):
        # What JITFunction.create_binder builds, for another target: the compiler takes the target (and the backend's
        # code generation its architecture) from these entries, not from the GPU in the machine.
        fn.create_binder()   # sets fn.compile / fn.ASTSource on a kernel not launched yet
        target = GPUTarget('cuda', capability, 32)
        backend = make_backend(target)
        return {}, {}, target, backend, create_function_from_signature(fn.signature, fn.params, backend)

    def run(self, *args, grid, warmup, **kwargs):
        if not warmup and not getattr(_local, 'busy', False):
            name = f'{self.fn.__module__}.{self.fn.__qualname__}'
            for arch in archs:
                # A separate cache per target: the in-memory cache key leaves the architecture out, so a kernel
                # compiled for the target must never land where the real launch looks it up.
                caches = self.__dict__.setdefault('_arch_check_caches', {})
                if arch not in caches:
                    caches[arch] = defaultdict(lambda fn=self, cap=int(arch[2:]): binder(fn, cap))
                real = self.device_caches
                self.device_caches, _local.busy = caches[arch], True
                try:
                    kernel = original(self, *args, grid=grid, warmup=True, **kwargs)
                    if kernel is not None and hasattr(kernel, 'result'):   # an async compile mode: wait for it
                        kernel = kernel.result()
                    shared = getattr(getattr(kernel, 'metadata', None), 'shared', 0) or 0
                    limit = SMEM.get(int(arch[2:]))
                    if limit and shared > limit:
                        record(f'{arch} {name}', f'shared memory {shared} B per block > {limit} B')
                    else:
                        record(f'{arch} {name}', 'ok')
                except Exception as e:   # noqa: BLE001  report, never break the run
                    lines = [x for x in str(e).strip().splitlines() if x.strip()]
                    record(f'{arch} {name}', f'{type(e).__name__}: ' + ' | '.join(lines[-3:])[:400])
                finally:
                    self.device_caches, _local.busy = real, False
        return original(self, *args, grid=grid, warmup=warmup, **kwargs)

    JITFunction.run = run


def _after_import(name, patch):
    """Run patch(module) right after `name` is first imported (before any other module can import from it)."""
    import importlib.abc
    import importlib.util

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != name:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(fullname)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return spec
            exec_module = spec.loader.exec_module

            def run(module):
                exec_module(module)
                patch(module)
            spec.loader.exec_module = run
            return spec

    sys.meta_path.insert(0, Finder())


def emulate(capability):
    """torch.cuda.get_device_capability / get_device_properties report `capability` (e.g. 89) to sglang and xtier.

    SGLang's CUDA JIT then builds the target's code paths (-DSGL_CUDA_ARCH, PDL, vector widths, ...) and TileLang
    lowers its kernels for the target; both are built as the target's PTX, which the driver compiles for the GPU in
    the machine, so the kernels run here. The CuTe DSL keeps the real target."""
    import torch

    major, minor = divmod(int(capability), 10)
    real_capability, real_properties = torch.cuda.get_device_capability, torch.cuda.get_device_properties
    real_major, real_minor = real_capability()

    def ours():
        caller = sys._getframe(2).f_globals.get('__name__', '')
        return caller.startswith(('sglang', 'xtier', 'tilelang'))

    class Properties:
        def __init__(self, p):
            self._p = p
            self.major, self.minor = major, minor

        def __getattr__(self, name):
            return getattr(self._p, name)

    def get_device_capability(device=None):
        return (major, minor) if ours() else real_capability(device)

    def get_device_properties(device=None):
        p = real_properties(device) if device is not None else real_properties()
        return Properties(p) if ours() else p

    torch.cuda.get_device_capability = get_device_capability
    torch.cuda.get_device_properties = get_device_properties

    def patch_toolchain(toolchain):
        target_flags = toolchain.target_flags

        def ptx_target_flags():
            # -gencode=arch=compute_89,code=sm_89 -> code=compute_89: PTX only, the binary is made at load time
            return [f.replace(',code=sm_', ',code=compute_') if f.startswith('-gencode=') else f
                    for f in target_flags()]
        toolchain.target_flags = ptx_target_flags

    _after_import('sglang.kernels.jit.utils.compile.toolchain', patch_toolchain)

    def patch_tilelang_nvcc(nvcc):
        # TileLang's builds (shared libraries, cubins, PTX) all run get_nvcc_compiler(): a wrapper that turns the
        # target's SASS into PTX (-arch=sm_XY and code=sm_XY -> compute_XY; a cubin becomes a fatbin holding the PTX,
        # which cuModuleLoadData takes as well)
        real = nvcc.get_nvcc_compiler()
        t = f'{major}{minor}'
        wrapper = Path(os.environ.get('XTIER_ARCH_CHECK_DIR') or tempfile.gettempdir(), f'nvcc-sm{t}')
        if not wrapper.exists():
            tmp = wrapper.with_name(f'{wrapper.name}.{os.getpid()}')
            tmp.write_text(
                '#!/usr/bin/env bash\n'
                f'# tools/arch_check.py --emulate sm{t}: TileLang builds for sm_{t} as PTX\n'
                'args=()\n'
                'for a in "$@"; do\n'
                '  case "$a" in\n'
                f'    -arch=sm_{t}) args+=("-gencode=arch=compute_{t},code=compute_{t}") ;;\n'
                '    --cubin|-cubin) args+=("--fatbin") ;;\n'
                f'    *) args+=("${{a//code=sm_{t}/code=compute_{t}}}") ;;\n'
                '  esac\n'
                'done\n'
                f'echo "${{args[*]}}" >> {wrapper}.log\n'
                f'exec {real} "${{args[@]}}"\n')
            tmp.chmod(0o755)
            os.replace(tmp, wrapper)
        nvcc.get_nvcc_compiler = lambda: str(wrapper)

    _after_import('tilelang.contrib.nvcc', patch_tilelang_nvcc)

    def patch_flashinfer(context):
        # SGLang's set_cuda_arch() puts the emulated capability into FLASHINFER_CUDA_ARCH_LIST, which FlashInfer's JIT
        # builds: its SASS (sm_XY) would not load here, the PTX does
        flags_list = context.CompilationContext.get_nvcc_flags_list
        t = f'{major}{minor}'

        def ptx_flags_list(self, *args, **kwargs):
            return [f.replace(f'code=sm_{t}', f'code=compute_{t}') if f.startswith('-gencode=') else f
                    for f in flags_list(self, *args, **kwargs)]
        context.CompilationContext.get_nvcc_flags_list = ptx_flags_list

    _after_import('flashinfer.compilation_context', patch_flashinfer)
    suffix = 'a' if real_major >= 9 else ''
    os.environ.setdefault('CUTE_DSL_ARCH', f'sm_{real_major}{real_minor}{suffix}')


def main():
    argv = sys.argv[1:]
    if '--' not in argv:
        raise SystemExit(__doc__)
    split = argv.index('--')
    head, command = argv[:split], argv[split + 1:]
    target = None
    if '--emulate' in head:
        i = head.index('--emulate')
        target = head[i + 1]
        head = head[:i] + head[i + 2:]
    archs = head
    valid = all(a.startswith('sm') and a[2:].isdigit() for a in archs + ([target] if target else []))
    if not archs or not command or not valid:
        raise SystemExit(__doc__)
    report = tempfile.mkdtemp(prefix='xtier-arch-check-')
    hook = Path(report, 'hook')
    hook.mkdir()
    (hook / 'sitecustomize.py').write_text(
        'import os\n'
        'if os.environ.get("XTIER_ARCH_CHECK"):\n'
        '    import sys\n'
        f'    sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})\n'
        '    import arch_check\n'
        '    if os.environ.get("XTIER_EMULATE_SM"):\n'
        '        arch_check.emulate(os.environ["XTIER_EMULATE_SM"])\n'
        '    arch_check.install(os.environ["XTIER_ARCH_CHECK"].split(","), os.environ["XTIER_ARCH_CHECK_DIR"])\n')
    env = dict(os.environ, XTIER_ARCH_CHECK=','.join(archs), XTIER_ARCH_CHECK_DIR=report,
               PYTHONPATH=os.pathsep.join([str(hook)] + [p for p in [os.environ.get('PYTHONPATH')] if p]))
    if target:
        env['XTIER_EMULATE_SM'] = target[2:]
    # a Python script or module runs on this interpreter; anything else (a launch script) as it is, with the hook
    # reaching its Python processes through PYTHONPATH
    python = command[0] == '-m' or command[0].endswith('.py')
    rc = subprocess.call(([sys.executable] if python else []) + command, env=env)
    merged = {}
    for f in Path(report).glob('*.jsonl'):
        for line in f.read_text().splitlines():
            k, v = json.loads(line)
            if v != 'ok' or k not in merged:
                merged[k] = v
    failed = {k: v for k, v in merged.items() if v != 'ok'}
    for k in sorted(failed):
        print(f'ARCH-CHECK FAIL {k}: {failed[k]}')
    print(f'ARCH-CHECK {",".join(archs)}{" (emulating " + target + ")" if target else ""}: '
          f'{len(merged) - len(failed)} kernel(s) compile, {len(failed)} fail (run exit code {rc}; report {report})')
    raise SystemExit(1 if failed or rc else 0)


if __name__ == '__main__':
    main()
