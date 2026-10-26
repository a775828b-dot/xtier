"""Elastic expert cache (XTIER_CACHE_ELASTIC=1): extra hot-expert slots in the GPU memory that only a prefill needs.

Between prefills the memory of their activations (and of SGLang's other transient buffers) lies idle. The partly
cached layers get address space for more slots than they start with (cache/pool.py LayerPool: S slots of address
space, the first `base` mapped for good); free memory backs extra slots, which the adaptive policy fills like any
victim (csrc/mailbox: the usable slots `active`). Before a forward that needs more memory than is free, the extra
slots go back: the policy pauses, a commit kernel applies what it posted, the copies in flight land, the experts of
the released slots leave the residency table (commands without an upload), their granules are unmapped (csrc/mailbox
xtier_mailbox_vmm_*). The arena pointers in the decode graphs never change. After a stretch of decode forwards
without such a forward, free memory backs extra slots again, a few at a time from a background thread.

What a forward needs is learnt from the forwards themselves (the larger of its peak allocated and peak reserved memory
over the amounts before it), kept by its size and context (the tokens the batch's requests hold: attention and
indexer workspaces can grow with it, and a long session must not make every later prefill look large). A need seen
bounds any forward no larger in either; taking the need as growing at most in proportion to the tokens and to the
context, it bounds a larger one too, scaled by both ratios up to XTIER_CACHE_ELASTIC_EXTRAPOLATE (2) times. The
tightest bound is the prediction; a forward no need seen bounds gets every extra slot released. Slots are added only
up to what leaves the usual small prefill (the 75th percentile of the recent needs of forwards up to
XTIER_CACHE_ELASTIC_SMALL, 4096 tokens) room on top of the reserve, so a coding agent's
turns do not release and refill them every time; a large prefill takes seconds and releases in a fraction of one. Only the device's
free memory counts (the allocator's cached blocks can be too fragmented for a forward's large allocations); when it
is short, the allocator's cache is emptied first.

Safety, so that a model never runs out of memory because of the extra slots:
- XTIER_CACHE_ELASTIC_RESERVE_GIB (2): free device memory never given to slots (lazy kernel loads, workspaces);
- before every forward that allocates (prefill, mixed, draft extend) the deficit to its predicted need x
  XTIER_CACHE_ELASTIC_MARGIN (1.25) plus the reserve is released;
- every XTIER_CACHE_ELASTIC_CHECK (16) decode forwards: free memory under XTIER_CACHE_ELASTIC_CRITICAL_GIB (1)
  releases every extra slot and stops growing for XTIER_CACHE_ELASTIC_COOLDOWN_S (600);
- an out-of-memory error in a forward while extra slots are held releases them all, empties the allocator's cache,
  turns the elastic cache off for the process and runs the forward once more.
Per layer (layout): address space for XTIER_CACHE_ELASTIC_MAX (2.0) x its configured slots
(XTIER_CACHE_ELASTIC_MAX_SLOTS: a number), at most every expert; XTIER_CACHE_ELASTIC_BASE (1.0) x them mapped for good.
XTIER_CACHE_ELASTIC_IDLE (32): decode forwards after an allocating one before slots are added again.
XTIER_CACHE_ELASTIC_EXTRAPOLATE (2): how far a need seen is scaled up to bound a larger forward.
XTIER_CACHE_ELASTIC_DEBUG=1: a log line for every need learnt and every prediction made while extra slots are held.
"""
import ctypes as C
import logging
import math
import os
import threading
import time

import torch

logger = logging.getLogger(__name__)
GIB = 2 ** 30
FORWARD = 'sglang.srt.model_executor.model_runner.ModelRunner.forward'
_governor = None


def enabled():
    return os.environ.get('XTIER_CACHE_ELASTIC', '0') == '1' and os.environ.get('XTIER_CACHE_ADAPT', '1') != '0'


def _env(name, default):
    value = os.environ.get(name, '')
    return float(value) if value.strip() else default


def layout(slots, experts):
    """(base, top) of a partly cached layer configured with `slots`: the slots mapped for good and the address space.
    XTIER_CACHE_ELASTIC_MAX (2.0, or XTIER_CACHE_ELASTIC_MAX_SLOTS) sets the top, XTIER_CACHE_ELASTIC_BASE (1.0) the
    base, both relative to `slots`: 2.0 / 1.0 adds slots in the prefill's memory to a layout; 1.0 / 0.1 keeps the
    layout's slots for decode but maps only a tenth for good, so that the launch can make more layers fully cached
    (they speed the prefill up) in the memory that frees. Without the elastic cache: (slots, slots)."""
    if not enabled() or slots <= 0 or slots >= experts:
        return slots, slots
    cap = os.environ.get('XTIER_CACHE_ELASTIC_MAX_SLOTS', '')
    top = int(cap) if cap.strip() else int(math.ceil(slots * _env('XTIER_CACHE_ELASTIC_MAX', 2.0)))
    top = max(1, min(experts, top))
    base = max(1, min(top, int(round(slots * _env('XTIER_CACHE_ELASTIC_BASE', 1.0)))))
    return base, top


class _Device:
    """__cuda_array_interface__ of reserved address space, for torch.as_tensor (no copy, not owned by torch)."""

    def __init__(self, ptr, shape):
        self.__cuda_array_interface__ = dict(shape=tuple(shape), typestr='|u1', data=(ptr, False), version=3)


_credit = [0]   # bytes the base slots given up so far leave unmapped (the elastic layers' plain bytes - mapped bytes)


def fit_base(lib, base, row_shapes):
    """The base of an elastic layer less the slots its arenas' granule rounding takes: the mapped base of every arena
    rounds up to a granule (2 MiB), so with the extra slots released the layers would hold up to four granules each
    more than plain slots; a model laid out to the last MiB (GLM-5.3: 0.6 GB free at its prefill peak) runs out of
    memory on that (10-26). A layer gives up a slot only when the credit the slots given up before leave cannot cover
    its rounding: the elastic layers so far never map more than their plain slots (GLM: one layer in three gives up
    a slot, MiniMax one in four, instead of every one)."""
    gran = lib.xtier_mailbox_vmm_granularity()
    if not gran:
        raise RuntimeError(lib.xtier_mailbox_error().decode())
    rows = [math.prod(r) for r in row_shapes]
    plain = base * sum(rows)
    b = base
    while b > 1 and sum(-(-b * r // gran) * gran for r in rows) - plain > _credit[0]:
        b -= 1
    _credit[0] -= sum(-(-b * r // gran) * gran for r in rows) - plain
    return b


def arena(lib, slots, row_shape, mapped, device):
    """uint8 [slots, *row_shape] over reserved device address space, the first `mapped` slots backed."""
    stride = math.prod(row_shape)
    va, gran = C.c_void_p(), C.c_size_t()
    if lib.xtier_mailbox_vmm_reserve(slots * stride, C.byref(va), C.byref(gran)) < 0:
        raise RuntimeError(lib.xtier_mailbox_error().decode())
    if lib.xtier_mailbox_vmm_map(va, 0, mapped * stride) < 0:
        raise RuntimeError(lib.xtier_mailbox_error().decode())
    t = torch.as_tensor(_Device(va.value, (slots,) + tuple(row_shape)), device=device)
    if t.data_ptr() != va.value:
        raise RuntimeError('xtier elastic cache: the arena tensor does not alias its address space')
    return t


def _check(lib, rc):
    if rc < 0:
        raise RuntimeError(lib.xtier_mailbox_error().decode())
    return rc


class Governor:
    """Sizes the extra slots of the elastic layers against the free device memory: `extra` slots over every layer's
    base (its planner / XTIER_CACHE_SLOTS count), at most its address space; all layers move together."""

    def __init__(self, pools, lib, commit):
        self.pools, self.lib, self.commit = pools, lib, commit
        self.device = pools[0].arena[0].device
        self.extra = 0
        self.cap = max(p.S - p.base for p in pools)
        self.row = [sum(a[0].numel() for a in p.arena) for p in pools]   # bytes of one slot per layer
        self.level = sum(self.row)   # bytes of one extra slot in every layer
        self.reserve = _env('XTIER_CACHE_ELASTIC_RESERVE_GIB', 2.0) * GIB
        self.critical = _env('XTIER_CACHE_ELASTIC_CRITICAL_GIB', 1.0) * GIB
        self.margin = _env('XTIER_CACHE_ELASTIC_MARGIN', 1.25)
        self.idle = int(_env('XTIER_CACHE_ELASTIC_IDLE', 32))
        self.every = max(1, int(_env('XTIER_CACHE_ELASTIC_CHECK', 16)))
        self.cooldown = _env('XTIER_CACHE_ELASTIC_COOLDOWN_S', 600.0)
        self.lock = threading.Lock()    # one resize at a time
        self.stop = threading.Event()   # asks a growth in progress to end
        self.grower = None
        self.since = 0                  # decode forwards since the last allocating one
        self.decodes = 0
        self.calm_until = 0.0
        self.off = False
        self.seen = {}                  # draft? -> {(tokens, context) rounded down: the largest need seen there}
        self.extrapolate = _env('XTIER_CACHE_ELASTIC_EXTRAPOLATE', 2.0)
        self.debug = os.environ.get('XTIER_CACHE_ELASTIC_DEBUG', '0') == '1'
        self.recent = []                # the needs of the last small allocating forwards
        self.small = int(_env('XTIER_CACHE_ELASTIC_SMALL', 4096))
        self.released = self.added = 0
        self.last_log = 0.0

    # --- sizes ---

    def predict(self, tokens, draft, ctx):
        """Bytes a forward of `tokens` tokens at a context of `ctx` tokens needs on top of what is in use (None: no
        need seen bounds it): the tightest bound from the needs seen, each scaled by how much larger this forward is
        in tokens and in context, if at most `extrapolate` times."""
        if ctx is None or tokens < 1:
            return None
        best = None
        for (t, c), n in self.seen.get(draft, {}).items():
            f = max(1.0, tokens / t) * max(1.0, ctx / c)
            if f <= self.extrapolate and (best is None or n * f < best):
                best = n * f
        return None if best is None else best * self.margin

    @staticmethod
    def _grid(x):
        """x rounded down to eighths of an octave."""
        return 2 ** (math.floor(8 * math.log2(x)) / 8)

    def learn(self, tokens, draft, need, ctx):
        """A forward's need, at its size and context rounded down to eighths of an octave (the largest need seen there;
        rounding down keeps every bound valid)."""
        if tokens <= self.small:
            self.recent = (self.recent + [need])[-32:]
        if self.debug:
            logger.info('xtier elastic cache: learnt %.2f GiB for %d tokens%s at %s', need / GIB, tokens,
                        ' (draft)' if draft else '', 'an unknown context' if ctx is None else f'a {ctx}-token context')
        if ctx is None or tokens < 1 or ctx < 1:
            return
        seen = self.seen.setdefault(draft, {})
        key = (self._grid(tokens), self._grid(ctx))
        seen[key] = max(need, seen.pop(key, 0))   # re-inserted: the most recent last
        if len(seen) > 1024:   # the oldest goes: a bound lost only makes the predictions more careful
            del seen[next(iter(seen))]

    def bytes(self, k0, k1):
        """Memory between k0 and k1 extra slots."""
        return sum(r * (min(k1, p.S - p.base) - min(k0, p.S - p.base)) for p, r in zip(self.pools, self.row))

    def _slots(self, p, k):
        return min(p.S, p.base + k)

    @staticmethod
    def free_bytes(empty=False):
        """Free device memory (after emptying the allocator's cache when `empty`)."""
        if empty:
            torch.cuda.empty_cache()
        return torch.cuda.mem_get_info()[0]

    # --- resizing ---

    def _stop_growth(self):
        g = self.grower
        if g is not None and g.is_alive():
            self.stop.set()
            g.join()
        self.stop.clear()

    def release(self, target, why):
        """Down to `target` extra slots (at least 0) before the memory is needed; returns the slots released."""
        self._stop_growth()
        with self.lock:
            target = max(0, min(target, self.extra))
            if target >= self.extra:
                return 0
            lib = self.lib
            t0 = time.perf_counter()
            _check(lib, lib.xtier_mailbox_pool_pause(1))
            try:
                for _ in range(400):
                    self.commit()
                    torch.cuda.synchronize()
                    rc = _check(lib, lib.xtier_mailbox_pool_quiesce(25))
                    if rc == 0:
                        break
                else:
                    raise RuntimeError('xtier elastic cache: the expert policy did not settle')
                _check(lib, lib.xtier_mailbox_pool_shrink(target))
                self.commit()                # the evictions leave the residency table ...
                torch.cuda.synchronize()     # ... before the granules go
                freed = self.bytes(target, self.extra)
                for p in self.pools:
                    keep = self._slots(p, target)
                    for a in p.arena:
                        _check(lib, lib.xtier_mailbox_vmm_unmap(C.c_void_p(a.data_ptr()), keep * a[0].numel()))
                    p.active = keep
                n, self.extra = self.extra - target, target
            finally:
                lib.xtier_mailbox_pool_resume()
        self.released += 1
        self._log(f'released {n} extra slots per layer ({freed / GIB:.1f} GiB) for {why} in '
                  f'{(time.perf_counter() - t0) * 1e3:.0f} ms; +{self.extra} of +{self.cap} left')
        return n

    def _grow(self, target, step):
        """Background: map granules for more usable slots, `step` slots at a time while memory allows."""
        try:
            torch.cuda.set_device(self.device)
            self._grow_steps(target, step)
        except Exception as e:   # noqa: BLE001 - a failed growth turns the elastic cache off, serving goes on
            self.off = True
            logger.error('xtier elastic cache: growth failed (%s); elastic cache off for this process', e)

    def _grow_steps(self, target, step):
        lib = self.lib
        while not self.stop.is_set():
            with self.lock:
                old = self.extra
                new = min(target, old + step, self.fit(torch.cuda.mem_get_info()[0] - self.reserve - self.buffer()))
                if new <= old:
                    return
                try:
                    for p in self.pools:
                        a0, a1 = self._slots(p, old), self._slots(p, new)
                        for a in p.arena:
                            st = a[0].numel()
                            _check(lib, lib.xtier_mailbox_vmm_map(C.c_void_p(a.data_ptr()), a0 * st, (a1 - a0) * st))
                except RuntimeError as e:   # out of memory on the way: back to where it was, no growth for a while
                    for p in self.pools:
                        for a in p.arena:
                            lib.xtier_mailbox_vmm_unmap(C.c_void_p(a.data_ptr()), self._slots(p, old) * a[0].numel())
                    self.calm_until = time.monotonic() + self.cooldown
                    self._log(f'growth stopped at +{old} slots: {e}')
                    return
                _check(lib, lib.xtier_mailbox_pool_grow(new))
                for p in self.pools:
                    p.active = self._slots(p, new)
                self.extra = new
                self.added += 1
            if new >= target:
                self._log(f'+{self.extra} of +{self.cap} extra slots per layer usable '
                          f'({self.bytes(0, self.extra) / GIB:.1f} GiB)')
                return

    def buffer(self):
        """Room kept free for the usual small prefill when slots are added: the 75th percentile of the recent needs
        of the forwards up to `small` tokens."""
        if len(self.recent) < 4:
            return 0
        return sorted(self.recent)[int(0.75 * (len(self.recent) - 1))] * self.margin

    def fit(self, room):
        """The most extra slots, from the current count up, whose memory fits in `room` bytes."""
        k = self.extra
        while k < self.cap and self.bytes(self.extra, k + 1) <= room:
            k += 1
        return k

    def keep(self, deficit):
        """The most extra slots, from the current count down, that free at least `deficit` bytes."""
        k = self.extra
        while k > 0 and self.bytes(k, self.extra) < deficit:
            k -= 1
        return k

    def emergency(self, why, disable=False):
        """Every extra slot back now; no growth for the cooldown (or ever)."""
        try:
            self.release(0, why)
        finally:
            self.calm_until = time.monotonic() + self.cooldown
            if disable:
                self.off = True
            logger.warning('xtier elastic cache: %s: extra slots released%s', why,
                           ', elastic cache off for this process' if disable else f', no growth for {self.cooldown:.0f} s')

    def _log(self, text, every=30.0):
        now = time.monotonic()
        if now - self.last_log >= every or 'stopped' in text or 'released' in text:
            self.last_log = now
            logger.info('xtier elastic cache: %s (%d releases, %d growth steps so far)', text, self.released,
                        self.added)

    # --- forward hooks ---

    def before_alloc(self, tokens, draft, ctx):
        """A forward that allocates: release what its predicted need and the reserve do not leave free."""
        self.since = 0
        self._stop_growth()   # before the free memory is read
        if self.extra <= 0:
            return
        what = (f'a forward of {tokens} tokens{" (draft)" if draft else ""} at '
                f'{"an unknown context" if ctx is None else f"a {ctx}-token context"}')
        need = self.predict(tokens, draft, ctx)
        if self.debug:
            logger.info('xtier elastic cache: %s predicts %s, %.1f GiB free, +%d extra', what,
                        'nothing' if need is None else f'{need / GIB:.2f} GiB', self.free_bytes() / GIB, self.extra)
        if need is None:
            self.release(0, f'{what}, no need seen bounds it')
            return
        deficit = need + self.reserve - self.free_bytes()
        if deficit > 0:
            deficit = need + self.reserve - self.free_bytes(empty=True)
        if deficit > 0:
            self.release(self.keep(deficit), f'{what} (needs {need / GIB:.1f} GiB)')

    def after_decode(self):
        self.since += 1
        self.decodes += 1
        if self.decodes % self.every:
            return
        if self.extra > 0 and self.free_bytes() < self.critical:
            self.emergency(f'free memory under {self.critical / GIB:.1f} GiB')
            return
        # not before a request's prefill has been seen: the decode-sized forwards of SGLang's start-up (CUDA graph
        # warm-ups, the adaptive speculation's captures) come before serving, and the captures need the memory
        if (self.off or not self.seen or self.since < self.idle or self.extra >= self.cap
                or time.monotonic() < self.calm_until or (self.grower is not None and self.grower.is_alive())):
            return
        from xtier.cache import pool
        if not pool.started():
            return
        torch.cuda.empty_cache()   # what the last prefill left cached in the allocator
        target = self.fit(torch.cuda.mem_get_info()[0] - self.reserve - self.buffer())
        if target <= self.extra:
            return
        self.grower = threading.Thread(target=self._grow, args=(target, 2), name='xtier-elastic-grow', daemon=True)
        self.grower.start()


def register(pools, lib, commit):
    """The elastic layers, once attached to the adaptive policy (cache/pool.py fill_all)."""
    global _governor
    _governor = Governor(pools, lib, commit)
    g = _governor
    bases = sorted({p.base for p in pools})
    logger.info('xtier elastic cache: %d layers, %s slots mapped, up to %d extra (%.2f GiB per extra slot in every '
                'layer); reserve %.1f GiB, critical %.1f GiB, margin %.2f', len(pools),
                bases[0] if len(bases) == 1 else f'{bases[0]}..{bases[-1]}', g.cap, g.level / GIB, g.reserve / GIB,
                g.critical / GIB, g.margin)


def _decode_like(forward_batch):
    mode = forward_batch.forward_mode
    return mode.is_decode_or_idle() or mode.is_target_verify()


def _context(forward_batch):
    """The tokens the batch's requests hold, from the host-side lengths (None: not there)."""
    n = getattr(forward_batch, 'seq_lens_sum', None)
    if n is None:
        cpu = getattr(forward_batch, 'seq_lens_cpu', None)
        n = None if cpu is None else int(cpu.sum())
    return None if n is None else int(n)


def _oom(e):
    return isinstance(e, torch.OutOfMemoryError) or (isinstance(e, RuntimeError) and 'out of memory' in str(e))


def _guarded(g, fn, *args):
    """A governor step that must never fail the forward: an error turns the elastic cache off (slots stay as they are,
    the policy runs on)."""
    try:
        fn(*args)
    except Exception as e:   # noqa: BLE001
        g.off = True
        try:
            g.lib.xtier_mailbox_pool_resume()
        except Exception:   # noqa: BLE001
            pass
        logger.error('xtier elastic cache: %s failed (%s); elastic cache off for this process', fn.__name__, e)


def _forward(original, runner, forward_batch, *args, **kwargs):
    """Around SGLang's ModelRunner.forward: resize the extra slots around the forwards that allocate."""
    g = _governor
    if g is None or g.off or torch.cuda.is_current_stream_capturing():
        return original(runner, forward_batch, *args, **kwargs)
    if _decode_like(forward_batch):
        out = original(runner, forward_batch, *args, **kwargs)
        _guarded(g, g.after_decode)
        return out
    ids = getattr(forward_batch, 'input_ids', None)
    tokens = int(ids.shape[0]) if ids is not None and ids.dim() > 0 else 0
    draft = bool(getattr(runner, 'is_draft_worker', False))
    ctx = _context(forward_batch)
    _guarded(g, g.before_alloc, tokens, draft, ctx)
    torch.cuda.reset_peak_memory_stats()
    before, held = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    retry = False
    try:
        out = original(runner, forward_batch, *args, **kwargs)
    except Exception as e:   # noqa: BLE001 - only out of memory with extra slots held is handled
        if not _oom(e) or g.extra <= 0:
            raise
        retry = True
    if retry:   # out of the except block: the failed attempt's frames and their tensors are released by now
        _guarded(g, g.emergency, f'out of memory in a forward of {tokens} tokens', True)
        g.off = True
        torch.cuda.empty_cache()
        out = original(runner, forward_batch, *args, **kwargs)
    g.learn(tokens, draft, max(0, torch.cuda.max_memory_allocated() - before, torch.cuda.max_memory_reserved() - held),
            ctx)
    return out


def install(registry, hook_type):
    """Scheduler processes: the forward hook (the governor appears once the elastic layers are attached)."""
    if not enabled():
        return False
    registry.register(FORWARD, _forward, hook_type.AROUND)
    return True
