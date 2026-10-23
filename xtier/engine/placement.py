"""CPU placement of the processes that run the CPU experts: one per tensor-parallel rank. Each rank's MoE layers hold
that rank's slice (I / tp) of every expert and SGLang all-reduces the partial outputs, so ranks never talk to each
other; they only need CPUs of their own (ported from lkqmoe 0.5.3's tp.py, validated there on 2 x EPYC 7642 with two
5070 Ti).

- Worker nodes: one process uses every NUMA node with CPUs. With TP > 1 (or XTIER_TP_NODES set), the nodes are split
  into tp equal contiguous groups and rank r takes the group holding its GPU's NUMA node when every rank's GPU sits in
  a different group, else group r (XTIER_TP_NODES=auto, the default); per-rank lists 'nodes of rank 0|nodes of rank
  1|...' choose them. They reach the engine as XTIER_NODES before its topology is fixed (its first call).
- Expert workers pin themselves, XTIER_THREADS per rank (a multiple of the rank's node count). The mailbox dispatcher
  pins to XTIER_PROBE_DISPATCH_CPU (one value, or one per rank 'a|b'); unset: the highest CPU of the rank's nodes
  without a worker, preferring cores whose SMT siblings have none either.
- Everything else of the process - Python, the scheduler, torch's own threads - stays off those CPUs:
  XTIER_MAIN_CPUS=auto applies "the rank's CPUs minus the workers and the dispatcher" to every existing thread, and
  threads created later inherit it; a list (or one list per rank) sets them instead.
With TP the placement waits for the first MoE layer (the rank is known only then); set XTIER_TP_NODES in the launcher
so the plugin does not place the process before (see plugin.load).
"""

import logging
import os

from xtier import config
from xtier.engine import worker_cpus

logger = logging.getLogger(__name__)
_applied = []


def _format(cpus) -> str:
    cpus, out, i = sorted(cpus), [], 0
    while i < len(cpus):
        j = i
        while j + 1 < len(cpus) and cpus[j + 1] == cpus[j] + 1:
            j += 1
        out.append(str(cpus[i]) if i == j else f'{cpus[i]}-{cpus[j]}')
        i = j + 1
    return ','.join(out)


def per_rank(name: str, rank: int, size: int) -> str:
    """The value of `name`, or rank's part of 'a|b|...'."""
    value = os.environ.get(name, '')
    if '|' not in value:
        return value
    parts = value.split('|')
    if len(parts) != size:
        raise RuntimeError(f'xtier: {name}={value}: {len(parts)} groups for {size} TP ranks')
    return parts[rank]


def cpu_nodes(root='/sys/devices/system/node') -> dict[int, list[int]]:
    """NUMA nodes with CPUs -> their CPUs."""
    nodes = {}
    with open(f'{root}/online') as f:
        online = config.cpulist(f.read())
    for n in online:
        with open(f'{root}/node{n}/cpulist') as f:
            cpus = config.cpulist(f.read())
        if cpus:
            nodes[n] = cpus
    return nodes


def gpu_numa(device: int):
    import torch
    p = torch.cuda.get_device_properties(device)
    bus = '%04x:%02x:%02x.0' % (p.pci_domain_id, p.pci_bus_id, p.pci_device_id)
    try:
        with open(f'/sys/bus/pci/devices/{bus}/numa_node') as f:
            n = int(f.read())
    except (OSError, ValueError):
        return None
    return n if n >= 0 else None


def auto_nodes(rank: int, size: int, gpu_nodes, nodes) -> list[int]:
    """Rank's NUMA nodes: `gpu_nodes` = the NUMA node of each rank's GPU (None when unknown)."""
    ids = sorted(nodes)
    if len(ids) % size:
        raise RuntimeError(f'xtier: XTIER_TP_NODES=auto: {len(ids)} NUMA nodes with CPUs cannot be split over {size} '
                           f'TP ranks; list the nodes per rank')
    per = len(ids) // size
    groups = [ids[g * per:(g + 1) * per] for g in range(size)]
    owner = [next((g for g, group in enumerate(groups) if n in group), None) for n in gpu_nodes]
    if len(owner) == size and None not in owner and len(set(owner)) == size:
        return groups[owner[rank]]
    return groups[rank]


def _rank_gpus(rank: int, size: int):
    """NUMA nodes of every rank's GPU: SGLang runs rank r on device base + r (gpu_id_step 1)."""
    import torch
    base = torch.cuda.current_device() - rank
    if base < 0 or base + size > torch.cuda.device_count():
        return []
    return [gpu_numa(base + r) for r in range(size)]


def _siblings(cpu: int) -> set[int]:
    try:
        with open(f'/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list') as f:
            return set(config.cpulist(f.read()))
    except OSError:
        return {cpu}


def dispatch_cpu(workers: set[int], allowed: set[int], rank: int = 0, size: int = 1) -> int:
    chosen = per_rank('XTIER_PROBE_DISPATCH_CPU', rank, size)
    if chosen:
        return int(chosen)
    free = sorted(allowed - workers, reverse=True)
    if not free:
        raise RuntimeError('xtier: no CPU left for the mailbox dispatcher; lower XTIER_THREADS')
    idle = [c for c in free if not (_siblings(c) & workers)]
    return (idle or free)[0]


def deferred() -> bool:
    """True when the launcher asked for per-rank placement (the plugin then leaves it to the first MoE layer)."""
    return bool(os.environ.get('XTIER_TP_NODES'))


def apply(threads: int, rank: int = 0, size: int = 1) -> None:
    """Once per process, before the first expert worker starts."""
    if _applied:
        if _applied[0] != (rank, size) and size > 1:
            raise RuntimeError(f'xtier: TP rank {(rank, size)} in a process placed as {_applied[0]}')
        return
    _applied.append((rank, size))
    allowed = set(os.sched_getaffinity(0))
    mine = None
    if size > 1 or deferred():
        nodes = cpu_nodes()
        spec = per_rank('XTIER_TP_NODES', rank, size) or 'auto'
        mine = auto_nodes(rank, size, _rank_gpus(rank, size), nodes) if spec == 'auto' else config.cpulist(spec)
        missing = [n for n in mine if n not in nodes]
        if missing:
            raise RuntimeError(f'xtier: XTIER_TP_NODES: NUMA nodes {missing} have no CPUs')
        os.environ['XTIER_NODES'] = _format(mine)
        allowed &= {c for n in mine for c in nodes[n]}
    workers = set(worker_cpus(threads))   # fixes the engine topology (XTIER_NODES)
    dispatcher = dispatch_cpu(workers, allowed, rank, size)
    os.environ['XTIER_PROBE_DISPATCH_CPU'] = str(dispatcher)
    spec = per_rank('XTIER_MAIN_CPUS', rank, size)
    if spec == 'auto':
        cpus = allowed - workers - {dispatcher}
    else:
        cpus = set(config.cpulist(spec)) & os.sched_getaffinity(0) if spec else set()
    if cpus:
        for tid in os.listdir('/proc/self/task'):
            try:
                os.sched_setaffinity(int(tid), cpus)
            except OSError:
                pass  # thread exited
    where = f'TP rank {rank}/{size}, NUMA nodes {_format(mine)}, ' if mine is not None else ''
    logger.info('xtier: %s%d expert workers, dispatcher CPU %d, other threads %s', where, threads, dispatcher,
                f'on {len(cpus)} CPUs' if cpus else 'unchanged')
