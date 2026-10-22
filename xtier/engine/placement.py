"""CPU placement of the process that runs the CPU experts (one process, no tensor parallelism yet).

The expert workers pin themselves (one per CPU, NUMA topology from sysfs) and so does the mailbox dispatcher
(XTIER_PROBE_DISPATCH_CPU, read by the mailbox library; when unset, XTier picks the highest CPU without a worker,
preferring cores whose SMT siblings have none either). Everything else - Python, the scheduler, torch's own
threads - should stay off those CPUs: XTIER_MAIN_CPUS=auto applies "all allowed CPUs minus the workers and the
dispatcher" to every existing thread of the process, and threads created later inherit it.
"""

import logging
import os

from xtier import config
from xtier.engine import worker_cpus

logger = logging.getLogger(__name__)
_applied = []


def _siblings(cpu: int) -> set[int]:
    try:
        with open(f'/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list') as f:
            return set(config.cpulist(f.read()))
    except OSError:
        return {cpu}


def dispatch_cpu(workers: set[int]) -> int:
    if os.environ.get('XTIER_PROBE_DISPATCH_CPU'):
        return int(os.environ['XTIER_PROBE_DISPATCH_CPU'])
    free = sorted(os.sched_getaffinity(0) - workers, reverse=True)
    if not free:
        raise RuntimeError('xtier: no CPU left for the mailbox dispatcher; lower XTIER_THREADS')
    idle = [c for c in free if not (_siblings(c) & workers)]
    return (idle or free)[0]


def apply(threads: int) -> None:
    """Once per process, before the first expert worker starts."""
    if _applied:
        return
    _applied.append(True)
    workers = set(worker_cpus(threads))
    dispatcher = dispatch_cpu(workers)
    os.environ['XTIER_PROBE_DISPATCH_CPU'] = str(dispatcher)
    spec = os.environ.get('XTIER_MAIN_CPUS', '')
    if spec == 'auto':
        cpus = os.sched_getaffinity(0) - workers - {dispatcher}
    else:
        cpus = set(config.cpulist(spec)) & os.sched_getaffinity(0) if spec else set()
    if cpus:
        for tid in os.listdir('/proc/self/task'):
            try:
                os.sched_setaffinity(int(tid), cpus)
            except OSError:
                pass  # thread exited
    logger.info('xtier: %d expert workers, dispatcher CPU %d, other threads %s', threads, dispatcher,
                f'on {len(cpus)} CPUs' if cpus else 'unchanged')
