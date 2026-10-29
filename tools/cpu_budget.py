"""Expert worker threads for this process's CPUs: the CPUs of its cgroup cpuset (a container's; as the engine, which
binds workers anywhere in it; without one, the affinity mask), capped by a cgroup v2 / v1 CPU quota, less one for the
scheduler, rounded down to a multiple of the NUMA nodes that hold those CPUs (the engine gives every worker node the
same number of threads). Prints the number; launch scripts use it for XTIER_THREADS. With --tp N the number is per
rank: one scheduler per rank reserved, the rest split over the N ranks, each rounded down to a multiple of its share
of the NUMA nodes (XTIER_TP_NODES=auto gives each rank its own group of nodes).

  python tools/cpu_budget.py [--reserve N] [--tp N]
"""
import glob
import os
import sys


def cpulist(text):
    cpus = set()
    for part in text.strip().split(','):
        if part:
            a, _, b = part.partition('-')
            cpus.update(range(int(a), int(b or a) + 1))
    return cpus


def quota_cpus():
    """CPUs worth of the cgroup CPU quota, or None without one."""
    for quota, period in (('/sys/fs/cgroup/cpu.max', None),
                          ('/sys/fs/cgroup/cpu/cpu.cfs_quota_us', '/sys/fs/cgroup/cpu/cpu.cfs_period_us')):
        try:
            q = open(quota).read().split()
            p = q[1] if period is None else open(period).read().strip()
        except (OSError, IndexError):
            continue
        if q[0] in ('max', '-1'):
            return None
        return max(1, int(q[0]) // int(p))
    return None


def cgroup_cpus():
    """CPUs of this process's cgroup cpuset (v2: the cgroup or its nearest ancestor with cpuset.cpus.effective; v1: the
    cpuset hierarchy), or None."""
    try:
        lines = open('/proc/self/cgroup').read().splitlines()
    except OSError:
        return None
    for line in lines:
        _, controllers, path = line.split(':', 2)
        if controllers and 'cpuset' not in controllers.split(','):
            continue
        root, names = (('/sys/fs/cgroup', ('cpuset.cpus.effective',)) if not controllers else
                       ('/sys/fs/cgroup/cpuset', ('cpuset.effective_cpus', 'cpuset.cpus')))
        while True:
            for name in names:
                try:
                    cpus = cpulist(open(f'{root}{"" if path == "/" else path}/{name}').read())
                except OSError:
                    continue
                if cpus:
                    return cpus
            if path in ('', '/'):
                break
            path = path.rsplit('/', 1)[0] or '/'
    return None


def threads(reserve=1, tp=1):
    allowed = cgroup_cpus() or os.sched_getaffinity(0)
    n = len(allowed)
    q = quota_cpus()
    if q is not None:
        n = min(n, q)
    nodes = sum(bool(cpulist(open(f).read()) & allowed) for f in glob.glob('/sys/devices/system/node/node[0-9]*/cpulist'))
    nodes = max(nodes, 1)
    if tp > 1:
        share = max(nodes // tp, 1)
        return max(share, (n - reserve) // tp // share * share)
    return max(nodes, (n - reserve) // nodes * nodes)


if __name__ == '__main__':
    tp = int(sys.argv[sys.argv.index('--tp') + 1]) if '--tp' in sys.argv else 1
    reserve = int(sys.argv[sys.argv.index('--reserve') + 1]) if '--reserve' in sys.argv else tp
    print(threads(reserve, tp))
