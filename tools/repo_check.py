"""Checks of this repository that need no GPU, no torch and no SGLang (CI runs them). Exit status 1 on a finding.

- every Python file parses, every shell script passes bash -n, no Windows line ends;
- xtier/env_names.txt lists every XTIER_* name the sources mention (the plugin warns at start about settings not listed;
  --write rewrites it), and every name the run scripts and console entries of configs/ set is one of them;
- every function of the native libraries the Python side names exists in csrc/, and the mailbox header declares all of
  that library's exports (mailbox.cpp includes the header, so the compiler checks their types);
- a tests/*_check.py that can print FAIL can also exit non-zero.

  python tools/repo_check.py [--write]
"""
import ast
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('xtier', 'csrc', 'tools', 'tests', 'bench')
SUFFIXES = {'.py', '.cpp', '.h', '.sh', '.txt', '.md', '.toml', '.yml'}
NAME = re.compile(r'\bXTIER_[A-Z0-9_]*[A-Z0-9]\b')
findings = []


def files(*dirs, suffixes=SUFFIXES):
    for d in dirs:
        for p in sorted((ROOT / d).rglob('*')):
            if p.is_file() and p.suffix in suffixes and '__pycache__' not in p.parts and 'results' not in p.parts:
                yield p


def text(p):
    return p.read_bytes().decode('utf-8', 'replace')


def syntax():
    for p in files(*SOURCES, 'configs', suffixes={'.py'}):
        try:
            ast.parse(text(p))
        except SyntaxError as e:
            findings.append(f'{p.relative_to(ROOT)}: {e}')
    bash = shutil.which('bash')
    for p in files(*SOURCES, 'configs', suffixes={'.sh'}):
        if bash and subprocess.run([bash, '-n', str(p)], capture_output=True).returncode:
            findings.append(f'{p.relative_to(ROOT)}: bash -n fails')
    for p in files(*SOURCES, 'configs', 'docs'):
        if b'\r' in p.read_bytes():
            findings.append(f'{p.relative_to(ROOT)}: Windows line ends')


def settings(write):
    known = set()
    listing = ROOT / 'xtier' / 'env_names.txt'
    for p in files(*SOURCES, suffixes={'.py', '.cpp', '.h', '.sh'}):
        known.update(NAME.findall(text(p)))
    for p in files('configs', suffixes={'.sh'}):   # what a run script itself reads
        known.update(re.findall(r'\$\{?(XTIER_[A-Z0-9_]*[A-Z0-9])\b', text(p)))
    wanted = ''.join(f'{n}\n' for n in sorted(known)).encode()
    if write:
        listing.write_bytes(wanted)
    elif not listing.exists() or listing.read_bytes() != wanted:
        findings.append('xtier/env_names.txt is out of date: tools/repo_check.py --write')
    for p in files('configs', suffixes={'.py', '.sh'}):
        for n in sorted(set(NAME.findall(text(p))) - known):
            findings.append(f'{p.relative_to(ROOT)}: sets {n}, which nothing reads')


def natives():
    csrc = '\n'.join(text(p) for p in files('csrc', suffixes={'.cpp', '.h'}))
    python = '\n'.join(text(p) for p in files('xtier', suffixes={'.py'}))
    used = set(re.findall(r'\b(?:lib|c)\.(xtier_[a-z0-9_]+)\b', python))
    for p in (ROOT / 'xtier/engine/__init__.py', ROOT / 'xtier/tables/__init__.py'):   # the names of the bindings' tables
        used |= set(re.findall(r"'(xtier_[a-z0-9_]+)'", text(p)))
    for name in sorted(used):
        if not re.search(rf'\b{name}\b', csrc):
            findings.append(f'xtier/ calls {name}, which csrc/ does not define')
    exports = set(re.findall(r'^[a-z][^\n(]*\b(xtier_mailbox_[a-z0-9_]+)\(', text(ROOT / 'csrc/mailbox/mailbox.cpp'), re.M))
    declared = set(re.findall(r'^[a-z][^\n(]*\b(xtier_mailbox_[a-z0-9_]+)\(', text(ROOT / 'csrc/include/xtier_mailbox.h'), re.M))
    for name in sorted(exports ^ declared):
        findings.append(f'{name}: in {"mailbox.cpp" if name in exports else "xtier_mailbox.h"} only')


def tests():
    for p in files('tests', suffixes={'.py'}):
        s = text(p)
        if p.name.endswith('_check.py') and 'FAIL' in s and not re.search(r'sys\.exit|raise SystemExit', s):
            findings.append(f'{p.relative_to(ROOT)}: can print FAIL but always exits 0')


syntax()
settings('--write' in sys.argv)
natives()
tests()
for f in findings:
    print(f)
print(f'{len(findings)} findings')
sys.exit(1 if findings else 0)
