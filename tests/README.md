# tests

Unit tests: kernel numerics against the reference, expert placement, multi-GPU emulation.

Most need a GPU and real checkpoints. CI (`.github/workflows/check.yml`) runs what needs neither: `tools/repo_check.py`,
`console_check.py`, `planner_check.py`. What to run by hand before a release: [docs/release-checklist.md](../docs/release-checklist.md).
A test prints FAIL and exits non-zero when it fails.
