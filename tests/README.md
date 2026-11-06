# tests

Unit tests: kernel numerics against the reference, expert placement, multi-GPU emulation.

Most need a GPU and real checkpoints. CI (`.github/workflows/check.yml`) runs what needs neither: `tools/repo_check.py`,
`console_check.py`, `planner_check.py`. What to run by hand before a release: [docs/release-checklist.md](../docs/release-checklist.md).
A test prints FAIL and exits non-zero when it fails.

`engram_ragged_check.py` requires SGLang and checks packed DSpark verification against independent request hashes,
including unequal lengths, sequence starts, image-token boundaries, padding, and unchanged committed history.
It checks the CPU fallback and, when available, the CUDA hash kernel; it does not require a checkpoint.
