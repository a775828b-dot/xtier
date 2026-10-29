# Qwen3.8 Flash Next on one 24 GB GeForce GPU (RTX 4090 / RTX 3090)

`run-qwen.sh`: upstream SGLang main + XTier for a single RTX 4090 (SM89) or RTX 3090 (SM86) with about 120 GB of host
memory, e.g. a rented one-GPU instance. Paths and options come from the environment (see the script's header); the
defaults assume such an instance's data disk at `/root/autodl-tmp`: the Python environment in `xtier-env/`, the
checkpoint in `models/qwen38`, a converted tokenizer in `models/qwen38-tokenizer` (as in
[`configs/qwen38-pro5000-nps4`](../qwen38-pro5000-nps4/README.md)) and, optionally, the cache ranking profile and the
MTP draft token map in `xtier-files/`. On a machine without a CUDA toolkit (only the CUDA 13 wheels), SGLang's JIT
links with `-L$CUDA_HOME/lib64 -lcudart`: add `lib/libcudart.so -> libcudart.so.13` and `lib64 -> lib` in the wheels'
`nvidia/cu13`. The driver must support CUDA 13.0 (580 or newer).

Where things live:
- GPU (24 GB): the dense weights (~5 GB), an FP8 KV cache of one 128K context (1.5 GB; 12 KiB per token), 48 hot
  experts per layer (6 GiB; 40 with MTP, whose draft keeps all 512 experts of its MoE layer on the GPU), ~3.6 GB left
  for prefill activations.
- Host memory: all 48 layers' experts for the CPU engine (63 GiB), the PLE row cache (4 GiB) and the runtime: ~71 GiB
  at the peak of a 120K-token prompt.
- Data disk: the checkpoint, including the PLE n-gram table (48 GiB), read by rows (`XTIER_PLE_ROWS=nvme`).

The GPU generation is handled in the code, not here: below SM100 the expert cache decodes FP4 in software, below SM89
FP8 as well (`xtier/arch.py`); below SM90 QSA decode runs on the Qwen pack's split-key Triton kernels
(`packs/qwen4_exp/qsa_decode.py`), and on SM86 the pack reads the FP8 KV cache in software (`qsa_fp4.py`). SGLang
serves the FP8 attention projections with its Triton W8A8 block-FP8 GEMM on SM89 and Marlin (W8A16) on SM86.

Checked on a rented RTX 4090 (24 GB; 16 vCPUs of a Xeon without AVX-512 BF16, so the CPU engine's AVX2 kernels; 120 GB
of host memory): copy test 0/8 at one and two requests, with MTP as well; PPL over 1K-token segments on average +0.003
from the RTX PRO 5000 emulating SM89; a 120K-token prompt without running out of memory. Single requests there: 32K
context 19.3 s to the first token and 36.3 tok/s decode, 120K 60.9 s and 36.5 tok/s, bound by that host's CPU (14
engine threads; MTP does not help there). Details in
[docs/reports/2026-10-29-rtx40.md](../../docs/reports/2026-10-29-rtx40.md).

Checked on a rented RTX 3090 as well (24 GB; a Xeon Platinum 8358P without AVX-512 BF16, 28 engine threads; 180 GB of
host memory): copy test 0/8 at one and two requests, with MTP as well; PPL over 1K-token segments on
average -0.006 from the RTX PRO 5000 emulating SM86 with the same limits; a 120K-token prompt without running out of memory.
Single requests there: 32K context 22.6 s to the first token and 42.1 tok/s decode, 120K 72.6 s and 41.8 tok/s.
Details in [docs/reports/2026-10-29-rtx30.md](../../docs/reports/2026-10-29-rtx30.md).

A prompt-logprob request of a 4096-token text needs 1.9 GiB of logits in one piece: with 24 GB keep PPL segments at
1024 tokens (`bench/ppl_eval.py --tokens 1024`).
