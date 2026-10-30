# Qwen3.8 Flash Next on one or two 24 GB GeForce GPUs (RTX 4090 / RTX 3090)

`run-qwen.sh`: upstream SGLang main + XTier for a single RTX 4090 (SM89) or RTX 3090 (SM86) with about 120 GB of host
memory, e.g. a rented one-GPU instance. Paths and options come from the environment (see the script's header); the
defaults assume such an instance's data disk at `/root/autodl-tmp`: the Python environment in `xtier-env/`, the
checkpoint in `models/qwen38`, a converted tokenizer in `models/qwen38-tokenizer` (as in
[`configs/qwen38-pro5000-nps4`](../qwen38-pro5000-nps4/README.md)) and, optionally, the cache ranking profile and the
MTP draft token map in `xtier-files/`. On a machine without a CUDA toolkit (only the CUDA 13 wheels), SGLang's JIT
links with `-L$CUDA_HOME/lib64 -lcudart`: add `lib/libcudart.so -> libcudart.so.13` and `lib64 -> lib` in the wheels'
`nvidia/cu13`. The driver must support CUDA 13.0 (580 or newer). With `TP=2` the script runs on two such GPUs, see
[Two GPUs](#two-gpus-tp2).

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

## Two GPUs (`TP=2`)

`TP=2 CUDA_VISIBLE_DEVICES=0,1 bash run-qwen.sh` runs tensor parallel over two such GPUs. SGLang's loader gives each
rank half the intermediate size of every expert, and the KV and linear-attention heads are split as well, so each GPU
holds half of every hot expert and half the KV cache. The script caps SGLang's SSM state cache at 16 entries, as in
production's long-context config (otherwise it takes what is left, ~2.4 GB per GPU, far more than two requests'
prefixes use), and gives the room to hot experts: 180 per layer (MTP 160) leave ~4.4 GB per GPU. The CPU engine runs one
process per rank, each on its own group of NUMA nodes (`XTIER_TP_NODES=auto`, set by the script; with one GPU per CPU
socket each rank works on its GPU's socket) with `tools/cpu_budget.py --tp 2` threads. A rank computes half of every
missed expert with half the threads, so the CPU time per layer stays that of one GPU; the gain is the larger hot-expert
pool (hit rate 0.61 at 48 slots, 0.92 at 180) and the second GPU's share of prefill. The GPUs need no P2P or NVLink:
without GPU P2P SGLang turns its custom all-reduce off by itself and all-reduces through NCCL. With MTP the draft keeps
the whole vocabulary (SGLang cuts the draft token map's head from the target's lm_head by global token ids, which a
vocab-parallel shard does not have). Host memory: ~9 GiB more than one GPU (a second process and PLE row cache), ~83 GiB
at the peak of a 120K-token prompt.

Two GPUs with different memory: SGLang refuses to start when one has under 90% of the other's free memory;
`SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0` turns that into a warning. Both ranks then have the smaller GPU's memory to
work with, the rest of the larger one stays unused: on 24 + 16 GiB (the second RTX 4090 held at 16 GiB with
`tools/vram_limit.py`), `XTIER_CACHE_SLOTS=64` leaves 4.7 GiB on the smaller GPU, copy test 0/8, decode 51.1 tok/s at 32K
and 49.4 at 120K, 60.0 s to the first token at 120K.

For GPUs of different size or speed, the second one can hold only hot experts instead (`TP=1`, the fast GPU first):
`XTIER_GPU2_SLOTS=auto CUDA_VISIBLE_DEVICES=0,1 bash run-qwen.sh` (`xtier/cache/second.py`). On the same 24 + 16 GiB: 48 + 113
slots per layer, decode 57.3 tok/s at 32K and 55.9 at 120K, 58.9 s to the first token at 120K; with the second GPU about 1.2
times slower (`XTIER_GPU2_SLOW=1`) 54.5 and 54.2. Details in
[docs/reports/2026-10-31-second-gpu-pool.md](../../docs/reports/2026-10-31-second-gpu-pool.md).

Checked on two RTX 4090 as well (SGLang main `6162b07199`; a shared PCIe switch, no P2P; 15 engine threads per rank):
copy test 0/8 at one and two requests, with MTP as well; PPL over 1K-token segments on average +0.0013 from the RTX 3090
pair. One GPU → two, decode at 32K: 44.7 → 61.5 tok/s, with MTP 81.9; the first token at 120K stays about 60 s, so
on this host the gain is decode only.

Checked on the two RTX 3090 of the instance above (one per CPU socket, no P2P between them; 14 engine threads per
rank): copy test 0/8 at one and two requests, with MTP as well; PPL over 1K-token segments on average +0.002 from one
GPU; a 120K-token prompt without running out of memory. Single requests, one GPU → two:

| Context | First token | Prefill tok/s | Decode tok/s | Decode with MTP |
|---|---:|---:|---:|---:|
| 32K | 22.6 → 15.1 s | 1,454 → 2,181 | 42.1 → 59.6 | 39.1 → 71.5 |
| 120K | 72.6 → 47.0 s | 1,654 → 2,554 | 41.8 → 61.7 | 39.0 → 69.1 |

Details in [docs/reports/2026-10-30-tp.md](../../docs/reports/2026-10-30-tp.md).
