"""vram_limit.py FREE_MIB WATCH_PID [READY_FILE]: a smaller GPU on this one. Takes the GPU's free memory except FREE_MIB
(one cudaMalloc through torch) and holds it until process WATCH_PID exits; READY_FILE, if given, gets the MiB left free
once the memory is held. configs/ds41-mini-5090sim uses it for an RTX 5090's 32 GB on the 72 GB RTX PRO 5000: every
allocator of the server (torch, CUDA graphs, VMM slots, FlashInfer workspaces) then sees the same limit."""
import os
import sys
import time

import torch

free_mib, watch = int(sys.argv[1]), int(sys.argv[2])
free, total = torch.cuda.mem_get_info()
hold = free - free_mib * 2**20
if hold <= 0:
    sys.exit(f'vram_limit: only {free >> 20} MiB free, nothing to hold for {free_mib} MiB')
block = torch.empty(hold, dtype=torch.uint8, device='cuda')
left, _ = torch.cuda.mem_get_info()
print(f'vram_limit: holding {hold >> 20} MiB of {total >> 20} MiB; {left >> 20} MiB left for pid {watch}', flush=True)
if len(sys.argv) > 3:
    with open(sys.argv[3], 'w') as f:
        f.write(f'{left >> 20}\n')
while os.path.exists(f'/proc/{watch}'):
    time.sleep(1)
