"""Host <-> GPU copy bandwidth over PCIe (is the RTX 5090's x8 link width only power saving that widens
under bus load?). Pinned 1 GiB buffer, copies for `seconds` each way, prints GB/s every second.
  python pcie_bw.py [seconds]
"""
import sys
import time

import torch

secs = float(sys.argv[1]) if len(sys.argv) > 1 else 15
n = 1 << 30
host = torch.empty(n, dtype=torch.uint8, pin_memory=True)
dev = torch.empty(n, dtype=torch.uint8, device="cuda")
for name, dst, src in (("H2D", dev, host), ("D2H", host, dev)):
    t_end = time.time() + secs
    while time.time() < t_end:
        t0, k = time.time(), 0
        while time.time() - t0 < 1:
            dst.copy_(src, non_blocking=True)
            k += 1
        torch.cuda.synchronize()
        print(f"{name} {k * n / (time.time() - t0) / 1e9:.1f} GB/s", flush=True)
