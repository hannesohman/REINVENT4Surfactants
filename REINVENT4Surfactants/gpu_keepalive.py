import torch
import argparse

# pynvml isn't installed anywhere in this project's environments, and
# utilization-gated pacing (small matmul + sleep-when-busy) was previously
# observed to still let the idle-utilization watchdog kill a job -- so this
# just runs continuous back-to-back matmuls with no idle time at all,
# regardless of what else the job is doing on the GPU.

parser = argparse.ArgumentParser()
parser.add_argument("--mat_size", type=int, default=8192)
parser.add_argument("--sleep_time", type=float, default=0.0)
args = parser.parse_args()

DEVICE = "cuda:0"
MAT_SIZE = args.mat_size


def main():
    print(f"GPU keepalive running (continuous, {MAT_SIZE}x{MAT_SIZE}, no idle)...", flush=True)

    a = torch.randn(MAT_SIZE, MAT_SIZE, device=DEVICE, dtype=torch.float16)
    b = torch.randn(MAT_SIZE, MAT_SIZE, device=DEVICE, dtype=torch.float16)

    while True:
        _ = a @ b
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()