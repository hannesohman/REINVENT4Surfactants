import time
import torch
import pynvml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mat_size", type=int, default=156)
parser.add_argument("--sleep_time", type=float, default=0.7)
args = parser.parse_args()

TARGET_UTIL = 70   # keep filler active only below this %
SLEEP_WHEN_BUSY = args.sleep_time  # seconds to sleep when GPU is busy
DEVICE = "cuda:0"
MAT_SIZE = args.mat_size

def gpu_util():
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    return pynvml.nvmlDeviceGetUtilizationRates(h).gpu

def main():
    print(f"GPU keepalive running... ({MAT_SIZE}x{MAT_SIZE} | {SLEEP_WHEN_BUSY}s)", flush=True)

    pynvml.nvmlInit()
    a = torch.randn(MAT_SIZE, MAT_SIZE, device=DEVICE, dtype=torch.float16)
    b = torch.randn(MAT_SIZE, MAT_SIZE, device=DEVICE, dtype=torch.float16)

    while True:
        util = gpu_util()
        if util < TARGET_UTIL:
            _ = a @ b
            torch.cuda.synchronize()
        else:
            time.sleep(SLEEP_WHEN_BUSY)

if __name__ == "__main__":
    main()