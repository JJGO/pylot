import torch.cuda
import subprocess
from ..util.gpu import gpu_stats


def nvidia_smi(experiment):
    def nvidia_smi_callback(_):
        print(subprocess.check_output(["nvidia-smi"]).decode(), flush=True)

    return nvidia_smi_callback


def GPUStats(experiment):
    def GPUStatsCallback(_):
        df = gpu_stats(visible=True)
        print(df)
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 ** 2
            max_allocated = torch.cuda.max_memory_allocated() / 1024 ** 2
            print(f"Current Allocated CUDA memory {allocated:.2f} MB")
            print(f"Maximum Allocated CUDA memory {max_allocated:.2f} MB")

    return GPUStatsCallback
