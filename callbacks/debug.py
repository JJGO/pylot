import torch.cuda
import subprocess


def nvidia_smi(experiment):
    def nvidia_smi_callback(_):
        print(subprocess.check_output(["nvidia-smi"]).decode(), flush=True)

    return nvidia_smi_callback


def CUDAMemory(experiment):
    def CUDAMemoryCallback(epoch):
        max_allocated = torch.cuda.max_memory_allocated() / 1024 ** 2
        allocated = torch.cuda.memory_allocated() / 1024 ** 2
        print(f"Allocated CUDA memory {allocated:.0f} / {max_allocated:.0f} MB")

    return CUDAMemoryCallback
