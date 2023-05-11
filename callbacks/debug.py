import time
import signal
import sys
import subprocess

from datetime import datetime

import stackprinter

SEPARATOR = "#" * 80


def inspect_job(job):
    if job.state == "RUNNING":
        cmd = ["scancel", job.job_id, "--signal", "USR2"]
        subprocess.check_call(cmd)
        print(job.stderr().split(SEPARATOR)[-1])
    else:
        print("Job state is not RUNNING")


def inspect_stack_handler(signum, frame):
    now = datetime.now().replace(microsecond=0).isoformat()
    print(SEPARATOR, file=sys.stderr)
    print(f"{now} Inspecting Stack", file=sys.stderr)

    stackprinter.show(frame)


class InspectStack:

    """The purpose of this callback is to use SIGUSR2 as a means of
    sampling the currect program stack and dumping it into stderr.

    This is useful for jobs that get stuck and unresponsive, so the
    callback will help determining where the process is located
    """

    def __init__(self, experiment):
        import stackprinter

        signal.signal(signal.SIGUSR2, inspect_stack_handler)

    def __call__(self):
        pass


######################################

import threading
import inspect
import os


def traceline_handler(signum, frame):
    info = inspect.getframeinfo(frame)
    now = datetime.now().replace(microsecond=0).isoformat()
    file = info.filename
    func = info.function
    lineno = info.lineno
    print(now, f'File "{file}", line {lineno}, in {func}', file=sys.stderr, flush=True)


def timer(pid, interval):
    threading.Timer(interval, timer, args=(pid, interval)).start()
    subprocess.check_call(["kill", "-USR2", str(pid)])


class TraceLine:
    """Callback that spawns a thread that will periodically sample the main job
    and print the program location (file and lineno) to stderr

    Useful to determine where frozen jobs get stuck at
    """

    def __init__(self, experiment, interval: int = 60):
        signal.signal(signal.SIGUSR2, traceline_handler)
        pid = os.getpid()
        timer(pid, interval)

    def __call__(self):
        pass


######################################

from ..util.torchutils import torch_traceback


class TorchTraceback:
    def __init__(self, experiment):
        torch_traceback()

    def __call__(self):
        pass


######################################
import torch.cuda
import subprocess

# from ..util.gpu import gpu_stats


def nvidia_smi(experiment):
    def nvidia_smi_callback(_):
        print(subprocess.check_output(["nvidia-smi"]).decode(), flush=True)

    return nvidia_smi_callback


# def GPUStats(experiment):
#     def GPUStatsCallback(_):
#         df = gpu_stats(visible=True)
#         print(df)
#         if torch.cuda.is_available():
#             allocated = torch.cuda.memory_allocated() / 1024 ** 2
#             max_allocated = torch.cuda.max_memory_allocated() / 1024 ** 2
#             print(f"Current Allocated CUDA memory {allocated:.2f} MB")
#             print(f"Maximum Allocated CUDA memory {max_allocated:.2f} MB")

#     return GPUStatsCallback
