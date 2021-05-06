import os


def restrict_GPU_pytorch(gpuid, use_cpu=False):
    """
    gpuid: str, comma separated list "0" or "0,1" or even "0,1,3"
    """
    if not use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpuid

        print("Using GPU:{}".format(gpuid))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("Using CPU")


def freest_GPU():
    from lums.gpu.query import GPUQuery

    q = GPUQuery.instance()
    gpus = q.run()
    # The max shenanigans below is just to do argmax
    freest_gpu = max(range(len(gpus)), key=lambda i: gpus[i]["memory_free"])
    return freest_gpu


def GPU_pytorch(i=None):
    if i is None:
        i = freest_GPU()
    restrict_GPU_pytorch(str(i))
    import torch

    device = torch.device("cuda")

    import torch.backends.cudnn as cudnn

    cudnn.benchmark = True

    return device


def gpu_stats(visible=False):
    import io
    import os
    import subprocess
    import pandas as pd
    stats = [
        "utilization.gpu",
        "utilization.memory",
        "memory.total",
        "memory.free",
        "memory.used",
    ]

    cmd = ['nvidia-smi', f"--query-gpu={','.join(stats)}", '--format=csv,noheader,nounits']
    output = subprocess.check_output(cmd)
    df = pd.read_csv(io.StringIO(output.decode()), names=stats)
    if visible:
        gpus = os.environ.get("CUDA_VISIBLE_DEVICES", '')
        if len(gpus) == 0:
            gpus = []
        else:
            gpus = [int(i) for i in gpus.split(',')]
        df = df.loc[gpus]
    return df
