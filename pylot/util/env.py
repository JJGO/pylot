import platform
import subprocess
import sys
import time
from datetime import datetime

import jc
import psutil
import torch


def platform_info():
    return {
        "system": platform.system(),
        "release": platform.release(),
        "architecture": "".join(platform.architecture()),
        "version": platform.version(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "node": platform.node(),
    }


def sys_info():
    return {
        "executable": sys.executable,
        "version": sys.version,
    }


def pip_packages():
    pip_list_output = subprocess.check_output(
        [sys.executable, "-m", "pip", "list"], text=True, stderr=subprocess.DEVNULL
    )
    packages = jc.parse("pip-list", pip_list_output)
    packages = dict(list(map(lambda x: x.values(), packages)))
    return packages


def hardware_info():
    # https://www.thepythoncode.com/article/get-hardware-system-information-python
    cpufreq = psutil.cpu_freq()
    svmem = psutil.virtual_memory()
    return {
        "cpu": {
            "physical_cores": str(psutil.cpu_count(logical=False)),
            "total_cores": str(psutil.cpu_count(logical=True)),
            "freq_max": f"{cpufreq.max/1000:.2f}GHz",
            "freq_min": f"{cpufreq.min/1000:.2f}GHz",
            "freq_current": f"{cpufreq.current/1000:.2f}GHz",
            "percent": f"{psutil.cpu_percent()}%",
        },
        "memory": {
            "total": svmem.total,
            "available": svmem.available,
            "used": svmem.used,
            "percent": str(svmem.percent) + "%",
        },
    }


def torch_info():
    info = torch.utils.collect_env.get_env_info()._asdict()
    # fmt: off
    if info['pip_packages'] is not None:
        info['pip_packages'] = {k: v for k, v in [line.split("==") for line in info["pip_packages"].split("\n")]}
    if info['conda_packages'] is not None:
        info["conda_packages"] = {k: v for k, v, *_ in [line.split() for line in info["conda_packages"].split("\n")]}
    if info['nvidia_gpu_models'] is not None:
        info["nvidia_gpu_models"] = [line for line in info["nvidia_gpu_models"].split("\n")]
    # fmt: on
    return info


def when_info():
    return {
        "timestamp": time.time(),
        "data": datetime.now().astimezone().replace(microsecond=0).isoformat(),
    }


def full_env_info():
    return {
        "platform": platform_info(),
        "sys": sys_info(),
        "pip": pip_packages(),
        "hardware": hardware_info(),
        "torch": torch_info(),
        "when": when_info(),
    }
