from collections import Counter
from datetime import datetime
import os
import platform
import psutil
import time
import subprocess
import sys
from torch.utils.collect_env import get_env_info


_EXCLUDES = [
    "SSH_CLIENT",
    "SSH_CONNECTION",
    "SSH_TTY",
    "TERM",
    "XDG_SESSION_ID",
    "XDG_RUNTIME_DIR",
    "P9K_TTY",
    "LESS_TERMCAP_mb",
    "LESS_TERMCAP_md",
    "LESS_TERMCAP_me",
    "LESS_TERMCAP_se",
    "LESS_TERMCAP_so",
    "LESS_TERMCAP_ue",
    "LESS_TERMCAP_us",
    "LS_COLORS",
    "GREP_COLOR",
    "GREP_COLORS",
    "P9K_SSH",
    "LSCOLORS",
    "CLICOLOR",
    "ANSIBLE_NOCOWS",
    "SSH_AUTH_SOCK",
    "SSH_AGENT_PID",
    "PORT0",
    "PORT1",
    "PORT2",
    "PORT3",
    "PORT4",
    "PORT5",
    "PORT6",
    "PORT7",
    "PORT8",
    "PORT9",
    "_CE_CONDA",
    "_CE_M",
    "NCURSES_NO_UTF8_ACS",
    "EDITOR",
    "VISUAL",
    "PAGER",
    "NCURSES_NO_UTF8_ACS",
    "LESS",
    "LESSOPEN",
    "MAIL",
    "CONDA_PROMPT_MODIFIER",
    "COMP_KNOWN_HOSTS_WITH_HOSTFILE",
    "is_vim",
    "tmux_version",
    "wg_is_keys_off",
    "LANG",
    "LANGUAGE",
    "S_COLORS",
    "XDG_DATA_DIRS",
    "MANPATH",
    "CONDA_SHLVL",
    "SHLVL",
    "TMUX",
    "TMUX_PANE",
    "TMUX_PLUGIN_MANAGER_PATH",
    "TERMCAP",
    "STY",
    "EOD",
    "WINDOW",
    "ET_VERSION",
    "FINGERS_ALT_ACTION",
    "FINGERS_COMPACT_HINTS",
    "FINGERS_COPY_COMMAND",
    "FINGERS_COPY_COMMAND_UPPERCASE",
    "FINGERS_CTRL_ACTION",
    "FINGERS_HIGHLIGHT_FORMAT",
    "FINGERS_HIGHLIGHT_FORMAT_NOCOLOR",
    "FINGERS_HIGHLIGHT_FORMAT_NOCOMPACT",
    "FINGERS_HIGHLIGHT_FORMAT_NOCOMPACT_NOCOLOR",
    "FINGERS_HINT_FORMAT",
    "FINGERS_HINT_FORMAT_NOCOLOR",
    "FINGERS_HINT_FORMAT_NOCOMPACT",
    "FINGERS_HINT_FORMAT_NOCOMPACT_NOCOLOR",
    "FINGERS_HINT_POSITION",
    "FINGERS_HINT_POSITION_NOCOMPACT",
    "FINGERS_KEYBOARD_LAYOUT",
    "FINGERS_MAIN_ACTION",
    "FINGERS_PATTERNS",
    "FINGERS_SELECTED_HIGHLIGHT_FORMAT",
    "FINGERS_SELECTED_HIGHLIGHT_FORMAT_NOCOLOR",
    "FINGERS_SELECTED_HIGHLIGHT_FORMAT_NOCOMPACT",
    "FINGERS_SELECTED_HIGHLIGHT_FORMAT_NOCOMPACT_NOCOLOR",
    "FINGERS_SELECTED_HINT_FORMAT",
    "FINGERS_SELECTED_HINT_FORMAT_NOCOLOR",
    "FINGERS_SELECTED_HINT_FORMAT_NOCOMPACT",
    "FINGERS_SELECTED_HINT_FORMAT_NOCOMPACT_NOCOLOR",
    "FINGERS_SHIFT_ACTION",
    "FINGERS_SYSTEM_COPY_COMMAND",
    "FINGERS_SYSTEM_OPEN_COMMAND",
    "FPATH",
]


def info_system():
    return {
        "OS": platform.system(),
        "architecture": "".join(platform.architecture()),
        "version": platform.version(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "node": platform.node(),
    }


def get_hardware_info():
    hw = {}
    # Get CPU info
    with open("/proc/cpuinfo", "r") as f:
        cpus = [l.split(":")[1].strip() for l in f.readlines() if "model name" in l]
        hw["cpus"] = dict(Counter(cpus))

    hw["ram"] = str(round(psutil.virtual_memory().total / (1024.0 ** 3))) + " GB"
    return hw


def get_full_env_info():

    system_env = info_system()

    # PyTorch env
    env_namedtuple = get_env_info()
    pytorch_env = dict(env_namedtuple._asdict())
    if pytorch_env["pip_packages"] is not None:
        pytorch_env["pip_packages"] = {
            k: v
            for k, v in [
                line.split("==") for line in pytorch_env["pip_packages"].split("\n")
            ]
        }
    if pytorch_env["conda_packages"]:
        pytorch_env["conda_packages"] = {
            k: v
            for k, v, *_ in [
                line.split() for line in pytorch_env["conda_packages"].split("\n")
            ]
        }

    if pytorch_env["nvidia_gpu_models"] is not None:
        pytorch_env["nvidia_gpu_models"] = [
            line for line in pytorch_env["nvidia_gpu_models"].split("\n")
        ]

    pytorch_env["executable"] = sys.executable

    # Env
    environ = dict(os.environ)

    for k in _EXCLUDES:
        environ.pop(k, None)

    # hw = get_hardware_info()
    when = dict(
        timestamp=time.time(),
        date=datetime.astimezone(datetime.now()),
    )

    return dict(sys=system_env, python=pytorch_env, os=environ, when=when)
