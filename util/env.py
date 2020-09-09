from datetime import datetime
import os
import platform
import time
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
    "CONDA_SHLVL",
    "SHLVL",
    "TMUX",
    "TMUX_PANE",
    "TMUX_PLUGIN_MANAGER_PATH",
    "TERMCAP",
    "STY",
    "EOD" "WINDOW",
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


def get_full_env_info():

    system_env = info_system()

    # PyTorch env
    env_namedtuple = get_env_info()
    pytorch_env = dict(env_namedtuple._asdict())
    pytorch_env["pip_packages"] = {
        k: v
        for k, v in [
            line.split("==") for line in pytorch_env["pip_packages"].split("\n")
        ]
    }
    pytorch_env["conda_packages"] = {
        k: v
        for k, v, *_ in [
            line.split() for line in pytorch_env["conda_packages"].split("\n")
        ]
    }
    pytorch_env["nvidia_gpu_models"] = [
        line for line in pytorch_env["nvidia_gpu_models"].split("\n")
    ]
    pytorch_env["executable"] = sys.executable

    # Env
    environ = dict(os.environ)

    for k in _EXCLUDES:
        environ.pop(k, None)

    when = dict(timestamp=time.time(), date=datetime.astimezone(datetime.now()),)

    return dict(sys=system_env, python=pytorch_env, os=environ, when=when)
