import subprocess


def nvidia_smi(experiment):

    def nvidia_smi_callback(experiment, _):
        print(subprocess.check_output(['nvidia-smi']).decode(), flush=True)
    return nvidia_smi_callback
