import subprocess
import tempfile


def fzf(choices):
    choices = "\n".join(map(str, choices))
    with tempfile.TemporaryFile('w') as input_file:
        print(choices, file=input_file)
        input_file.seek(0)
        choice = subprocess.check_output(['fzf'], stdin=input_file).decode().strip()
    return choice

