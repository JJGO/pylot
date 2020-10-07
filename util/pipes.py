from contextlib import redirect_stdout, redirect_stderr, contextmanager
import os
import pathlib
import sys

@contextmanager
def redirect_std(outfile, mode="a"):
    with open(outfile, mode) as fo:
        with redirect_stdout(fo):
            yield
    return

# Usage is like sys.stdout = Tee(filename, mode)
# Can also do redirect_stdout(Tee(filename, mode)) to do it temporariliy
class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
