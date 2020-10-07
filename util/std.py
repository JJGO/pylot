from contextlib import redirect_stdout, redirect_stderr, contextmanager
import os
import pathlib


@contextmanager
def redirect_std(outfile, mode="a"):
    with open(outfile, mode) as fo:
        with redirect_stdout(fo):
            yield
    return
