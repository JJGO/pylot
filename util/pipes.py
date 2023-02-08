import os
import pathlib
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from io import StringIO

# @contextmanager
# def redirect_std(outfile, mode="a"):
#     with open(outfile, mode) as fo:
#         with redirect_stdout(fo):
#             yield
#     return

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


@contextmanager
def redirect_std(outfile, errfile, mode="a", unbuffered=True):
    with open(outfile, mode) as fo, open(errfile, mode) as fe:
        if unbuffered:
            fo = Unbuffered(fo)
            fe = Unbuffered(fe)
        with redirect_stdout(fo), redirect_stderr(fe):
            yield
    return


@contextmanager
def quiet_std():
    with redirect_std("/dev/null", "/dev/null", unbuffered=False):
        yield


@contextmanager
def quiet_stdout():
    dev_null = open("/dev/null", "a")
    with redirect_stdout(dev_null):
        yield


@contextmanager
def temporary_save_path(filepath):
    """Yields a path where to save a file and moves it
    afterward to the provided location (and replaces any
    existing file)
    This is useful to avoid processes monitoring the filepath
    to break if trying to read when the file is being written.
    Note
    ----
    The temporary path is the provided path appended with .save_tmp
    """
    filepath = pathlib.Path(filepath)
    tmppath = filepath.with_suffix(filepath.suffix + ".save_tmp")
    assert not tmppath.exists(), "A temporary saved file already exists."
    yield tmppath
    if not tmppath.exists():
        raise FileNotFoundError("No file was saved at the temporary path.")
    if filepath.exists():
        os.remove(filepath)
    os.rename(tmppath, filepath)


class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout
