import os
from contextlib import contextmanager

@contextmanager
def tmp_env(**kwargs):
    backup = {}
    for k, v in kwargs.items():
        if k in os.environ:
            backup[k] = os.environ[k]
        os.environ[k] = v
    yield
    for k in kwargs:
        os.environ.pop(k)
        if k in backup:
            os.environ[k] = backup[k]
