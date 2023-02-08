import fcntl
import os
from contextlib import contextmanager


@contextmanager
def lock_file(file_path: str):

    if not os.path.exists(file_path):
        raise FileNotFoundError("Cannot lock non-existent file")

    # Open the lock file in read/write mode
    file = open(file_path, "r+")

    # Acquire an exclusive lock on the lock file
    fcntl.lockf(file, fcntl.LOCK_EX)
    try:
        yield
    finally:
        # Release the lock
        fcntl.lockf(file, fcntl.LOCK_UN)
        file.close()


@contextmanager
def lock_folder(folder_path):
    lock = os.path.join(folder_path, ".lock")
    # Create lock file if it doesn't exist
    if not os.path.exists(lock):
        open(lock, "w").close()

    with lock_file(lock):
        yield


if __name__ == "__main__":

    N = 1_000
    workers = 50
    folder = "/tmp/testlock"
    counter = os.path.join(folder, "counter")

    def increment(_):
        with lock_folder(folder):
            with open(counter, "r") as f:
                n = int(f.read().strip()) + 1
            with open(counter, "w") as f:
                print(n, file=f)

    with open(counter, "w") as f:
        print("0", file=f)

    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=workers) as executor:
        list(executor.map(increment, range(N)))

    with open(counter, "r") as f:
        assert int(f.read().strip()) == N
