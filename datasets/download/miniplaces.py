#!/usr/bin/env python
import shutil
import pathlib
import subprocess


def download_miniplaces():
    root = pathlib.Path()
    download_cmds = [
        'curl -O http://miniplaces.csail.mit.edu/data/data.tar.gz',
        'curl -O https://raw.githubusercontent.com/CSAILVision/miniplaces/master/data/categories.txt',
        'curl -O https://raw.githubusercontent.com/CSAILVision/miniplaces/master/data/train.txt',
        'curl -O https://raw.githubusercontent.com/CSAILVision/miniplaces/master/data/val.txt',
        'tar -xvzf data.tar.gz images/',
        'rm data.tar.gz',
    ]

    for cmd in download_cmds:
        subprocess.run(cmd.split(' '))

    # mv images/x x
    for f in ('train', 'val', 'test'):
        fromf = (root / 'images' / f)
        if fromf.exists():
            tof = (root / f).as_posix()
            shutil.move(fromf.as_posix(), tof)

    (root / 'images').rmdir()

    # Read category files
    categories = {}  # name -> int
    with open(root / 'categories.txt', 'r') as f:
        for line in f.read().strip().split('\n'):
            p, i = line.split(' ')
            categories[p[1:]] = i

    from_to = {c: c.split('/')[1] for c in categories}

    # Move train files

    trash = set()
    for fromf, tof in from_to.items():

        (root / 'val' / tof).mkdir(parents=True, exist_ok=True)

        fromf = (root / 'train' / fromf).as_posix()
        tof = (root / 'train' / tof).as_posix()

        shutil.move(fromf, tof)
        trash.add("/".join(fromf.split("/")[:-1]))

    # remove train empty folders
    for folder in sorted(trash, key=lambda x: -len(x)):
        f = pathlib.Path(folder)
        if f.exists():
            f.rmdir()

    # inverse categories
    icategories = {i: c.split('/')[1] for c, i in categories.items()}
    # map files to their folders
    val = {}
    with open(root / 'val.txt', 'r') as f:
        for line in f.read().strip().split('\n'):
            fromf, c = line.split(' ')
            val[fromf] = icategories[c]

    # Relocate val files
    for fromf, tof in val.items():
        fromf = (root / fromf).as_posix()
        tof = (root / 'val' / tof).as_posix()
        shutil.move(fromf, tof)


if __name__ == "__main__":
    download_miniplaces()
