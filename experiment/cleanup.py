#!/usr/bin/env python


import argparse
import pathlib
import shutil

import pandas as pd

parser = argparse.ArgumentParser(description="Cleanup failed experiments")
parser.add_argument(
    "-n",
    "--dry-run",
    dest="dryrun",
    action="store_true",
    help="Do dry run without deleting experiments",
)
parser.add_argument(
    "-e",
    "--epochs",
    dest="epochs",
    default=5,
    type=int,
    help="Number of epochs to use as a threshold",
)
parser.add_argument(
    "root", type=str, default="results", help="Folder containing the experiment folders"
)

if __name__ == "__main__":

    args = parser.parse_args()
    path = pathlib.Path(args.root)

    assert path.exists(), f"Couldn't find {path}"

    for expdir in path.iterdir():

        delete = False
        logfile = expdir / "logs.csv"
        if logfile.exists():
            df = pd.read_csv(logfile)
            if len(df) < args.epochs:
                delete = True
        else:
            delete = True

        if delete:
            if args.dryrun:
                print(expdir)
            else:
                shutil.rmtree(expdir)
