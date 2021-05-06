from pathlib import Path
import shutil

import typer


def cleanup_experiments(
    root: Path,
    dryrun: bool = typer.Option(False, '-n', '--dry-run', help='Perform dry-run without deleting the experiments'),
    min_epochs: int = typer.Option(3, '-e', '--epochs', help='Number of epochs to use as a threshold')
):

    for expdir in root.iterdir():

        delete = False

        logfile = expdir / 'logs.csv'
        if not logfile.exists():
            logfile = expdir / '0/logs.csv'

        if logfile.exists():
            with open(logfile, 'r') as f:
                epochs = sum(1 for _ in f) - 1
                if epochs < min_epochs:
                    delete = True
        else:
            delete = True

        if delete:
            if dryrun:
                print(expdir)
            else:
                shutil.rmtree(expdir)


if __name__ == '__main__':
    typer.run(cleanup_experiments)
