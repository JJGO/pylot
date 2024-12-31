import pathlib
from typing import Union
from ..util import S3Path


def fetch_dataset(name: str, dest: Union[str, pathlib.Path]):

    path = S3Path(f'data/{name}')
    target = pathlib.Path(dest)

    assert path.exists(), f'Folder s3://data/{name} does not exist'

    path._s3driver.get(str(path), target / name)


