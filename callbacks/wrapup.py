import os
import s3fs
import getpass

class S3Copy:
    def __init__(self, experiment):
        self.experiment = experiment
        self._host = os.environ['S3_HOST']
        self._key = os.environ['S3_KEY']
        self._secret = os.environ['S3_SECRET']

    def __call__(self):
        user = getpass.getuser()
        remote_folder = f"results/omni/remote/{user}"
        rpath = remote_folder + str(self.experiment.path.absolute())
        s3driver = s3fs.S3FileSystem(self._host, self._key, self._secret)
        print(f"Copying experiment to {rpath}")
        s3driver.put(str(self.experiment.path), rpath, recursive=True)
