import os

def get_fs_backend(path):
    import subprocess
    import io
    import pandas as pd
    output = subprocess.check_output(['df', '-T', str(path)]).decode()
    buf = io.StringIO(output.replace('Mounted on', 'mountpoint'))
    df = pd.read_csv(buf, delim_whitespace=True)
    ser = df.iloc[0]
    fs = ser['Type']
    if fs in ('tmpfs', 'devtmps'):
        return 'mem'
    if fs in ('ext4', 'ext3', 'ext2', 'vfat', 'zfs'):
        return 'local'
    if fs in ('nfs', 'nfs4', 'afs', 'cifs', 'fuse.sshfs'):
        return 'remote'
    return 'unknown'



def scantree(path):
    """Recursively yield DirEntry objects for given directory."""
    for entry in os.scandir(path):
        if entry.is_dir(follow_symlinks=False):
            yield from scantree(entry.path)  # see below for Python 2.x
        elif entry:
            yield entry
