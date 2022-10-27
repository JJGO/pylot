def get_fs_backend(path):
    import subprocess
    import io
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
