def remove_prefix(key: str, prefix: str) -> str:
    if key.startswith(prefix):
        key = key[len(prefix) :]
    return key


def remove_suffix(key: str, suffix: str) -> str:
    if key.endswith(suffix):
        key = key[:-len(suffix)]
    return key
