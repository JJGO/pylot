def printy(mapping):
    import yaml

    print(yaml.safe_dump(mapping))


def hsize(obj):
    import sys
    from humanize import naturalsize

    return naturalsize(sys.getsizeof(obj))
