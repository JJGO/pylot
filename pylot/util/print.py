def printy(mapping):
    import yaml

    print(yaml.safe_dump(mapping))


def hsizeof(obj, recursive=True):
    from humanize import naturalsize
    if recursive:
        from pympler.asizeof import asizeof
        return naturalsize(asizeof(obj))
    else:
        import sys
        return naturalsize(sys.getsizeof(obj))
