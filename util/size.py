def hsize(obj):
    import sys
    from humanize import naturalsize
    return naturalsize(sys.getsizeof(obj))
