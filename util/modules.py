def any_getattr(modules, attr):
    for module in reversed(modules):
        if hasattr(module, attr):
            return getattr(module, attr)
    module_names = [m.__name__ for m in modules]
    raise ImportError(f"Attribute {attr} not found in any of {module_names}")
