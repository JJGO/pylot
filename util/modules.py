import importlib


def any_getattr(modules, attr):
    if "." in attr:
        module, obj = attr.rsplit(".", 1)
        from importlib import import_module

        return getattr(import_module(module), obj)
    for module in reversed(modules):
        if hasattr(module, attr):
            return getattr(module, attr)
    # Try absolute import
    if "." in attr:
        module, _, attr = attr.rpartition(".")
        if importlib.util.find_spec(module) is not None:
            module = importlib.import_module(module)
            if hasattr(module, attr):
                return getattr(module, attr)

    module_names = [m.__name__ for m in modules]
    raise ImportError(f"Attribute {attr} not found in any of {module_names}")
