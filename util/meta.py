import inspect


def separate_kwargs(kwargs, func):

    func_params = inspect.signature(func).parameters

    func_kwargs = {k: v for k, v in kwargs.items() if k in func_params}
    other_kwargs = {k: v for k, v in kwargs.items() if k not in func_params}
    return func_kwargs, other_kwargs
