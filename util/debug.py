
import inspect
from functools import wraps

def printcall(func):
    """
    Decorator to print function call details.

    This includes parameters names and effective values.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_args = inspect.signature(func).bind(*args, **kwargs).arguments
        func_args_str = ", ".join(map("{0[0]} = {0[1]!r}".format, func_args.items()))
        print(f"+{func.__module__}.{func.__qualname__} ( {func_args_str} )")
        result = func(*args, **kwargs)
        print(f"-{func.__module__}.{func.__qualname__} ( {func_args_str} ) -> {result}")
        return result

    return wrapper

def torch_traceback():
    import torch
    from rich.traceback import install
    install(show_locals=True)
    def repr(self):
        return f"Tensor<{', '.join(map(str, self.shape))}|{str(self.dtype)[6:]}|{str(self.device)}>"
    torch.Tensor.__repr__ = repr




