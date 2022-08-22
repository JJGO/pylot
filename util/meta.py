import inspect
import re
import functools


# Decorator to monkey patch static
# kwarg vars to a function
def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


def separate_kwargs(kwargs, func):

    func_params = inspect.signature(func).parameters

    func_kwargs = {k: v for k, v in kwargs.items() if k in func_params}
    other_kwargs = {k: v for k, v in kwargs.items() if k not in func_params}
    return func_kwargs, other_kwargs


def get_default_kwargs(func):
    return {
        k: v.default
        for k, v in inspect.signature(func).parameters.items()
        if v.default != inspect.Parameter.empty
    }


# Taken from fastcore.foundation
def delegates(to=None, keep=False, but=None, via="kwargs"):
    "Decorator: replace `**kwargs` in signature with params from `to`"
    if but is None:
        but = []

    def _f(f):
        if to is None:
            to_f, from_f = f.__base__.__init__, f.__init__
        else:
            to_f, from_f = to, f
        from_f = getattr(from_f, "__func__", from_f)
        to_f = getattr(to_f, "__func__", to_f)
        if hasattr(from_f, "__delwrap__"):
            return f
        sig = inspect.signature(from_f)
        sigd = dict(sig.parameters)
        k = sigd.pop(via)
        s2 = {
            k: v
            for k, v in inspect.signature(to_f).parameters.items()
            if v.default != inspect.Parameter.empty and k not in sigd and k not in but
        }
        sigd.update(s2)
        if keep:
            sigd[via] = k
        else:
            from_f.__delwrap__ = to_f
        from_f.__signature__ = sig.replace(parameters=sigd.values())
        return f

    return _f


def _store_attr(self, **attrs):
    for n, v in attrs.items():
        setattr(self, n, v)
        self.__stored_args__[n] = v


def store_attr(names=None, self=None, but=None, **attrs):
    "Store params named in comma-separated `names` from calling context into attrs in `self`"
    but = but or []
    fr = inspect.currentframe().f_back
    args, varargs, keyw, locs = inspect.getargvalues(fr)
    if self is None:
        self = locs[args[0]]
    if not hasattr(self, "__stored_args__"):
        self.__stored_args__ = {}
    if attrs:
        return _store_attr(self, **attrs)

    ns = re.split(", *", names) if names else args[1:]
    _store_attr(self, **{n: fr.f_locals[n] for n in ns if n not in but})


def partial(f, *args, order=None, **kwargs):
    "Like `functools.partial` but also copies over docstring and name"
    fnew = functools.partial(f, *args, **kwargs)
    fnew.__doc__ = f.__doc__
    fnew.__name__ = f.__name__
    if order is not None:
        fnew.order = order
    elif hasattr(f, "order"):
        fnew.order = f.order
    return fnew


def custom_dir(c, add: list):
    "Implement custom `__dir__`, adding `add` to `cls`"
    return dir(type(c)) + list(c.__dict__.keys()) + add


class GetAttr:
    "Inherit from this to have all attr accesses in `self._xtra` passed down to `self.default`"
    _default = "default"

    def _component_attr_filter(self, k):
        if k.startswith("__") or k in ("_xtra", self._default):
            return False
        xtra = getattr(self, "_xtra", None)
        return xtra is None or k in xtra

    def _dir(self):
        return [
            k
            for k in dir(getattr(self, self._default))
            if self._component_attr_filter(k)
        ]

    def __getattr__(self, k):
        if self._component_attr_filter(k):
            attr = getattr(self, self._default, None)
            if attr is not None:
                return getattr(attr, k)
        raise AttributeError(k)

    def __dir__(self):
        return custom_dir(self, self._dir())

    #     def __getstate__(self): return self.__dict__
    def __setstate__(self, data):
        self.__dict__.update(data)
