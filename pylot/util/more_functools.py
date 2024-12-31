import functools
import inspect
import copy


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


# Decorator that lets you have attributes in functions
# to keep track of persistent state without requiring
# a full class. Good way to avoid needing globals
def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


# Memoization
def memoize(obj):
    cache = {}

    @functools.wraps(obj)
    def memoizer(*key):
        key = tuple(key)
        if key not in cache:
            cache[key] = obj(*key)
        return cache[key]

    memoizer.cache = cache

    return memoizer



# Decorator for the Fluent API/Builder pattern
# A method, when decorated with newobj will return a clone of the
# object but the implementation can be transparent to that
# https://kracekumar.com/post/100897281440/fluent-interface-in-python/
def newobj(method):
    @functools.wraps(method)
    # Well, newobj can be decorated with function, but we will cover the case
    # where it decorated with method
    def inner(self, *args, **kwargs):
        obj = self.__class__.__new__(self.__class__)
        obj.__dict__ = copy.deepcopy(self.__dict__)
        method(obj, *args, **kwargs)
        return obj

    return inner


#################################################

def _assign_args(instance, args, kwargs, function):
    def set_attribute(instance, parameter, default_arg):
        if not parameter.startswith("_"):
            setattr(instance, parameter, default_arg)

    def assign_keyword_defaults(parameters, defaults):
        for parameter, default_arg in zip(reversed(parameters), reversed(defaults)):
            set_attribute(instance, parameter, default_arg)

    def assign_positional_args(parameters, args):
        for parameter, arg in zip(parameters, args.copy()):
            set_attribute(instance, parameter, arg)
            args.remove(arg)

    def assign_keyword_args(kwargs):
        for parameter, arg in kwargs.items():
            set_attribute(instance, parameter, arg)

    def assign_keyword_only_defaults(defaults):
        return assign_keyword_args(defaults)

    def assign_variable_args(parameter, args):
        set_attribute(instance, parameter, args)

    (
        POSITIONAL_PARAMS,
        VARIABLE_PARAM,
        _,
        KEYWORD_DEFAULTS,
        _,
        KEYWORD_ONLY_DEFAULTS,
        _,
    ) = inspect.getfullargspec(function)
    POSITIONAL_PARAMS = POSITIONAL_PARAMS[1:]  # remove 'self'

    if KEYWORD_DEFAULTS:
        assign_keyword_defaults(parameters=POSITIONAL_PARAMS, defaults=KEYWORD_DEFAULTS)
    if KEYWORD_ONLY_DEFAULTS:
        assign_keyword_only_defaults(defaults=KEYWORD_ONLY_DEFAULTS)
    if args:
        assign_positional_args(parameters=POSITIONAL_PARAMS, args=args)
    if kwargs:
        assign_keyword_args(kwargs=kwargs)
    if VARIABLE_PARAM:
        assign_variable_args(parameter=VARIABLE_PARAM, args=args)


def auto_assign_arguments(function):
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        _assign_args(self, list(args), kwargs, function)
        function(self, *args, **kwargs)

    return wrapped
