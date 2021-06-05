from functools import wraps

# Decorator that lets you have attributes in functions
# to keep track of persistent state without requiring
# a full class. Good way to avoid needing globals
def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


# Decorator for the Fluent API/Builder pattern
# A method, when decorated with newobj will return a clone of the
# object but the implementation can be transparent to that 
# https://kracekumar.com/post/100897281440/fluent-interface-in-python/
def newobj(method):
    @wraps(method)
    # Well, newobj can be decorated with function, but we will cover the case
    # where it decorated with method
    def inner(self, *args, **kwargs):
        obj = self.__class__.__new__(self.__class__)
        obj.__dict__ = self.__dict__.copy()
        method(obj, *args, **kwargs)
        return obj
    return inner
