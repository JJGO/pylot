from pydantic import validate_arguments


# This decorator is necessary because decorating directly
# results in the class not being a valid type
def validate_arguments_init(class_):
    class_.__init__ = validate_arguments(class_.__init__)
    return class_
