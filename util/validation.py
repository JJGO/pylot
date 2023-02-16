import json
from typing import Dict, List, Type, Union

from pydantic import validate_arguments

JSONType = Type[
    Union[int, float, str, bool, None, Dict[str, "JSONType"], List["JSONType"]]
]

# This decorator is necessary because decorating directly
# results in the class not being a valid type
def validate_arguments_init(class_):
    class_.__init__ = validate_arguments(class_.__init__)
    return class_


def assert_json(data: JSONType):
    assert data == json.loads(json.dumps(data))
    return data
