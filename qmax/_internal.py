from typing import Any, TypeVar

import dataclasses

T = TypeVar("T")

def _update_field(obj: T, name: str, new_value: Any) -> T:
    new_obj = object.__new__(type(obj))
    for field in dataclasses.fields(obj):
        value = new_value if field.name == name else getattr(obj, field.name)
        object.__setattr__(new_obj, field.name, value)
    return new_obj
