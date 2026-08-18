from typing import Any, TypeVar

import dataclasses

T = TypeVar("T")

def _update_field(obj: T, name: str, new_value: Any) -> T:
    # workaround for eqx.Modules where eqx.tree_at would normally break
    #   see e.g. Operator.with_exponential
    new_obj = object.__new__(type(obj))
    for field in dataclasses.fields(obj):
        value = new_value if field.name == name else getattr(obj, field.name)
        object.__setattr__(new_obj, field.name, value)
    return new_obj

def _overrides(cls, name: str, base: type) -> bool:
    return getattr(cls, name) is not getattr(base, name)



