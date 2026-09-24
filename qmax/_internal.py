import dataclasses
from typing import Any, TypeVar

import equinox as eqx

T = TypeVar("T", bound=eqx.Module)


def _update_fields(obj: T, **updates: Any) -> T:
    # workaround for eqx.Modules since eqx.tree_at normally breaks
    # on objects without leafs
    new_obj = object.__new__(type(obj))
    for field in dataclasses.fields(obj):
        value = updates.get(field.name, getattr(obj, field.name))
        object.__setattr__(new_obj, field.name, value)
    return new_obj


def _update_field(obj: T, name: str, new_value: Any) -> T:
    return _update_fields(obj, **{name: new_value})


def _overrides(cls, name: str, base: type) -> bool:
    # checks whether cls overrides method name in base class
    return getattr(cls, name) is not getattr(base, name)
