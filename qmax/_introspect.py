from __future__ import annotations

from typing import Any, Union, Optional, TYPE_CHECKING

import dataclasses
from functools import reduce

from ._internal import _update_field

if TYPE_CHECKING:
    from .operator import Operator


BRANCH = "├"
PIPE = "|"
ANGLE = "└"
DASH = "─"
PAD = "    "


@dataclasses.dataclass
class Field:
    name: str 
    index: Optional[int]

    def __init__(self, name: str="", index: Optional[int]=None):
        self.name = name 
        self.index = index

    def __repr__(self) -> str:
        if self.index is None:
            return self.name 
        else:
            return f"{self.name}[{self.index}]"

    def __hash__(self) -> int:
        return hash((self.name, self.index))


def child_field(index: int) -> Field:
    """The Field addressing the child of an operator at the given index."""
    return Field("children", index)


@dataclasses.dataclass
class RenderTree:
    """
    Simplified representation of an operator expression tree for the purposes 
    of rendering. 
    """
    label: str
    count: Optional[Count] = None                
    children: list[RenderTree] = dataclasses.field(default_factory=list)

    def __str__(self) -> str:
        return "\n".join(line for line, _ in _rows(self))

    def __repr__(self) -> str:
        return self.__str__()


def _rows(
    node: RenderTree, 
    prefix: str="", 
    is_last: bool=True, 
    is_root: bool=True) -> list[tuple[str, Optional[Count]]]:

    if is_root:
        line, child_prefix = node.label, ""
    elif is_last:
        line = f"{prefix}{PAD + ANGLE}{DASH}{node.label}"
        child_prefix = prefix + PAD + " "
    else:
        line = f"{prefix}{PAD + BRANCH}{DASH}{node.label}"
        child_prefix = prefix + PAD + PIPE

    rows = [(line, node.count)]
    for idx, child in enumerate(node.children):
        rows += _rows(child, child_prefix, idx == len(node.children) - 1, False)
    return rows



@dataclasses.dataclass(frozen=True)
class Path:
    # tuple of (field, op_label)
    steps: tuple[tuple[Field, Operator], ...] = ()

    def append(self, field: Field, op: Operator) -> Path:
        return Path(self.steps + ((field, op),))

    def __repr__(self) -> str:
        if not self.steps:
            return "<empty>"
    
        _, op = self.steps[0]
        path_str = op.label() 
    
        for field, op in self.steps[1:]:
            path_str += f".{field} → {op.label()}"
        return path_str

    def __hash__(self) -> int:
        return hash(tuple((field, id(op)) for (field, op) in self.steps))


@dataclasses.dataclass
class Count:
    actions:     int = 0
    adj_actions: int = 0
    solves:      int = 0
    exp_actions: int = 0

    def __add__(self, other: Count) -> Count:
        if not isinstance(other, Count):
            return NotImplemented

        return Count(
            self.actions     + other.actions, 
            self.adj_actions + other.adj_actions, 
            self.solves      + other.solves, 
            self.exp_actions + other.exp_actions
        )

    def __rmul__(self, other: int) -> Count:
        if not isinstance(other, int):
            return NotImplemented

        return Count(
            other * self.actions,
            other * self.adj_actions, 
            other * self.solves, 
            other * self.exp_actions
        )

    def __repr__(self) -> str:
        args_list = [
            f"{attr}={getattr(self, attr)}" for attr in 
            ["actions", "adj_actions", "solves", "exp_actions"] if getattr(self, attr) ]
        args = reduce(lambda a, b: a + ", " + b, args_list)
        return f"{args}"


@dataclasses.dataclass
class CountDict:
    """
    Wrapped dictionary of counts corresponding to each leaf in an operator 
    expression tree, keyed by the paths to the leaves. 
    """

    ct_dict: dict[Path, Count] = dataclasses.field(default_factory=dict)

    def __getitem__(self, key: Path) -> Count:          
        return self.ct_dict[key]

    def __contains__(self, key: Path) -> bool:
        return key in self.ct_dict

    def __iter__(self):
        return self.ct_dict.__iter__()

    def __len__(self) -> int:
        return self.ct_dict.__len__()

    def keys(self) -> list[Path]:
        return self.ct_dict.keys()

    def values(self) -> list[Count]:
        return self.ct_dict.values()

    def items(self) -> list[Count]:
        return self.ct_dict.items()

    def __add__(self, other: CountDict) -> CountDict:
        if not isinstance(other, CountDict):
            return NotImplemented

        duplicates = [key for key in self if key in other]
        u1 = [key for key in self if key not in other]
        u2 = [key for key in other if key not in self]

        new_dict = {key: self[key] + other[key] for key in duplicates}
        new_dict |= {key: self[key] for key in u1}
        new_dict |= {key: other[key] for key in u2}
        return CountDict(new_dict)

    def __rmul__(self, other: int) -> CountDict:
        if not isinstance(other, int):
            return NotImplemented

        return CountDict({path: other * count for (path, count) in self.ct_dict.items()})

    #def __str__(self) -> str:
    #    string = ""
    #    for path, count in self.items():
    #        string += f"{path}: {count}\n"
    #    return string

    def to_tree(self) -> RenderTree:
        root = RenderTree(label="")
        index = {(): root}

        for path, count in self.ct_dict.items():
            for idx in range(1, len(path.steps) + 1):
                prefix = path.steps[:idx]
                if prefix not in index:
                    field, op = prefix[-1]
                    tree = RenderTree(label=op.label())
                    index[prefix[:-1]].children.append(tree)
                    index[prefix] = tree
            index[path.steps].count = count

        return root

    def __repr__(self) -> str:
        if not self.ct_dict:
            return "CountDict(empty)"

        rows = []
        for child in self.to_tree().children:
            rows += _rows(child)

        width = max(len(line) for line, _ in rows) + 4
        out = [
            line if c is None else f"{line} {'·' * (width - len(line) - 2)}  {c}"
            for line, c in rows
        ]

        total = reduce(lambda a, b: a + b, self.ct_dict.values())
        out.append("─" * (width - 1))
        out.append("total:".ljust(width - 1) + f"  {total}")
        return "\n".join(out)


type CountType = Union[CountDict, type(NotImplemented)]

def _to_ct_type(val: Union[CountType, dict]) -> CountType:
    if isinstance(val, CountDict):
        return val 
    if isinstance(val, dict):
        return CountDict(val)
    return val


@dataclasses.dataclass
class InterfaceCount:
    action:     CountType
    adj_action: CountType
    solve:      CountType
    exp_action: CountType

    def __init__(
        self, 
        action: Union[CountType, dict],
        adj_action: Union[CountType, dict],
        solve: Union[CountType, dict],
        exp_action: Union[CountType, dict]):

        self.action = _to_ct_type(action)
        self.adj_action = _to_ct_type(adj_action)
        self.solve = _to_ct_type(solve)
        self.exp_action = _to_ct_type(exp_action)

