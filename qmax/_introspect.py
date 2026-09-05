from __future__ import annotations

from typing import Any, Union, Optional, TYPE_CHECKING

import dataclasses
from functools import reduce

from ._internal import _update_field

if TYPE_CHECKING:
    from .operator import Operator


BRANCH = "├"
PIPE = "| "
ANGLE = "└"
DASH = "─"
PAD = ""


@dataclasses.dataclass
class RenderTree:
    """
    Simplified representation of an operator expression tree for the purposes 
    of rendering. 
    """
    label: str
    count: Optional[Count] = None                
    children: list[RenderTree] = dataclasses.field(default_factory=list)


def _rows(
    node: RenderTree | Operator, 
    prefix: str="", 
    is_last: bool=True, 
    is_root: bool=True) -> list[tuple[str, Optional[Count]]]:

    if is_root:
        line, child_prefix = node.label, ""
    elif is_last:
        line = f"{prefix}{PAD + ANGLE}{DASH}{node.label}"
        child_prefix = prefix + PAD + "  "
    else:
        line = f"{prefix}{PAD + BRANCH}{DASH}{node.label}"
        child_prefix = prefix + PAD + PIPE

    if isinstance(node, RenderTree):
        rows = [(line, node.count)]
    else:
        rows = [(line, None)]

    for idx, child in enumerate(node.children):
        rows += _rows(child, child_prefix, idx == len(node.children) - 1, False)
    return rows


@dataclasses.dataclass(frozen=True)
class Path:
    root_label: str = ""
    steps: tuple[tuple[int, str], ...] = ()

    @property
    def root(self) -> bool:
        return Path(self.root_label)

    def append(self, index: int, label: str) -> Path:
        return Path(self.root_label, self.steps + ((index, label),))

    def descend(self) -> tuple[int, Path]:
        (index, label), new_path = self.steps[0], self.steps[1:]
        return index, Path(label, new_path)

    @property
    def labels(self) -> list[str]:
        return [self.root_label] + [label for _, label in self.steps]

    def __repr__(self) -> str:
        return self.root_label + "".join(
            f".children[{index}] → {label}" for index, label in self.steps)

    def __len__(self) -> int:
        return len(self.steps)

        
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
            ["actions", "adj_actions", "solves", "exp_actions"] if getattr(self, attr)]
        args = ", ".join(args_list)
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

    def keys(self):
        return self.ct_dict.keys()

    def values(self):
        return self.ct_dict.values()

    def items(self):
        return self.ct_dict.items()

    def __or__(self, other: CountDict) -> CountDict:
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

    def render_trees(self) -> RenderTree:
        if not self.ct_dict:
            return RenderTree(label="")

        key = next(iter(self.ct_dict))
        roots = list(set([path.root for path in self.ct_dict.keys()]))
        trees = []

        for root in roots:
            root_node = RenderTree(label=root.root_label)
            index = {(): root_node}

            for path, count in self.ct_dict.items():
                if not path.root == root:
                    continue

                for idx in range(1, len(path) + 1):
                    prefix = path.steps[:idx]
                    if prefix not in index:
                        tree = RenderTree(label=prefix[-1][1])
                        index[prefix[:-1]].children.append(tree)
                        index[prefix] = tree
                index[path.steps].count = count

            trees += [root_node]

        return trees

    @property
    def total(self) -> Count:
        return reduce(lambda a, b: a + b, self.ct_dict.values())

    def __repr__(self) -> str:
        string = ""
        for path, count in self.items():
            string += f"{path}: {count}\n"
        return string

    def tree(self) -> str:
        if not self.ct_dict:
            return "CountDict(empty)"

        all_rows = []
        width = 0

        for tree in self.render_trees():
            rows = _rows(tree)
            width = max(width, max(len(line) + 4 for line, _ in rows))
            all_rows += rows

        out = [
            line if c is None else f"{line} {'·' * (width - len(line) - 2)}  {c}"
            for line, c in all_rows
        ]

        out.append("─" * (width - 1))
        out.append("total:".ljust(width - 1) + f"  {self.total}")
        return "\n".join(out)


type CountType = Union[CountDict, type(NotImplemented)]

def _to_ct_type(val: CountType | dict) -> CountType:
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
        action: CountType | dict,
        adj_action: CountType | dict,
        solve: CountType | dict,
        exp_action: CountType | dict):

        self.action = _to_ct_type(action)
        self.adj_action = _to_ct_type(adj_action)
        self.solve = _to_ct_type(solve)
        self.exp_action = _to_ct_type(exp_action)

