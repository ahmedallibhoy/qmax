from __future__ import annotations

from typing import Callable, Optional

import equinox as eqx 

from ._introspect import Path, _rows
from ._internal import _update_field
from .hilbert_space import AbstractHilbertSpace


class IncompatibleDomainError(TypeError):
    pass


class AbstractExpressionTree(eqx.Module):
    domain: AbstractHilbertSpace = eqx.field(static=True)
    children: tuple[AbstractExpressionTree, ...] = eqx.field(default=(), kw_only=True)
    name: Optional[str] = eqx.field(default=None, static=True, kw_only=True)

    def _check_compatible(self, other: ScalarLike | AbstractExpressionTree):
        if isinstance(other, AbstractExpressionTree) and other.domain != self.domain:
            raise IncompatibleDomainError(
                f"{self} acts on {self.domain}, but {other} acts on {other.domain}")

    def path(self, parent_path: Optional[Path]=None, child_idx: Optional[int]=None) -> Path:
        # parent_path is None at the entry point of a traversal, where self is the root
        if parent_path is None:
            return Path(self.label)
        return parent_path.append(child_idx, self.label)

    def child_at(self, path: Path) -> AbstractExpressionTree:
        if not path:
            return self
        index, rest = path.descend()
        return self.children[index].child_at(rest)

    def leaves(self, parent_path: Optional[Path]=None, child_idx: Optional[int]=None) -> list[Path]:
        path = self.path(parent_path, child_idx)

        if not self.children:
            return [path]

        leaves = [op.leaves(path, idx) for idx, op in enumerate(self.children)]
        return [leaf for leaf_list in leaves for leaf in leaf_list]

    def set_at_path(
        self,
        update: Callable[[AbstractExpressionTree, Optional[Path], Optional[int]], AbstractExpressionTree],
        path: Path=Path(),
        parent_path: Optional[Path]=None,
        child_idx: Optional[int]=None, 
        **kwargs) -> AbstractExpressionTree:

        """
        Rebuilds self with update(op, parent_path, child_idx) applied to the node at path,
        where parent_path and child_idx locate that node relative to the root.
        """
        if path:
            index, new_path = path.descend()
            fn = lambda o: o.children[index]

            child = fn(self)
            new_child = child.set_at_path(
                update, new_path, self.path(parent_path, child_idx), index, **kwargs)
            return eqx.tree_at(fn, self, new_child)

        return update(self, parent_path, child_idx, **kwargs)

    def with_name(self, name: str, path: Path=Path()) -> AbstractExpressionTree:
        return self._at_path(lambda op, _, __: _update_field(op, "name", name), path)

    @property
    def label(self) -> str:
        return type(self).__name__ if self.name is None else self.name

    def __repr__(self) -> str:
        return self.label

    def tree(self) -> str:
        return "\n".join(line for line, _ in _rows(self))
