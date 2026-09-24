from typing import Optional

import jax.numpy as jnp
from jaxtyping import Array, ScalarLike
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, to_rgb

from .spaces.spatial_discretization import SpatiallyDiscretizedState

__all__ = ["angle_mesh"]

def get_abs_filter(bg="white") -> LinearSegmentedColormap:
    color = to_rgb(bg)

    return LinearSegmentedColormap.from_list(
        "abs_filter", [[color[0], color[1], color[2], 1], [0, 0, 0, 0]]
    )


def angle_mesh(
    X1: Array,
    X2: Array,
    y: SpatiallyDiscretizedState | Array,
    ax: Axes,
    *,
    p: ScalarLike = 1,
    abs_max: Optional[float] = None,
    cmap: LinearSegmentedColormap | str = "hsv",
    bg="white",
):
    """
    Plots colormesh of a spatial state or array y, where color corresponds to the phase,
    and the brightness corresponds to the magnitude.
    """

    if isinstance(y, SpatiallyDiscretizedState):
        y_vals = y.values
    else:
        y_vals = y

    pm1 = ax.pcolormesh(X1, X2, jnp.angle(y_vals).T, vmin=-jnp.pi, vmax=jnp.pi, cmap=cmap)
    pm2 = ax.pcolormesh(
        X1, X2, (jnp.abs(y_vals).T) ** p, vmin=0, vmax=abs_max, cmap=get_abs_filter(bg)
    )
    return pm1, pm2
