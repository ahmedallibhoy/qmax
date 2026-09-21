# Hilbert Spaces

## Base classes for spaces

::: qmax.hilbert_space.AbstractHilbertSpace
    options:
        members:
            - dim
            - from_coeffs
            - zeros
            - random
            - zeros_like
            - stack
            - concatenate
            - identity
            - zero_operator

::: qmax.spaces.spatial_discretization.SpatialDiscretization
    options:
        members:
            - spatial_dim
            - from_values
            - from_function
            - x_ranges
            - x_range
            - points
            - eval
            - laplacian
            - potential_energy
            - position
            - momentum

## Base classes for elements

::: qmax.hilbert_space.AbstractState
    options:
        members:
            - innerp
            - norm 
            - norm2
            - expected_value
            - shape
            - rank
            - __getitem__

::: qmax.spaces.spatial_discretization.SpatiallyDiscretizedState
    options:
        members:
            - values