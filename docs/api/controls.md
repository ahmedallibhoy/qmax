# Controls

Controls provide way to add time-dependence to operators in qmax. There are two ways that 
controls can be used.

1. Specifying a time-varying operator $H(t)$ (see [Timevarying Operator](operators/timevarying.md))
2. Specifiying inputs to a controlled operator $H(u) = H_0 + \sum_{j=1}^{n}u_jH_j$ (see [Controlled Operator](operators/controlled.md))

The latter is useful for optimal control applications, where one might want 
to optimize a cost function with respect to a cost function computed by [qmax.Propagator.propagate][]. 

For optimal control applications, qmax provides interpolated control objects 
which define a control input interpolated between values at evenly spaced grid points. 
Because these control objects are [equinox Modules](https://docs.kidger.site/equinox/api/module/module/),
they are automatically PyTrees, and thus can be differentiated directly.

::: qmax.control.AbstractControl 
    options:
        members:
            - __call__

::: qmax.control.ControlFunction 
    options:
        members:
            -

::: qmax.control.PiecewiseConstantControl 
    options:
        members:
            - from_function

::: qmax.control.PiecewiseLinearControl 
    options:
        members:
            - from_function