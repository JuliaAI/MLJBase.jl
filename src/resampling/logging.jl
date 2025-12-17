const DOC_DEFAULT_LOGGER =
"""

The default logger is used in calls to [`evaluate!`](@ref) and [`evaluate`](@ref), and
in the constructors `TunedModel` and `IteratedModel`, unless the `logger` keyword is
explicitly specified.

!!! note

    Prior to MLJ v0.20.7 (and MLJBase 1.5) the default logger was always `nothing`.

"""

"""
    default_logger()

Return the current value of the default logger for use with supported machine learning
tracking platforms, such as [MLflow](https://mlflow.org/docs/latest/index.html).

$DOC_DEFAULT_LOGGER

When MLJBase is first loaded, the default logger is `nothing`.

"""
default_logger() = DEFAULT_LOGGER[]

"""
    default_logger(logger)

Reset the default logger.

# Example

Suppose an [MLflow](https://mlflow.org/docs/latest/index.html) tracking service is running
on a local server at `http://127.0.0.1:500`. Then in every `evaluate` call in which
`logger` is not specified, the peformance evaluation is
automatically logged to the service, as here:

```julia
using MLJ
logger = MLJFlow.Logger("http://127.0.0.1:5000/api")
default_logger(logger)

X, y = make_moons()
model = ConstantClassifier()
evaluate(model, X, y, measures=[log_loss, accuracy)])
```

"""
function default_logger(logger)
    DEFAULT_LOGGER[] = logger
end
