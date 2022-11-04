const ERR_PIPELINE = ErrorException(
    "The `@pipeline` macro is deprecated. For pipelines without "*
    "target transformations use pipe syntax, as in "*
    "`ContinuousEncoder() |> Standardizer() |> my_classifier`. "*
    "For details and advanced optioins, query the `Pipeline` docstring. "*
    "To wrap a supervised model in a target transformation, use "*
    "`TransformedTargetModel`, as in "*
    "`TransformedTargetModel(my_regressor, target=Standardizer())`"
)

macro pipeline(ex...) throw(ERR_PIPELINE) end
