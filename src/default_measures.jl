# # DEFAULT MEASURES

"""
    default_measure(model)

Return a measure that should work with `model`, or return `nothing` if none can be
reliably inferred.

For Julia 1.9 and higher, `nothing` is returned, unless StatisticalMeasures.jl is
loaded.

# New implementations

This method dispatches `default_measure(model, observation_scitype)`, which has
`nothing` as the fallback return value. Extend `default_measure` by overloading this
version of the method. See for example the MLJBase.jl package extension,
DefaultMeausuresExt.jl.

"""
default_measure(m) = nothing
default_measure(m::Union{Supervised,Annotator}) =
    default_measure(m, nonmissingtype(guess_model_target_observation_scitype(m)))
default_measure(m, S) = nothing
