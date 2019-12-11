# The transformer below, not intended for the general user, wraps a
# single data-manipulating function without parameters, and is used by
# the @pipeline macro.  Generally, the only time the user will need to
# implement such a tranformer is when it also depends on parameters,
# and the recommendation is to instead subtype `Static` with a new
# custom type. In the context of a learning network, the user can use
# the `node` function to overload data-manipulating operations for
# nodes (if they are not already overloaded out-of-the-box).

const STATIC_TRANSFORMER_DESCR = "Applies a given data transformation
`f` (either a function or callable)."

"""
$STATIC_TRANSFORMER_DESCR

## Field

* `f=identity`: function or callable object to use for the data transformation.
"""
mutable struct StaticTransformer <: MLJBase.Unsupervised
    f
end
StaticTransformer(;f=identity) = StaticTransformer(f)

MLJBase.fitted_params(::StaticTransformer) = NamedTuple()
MLJBase.fit(::StaticTransformer, ::Integer, _) = nothing, nothing, NamedTuple()
MLJBase.transform(model::StaticTransformer, fitresult, Xnew) = (model.f)(Xnew)

metadata_model(StaticTransformer,
               input=MLJBase.Table(Scientific),
               output=MLJBase.Table(Scientific),
               weights=false,
               descr=STATIC_TRANSFORMER_DESCR,
               path="MLJBase.StaticTransformer")

metadata_pkg(StaticTransformer,
             name="MLJBase",
             julia=true,
             license="MIT")

