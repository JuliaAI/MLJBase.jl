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
mutable struct WrappedFunction <: MLJBase.Static
    f
end
WrappedFunction(;f=identity) = WrappedFunction(f)

fitted_params(::WrappedFunction) = NamedTuple()
fit(::WrappedFunction, ::Integer, _) = nothing, nothing, NamedTuple()
transform(model::WrappedFunction, fitresult, Xnew) = (model.f)(Xnew)

MMI.is_wrapper(::Type{<:WrappedFunction}) = true
MMI.is_pure_julia(::Type{<:WrappedFunction}) = true
MMI.load_path(::Type{<:WrappedFunction}) = "MLJBase.WrappedFunction"
MMI.package_name(::Type{<:WrappedFunction}) = "MLJBase"
