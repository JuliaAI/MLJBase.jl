istransparent(::Any) = false
istransparent(::MLJType) = true

"""
    params(m::MLJType)

Recursively convert any transparent object `m` into a named tuple,
keyed on the property names of `m`. An object is *transparent* if
`MLJBase.istransparent(m) == true`. The named tuple is possibly nested
because `params` is recursively applied to the property values, which
themselves might be transparent.

For most `MLJType` objects, properties are synonymous with fields, but
this is not a hard requirement.

Most objects of type `MLJType` are transparent.

    julia> params(EnsembleModel(atom=ConstantClassifier()))
    (atom = (target_type = Bool,),
     weights = Float64[],
     bagging_fraction = 0.8,
     rng_seed = 0,
     n = 100,
     parallel = true,)

"""
params(m) = params(m, Val(istransparent(m)))
params(m, ::Val{false}) = m
function params(m, ::Val{true})
    fields = propertynames(m)
    NamedTuple{fields}(Tuple([params(getproperty(m, field))
                              for field in fields]))
end
