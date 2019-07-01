istransparent(::Any) = false
istransparent(::MLJType) = true

"""
    params(m)

Recursively convert any transparent object `m` into a named tuple,
keyed on the fields of `m`. A model is *transparent* if
`MLJBase.istransparent(m) == true`. The named tuple is possibly nested
because `params` is recursively applied to the field values, which
themselves might be transparent.

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
    fields = fieldnames(typeof(m))
    NamedTuple{fields}(Tuple([params(getfield(m, field)) for field in fields]))
end

# applying the following to `:(a.b.c)` returns `(:(a.b), :c)`
function reduce_nested_field(ex)
    ex.head == :. || throw(ArgumentError)
    tail = ex.args[2]
    tail isa QuoteNode || throw(ArgumentError)
    field = tail.value
    field isa Symbol || throw(ArgmentError)
    subex = ex.args[1]
    return (subex, field)
end


function Base.getproperty(obj, ex::Expr)
    subex, field = reduce_nested_field(ex)
    return getproperty(getproperty(obj, subex), field)
end

function Base.setproperty!(obj, ex::Expr, value)
    subex, field = reduce_nested_field(ex)
    last_obj = getproperty(obj, subex)
    return setproperty!(last_obj, field, value)
end



