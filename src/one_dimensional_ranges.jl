## PARAMETER RANGES


#     Scale = SCALE()

# Object for dispatching on scales and functions when generating
# parameter ranges. We require different behaviour for scales and
# functions:

#      transform(Scale, scale(:log10), 100) = 2
#      inverse_transform(Scale, scale(:log10), 2) = 100

# but
#     transform(Scale, scale(log10), 100) = 100       # identity
#     inverse_transform(Scale, scale(log10), 100) = 2


struct SCALE end
Scale = SCALE()
scale(s::Symbol) = Val(s)
scale(f::Function) = f
MLJBase.transform(::SCALE, ::Val{:linear}, x) = x
MLJBase.inverse_transform(::SCALE, ::Val{:linear}, x) = x
MLJBase.transform(::SCALE, ::Val{:log}, x) = log(x)
MLJBase.inverse_transform(::SCALE, ::Val{:log}, x) = exp(x)
MLJBase.transform(::SCALE, ::Val{:log10}, x) = log10(x)
MLJBase.inverse_transform(::SCALE, ::Val{:log10}, x) = 10^x
MLJBase.transform(::SCALE, ::Val{:log2}, x) = log2(x)
MLJBase.inverse_transform(::SCALE, ::Val{:log2}, x) = 2^x
MLJBase.transform(::SCALE, f::Function, x) = x            # not a typo!
MLJBase.inverse_transform(::SCALE, f::Function, x) = f(x) # not a typo!

abstract type ParamRange <: MLJType end

Base.isempty(::ParamRange) = false

struct NominalRange{T} <: ParamRange
    field::Union{Symbol,Expr}
    values::Tuple{Vararg{T}}
end

struct NumericRange{T,D} <: ParamRange
    field::Union{Symbol,Expr}
    lower::Union{T,Float64} # Float64 to allow for -Inf
    upper::Union{T,Float64} # Float64 to allow for Inf
    origin::T
    unit::T
    scale::D
end

MLJBase.show_as_constructed(::Type{<:ParamRange}) = true

"""
    r = range(model, :hyper; values=nothing)

Defines a `NominalRange` object for a field `hyper` of `model`,
assuming the field is a not a subtype of `Real`. Note that `r` is not
directly iterable but `iterator(r)` iterates over `values`.

A nested hyperparameter is specified using dot notation. For example,
`:(atom.max_depth)` specifies the `:max_depth` hyperparameter of the hyperparameter `:atom` of `model`.

    r = range(model, :hyper; upper=nothing, lower=nothing, scale=:linear)

Defines a `NumericRange` object for a `Real` field `hyper` of `model`.
Note that `r` is not directly iteratable but `iterator(r, n)` iterates
over `n` values between `lower` and `upper` values, according to the
specified `scale`. The supported scales are `:linear, :log, :log10,
:log2`. Values for `Integer` types are rounded (with duplicate values
removed, resulting in possibly less than `n` values).

Alternatively, if a function `f` is provided as `scale`, then
`iterator(r, n)` iterates over the values `[f(x1), f(x2), ... ,
f(xn)]`, where `x1, x2, ..., xn` are linearly spaced between `lower`
and `upper`.


"""
function Base.range(model, field::Union{Symbol,Expr}; values=nothing,
                    lower=nothing, upper=nothing,
                    origin=nothing, unit=nothing, scale::D=:linear) where D
    value = recursive_getproperty(model, field)
    T = typeof(value)
    if T <: Real
        if lower === nothing && upper === nothing
            error("You must specify `lower=...` or "*
                  "`upper=...` or both, for a `Real` parameter. ")
        end
        lower === Inf && error("`lower` must be finite or `-Inf`. ")
        upper === -Inf && error("`upper` must be finite or `Inf`")
        lower === nothing && (lower = -Inf)
        upper === nothing && (upper = Inf)
        lower < upper || errof("`lower` must be strictly less than `upper`. ")
        isunbounded = (lower === -Inf || upper === Inf)
        if origin == nothing
            isunbounded && error("For an unbounded range you must "*
                                 "specify `origin=...` to "*
                                 "define a centre.\nTo make the range "*
                                 "bounded, specify finite `upper=...` "*
                                 "and `lower=...`")
            origin = _round(T, (upper + lower)/2)
        end
        if unit == nothing
            isunbounded && error("For an unbounded range you must "*
                                 "specify `unit=...` to "*
                                 "define a unit of scale.\nTo make the range "*
                                 "bounded, specify finite `upper=...` "*
                                 "and `lower=...`")
            unit = _floor(T, (upper - lower)/2)
        end

        unit > 0 || error("`unit` must be positive. ")
        origin < upper && origin > lower ||
            error("`origin` must lie strictly between `lower` and `upper`." )


        values !== nothing && @warn "`values` in the case of a "*
        "numeric range are ignored. "

        return NumericRange{T,D}(field, lower, upper, origin, unit, scale)
    else
        values === nothing && error("You must specify values=... "*
                                    "for a nominal parameter. ")
        return NominalRange{T}(field, Tuple(values))
    end
end

# to only round and floor for integers:
_floor(::Type, x) = x
_floor(T::Type{<:Integer}, x) = convert(T, floor(x))
_round(T::Type, x) = x
_round(T::Type{<:Integer}, x) = convert(T, round(x))


"""
    MLJTuning.scale(r::ParamRange)

Return the scale associated with the `ParamRange` object `r`. The
possible return values are: `:none` (for a `NominalRange`), `:linear`,
`:log`, `:log10`, `:log2`, or `:custom` (if `r.scale` is function).

"""
scale(r::NominalRange) = :none
scale(r::NumericRange) = :custom
scale(r::NumericRange{T,Symbol}) where T =
    r.scale


## ITERATORS FROM A PARAMETER RANGE


"""
    MLJTuning.iterator(r::NominalRange)
    MLJTuning.iterator(r::NumericRange, n)

Return an iterator (currently a vector) for a `ParamRange` object `r`.
In the first case iteration is over all values. In the second case,
the iteration is over `N` ordered values in the range, as specified by
the `lower`, `upper` and `scale` attributes of the range. For
`AbstractFloat` ranges `N=n`. For `Integer` ranges `N` could be less
than `n`, because of rounding.

Iterating over unbounded ranges (`lower=-Inf` or `upper=Inf`) is not
supported.

 """

iterator(param_range::NominalRange) = collect(param_range.values)

function iterator(param_range::NumericRange{T}, n::Int) where {T<:Real}
    s = scale(param_range.scale)
    transformed = range(transform(Scale, s, param_range.lower),
                stop=transform(Scale, s, param_range.upper),
                length=n)
    inverse_transformed = map(transformed) do value
        inverse_transform(Scale, s, value)
    end
    return unique(inverse_transformed)
end

# in special case of integers, round to nearest integer:
function iterator(param_range::NumericRange{I}, n::Int) where {I<:Integer}
    s = scale(param_range.scale)
    transformed = range(transform(Scale, s, param_range.lower),
                stop=transform(Scale, s, param_range.upper),
                length=n)
    inverse_transformed =  map(transformed) do value
        round(I, inverse_transform(Scale, s, value))
    end
    return unique(inverse_transformed)
end
