## PARAMETER RANGES

abstract type ParamRange <: MLJType end

Base.isempty(::ParamRange) = false

abstract type Boundedness end

abstract type   Bounded <: Boundedness end
abstract type Unbounded <: Boundedness end

abstract type   LeftUnbounded <: Unbounded end
abstract type  RightUnbounded <: Unbounded end
abstract type DoublyUnbounded <: Unbounded end

struct NumericRange{B<:Boundedness,T,D} <: ParamRange
    field::Union{Symbol,Expr}
    lower::Union{T,Float64}     # Float64 to allow for -Inf
    upper::Union{T,Float64}     # Float64 to allow for Inf
    origin::Float64
    unit::Float64
    scale::D
end

struct NominalRange{T} <: ParamRange
    field::Union{Symbol,Expr}
    values::Tuple{Vararg{T}}
end

MLJBase.show_as_constructed(::Type{<:ParamRange}) = true

"""
    r = range(model, :hyper; values=nothing)

Defines a `NominalRange` object for a field `hyper` of `model`,
assuming the field value does not subtype `Real`. Note that `r` is not
directly iterable but `iterator(r)` iterates over `values`.

The specific type of the hyperparameter is automatically determined
from the current value at `model`. To override, specify a type in
place of `model`.

A nested hyperparameter is specified using dot notation. For example,
`:(atom.max_depth)` specifies the `:max_depth` hyperparameter of the
hyperparameter `:atom` of `model`.

    r = range(model, :hyper; upper=nothing, lower=nothing,
              scale=nothing, values=nothing)

Assuming `values == nothing`, this defines a `NumericRange` object for
a `Real` field `hyper` of `model`.  Note that `r` is not directly
iteratable but `iterator(r, n)` iterates over `n` values controlled by
the various parameters (see more at [iterator](@ref).  The supported
scales are `:linear`,` :log`, `:logminus`, `:log10`, `:log2`, or a
function (see below).  Values for `Integer` types are rounded (with
duplicate values removed, resulting in possibly less than `n` values).

If `scale` is unspecified, it is set to `:linear`, `:log`,
`:logminus`, or `:linear`, according to whether the interval `(lower,
upper)` is bounded, right-unbounded, left-unbounded, or doubly
unbounded, respectively.  Note `upper=Inf` and `lower=-Inf` are
allowed.

If `values` is specified, the other keyword arguments are ignored and
a `NominalRange` object is returned (see above).

To override the automatically detected hyperparameter type, substitute
a type in place of `model`.

If a function `f` is provided as `scale`, then
`iterator(r, n)` iterates over the values `[f(x1), f(x2), ... ,
f(xn)]`, where `x1, x2, ..., xn` are linearly spaced between `lower`
and `upper`.
"""
function Base.range(model::Union{Model, Type}, field::Union{Symbol,Expr};
                    values=nothing, lower=nothing, upper=nothing,
                    origin=nothing, unit=nothing, scale::D=nothing) where D
    if model isa Model
        value = recursive_getproperty(model, field)
        T = typeof(value)
    else
        T = model
    end
    if T <: Real && values === nothing
        return numeric_range(T, D, field, lower, upper, origin, unit, scale)
    else
        return nominal_range(T, field, values)
    end
end

function numeric_range(T, D, field, lower, upper, origin, unit, scale)
    lower === Inf &&
        throw(ArgumentError("`lower` must be finite or `-Inf`."))
    upper === -Inf &&
        throw(ArgumentError("`upper` must be finite or `Inf`."))

    lower === nothing && (lower = -Inf)
    upper === nothing && (upper = Inf)

    lower < upper ||
        throw(ArgumentError("`lower` must be strictly less than `upper`."))

    is_unbounded = (lower === -Inf || upper === Inf)

    if origin === nothing
        is_unbounded &&
            throw(DomainError("For an unbounded range you must specify " *
                              "`origin=...` to define a centre.\nTo make " *
                              "the range bounded, specify finite " *
                              "`upper=...` and `lower=...`."))
        origin = (upper + lower)/2
    end
    if unit === nothing
        is_unbounded &&
            throw(DomainError("For an unbounded range you must specify " *
                              "`unit=...` to define a unit of scale.\nTo " *
                              "make the range bounded, specify finite " *
                              "`upper=...` and `lower=...`."))
        unit = (upper - lower)/2
    end
    unit > 0 || throw(DomainError("`unit` must be positive."))
    origin < upper && origin > lower ||
        throw(DomainError("`origin` must lie strictly between `lower` and " *
                          "`upper`."))
    if lower === -Inf
        if upper === Inf
            B = DoublyUnbounded
            scale === nothing && (scale = :linear)
        else
            B = LeftUnbounded
            scale === nothing && (scale = :logminus)
        end
    else
        if upper === Inf
            B = RightUnbounded
            scale === nothing && (scale = :log)
        else
            B = Bounded
            scale === nothing && (scale = :linear)
        end
    end
    scale isa Symbol && (D = Symbol)
    return NumericRange{B,T,D}(field, lower, upper, origin, unit, scale)
end

nominal_range(T, field, values) = throw(ArgumentError(
    "`values` does not have an appropriate type."))

nominal_range(T, field, ::Nothing) = throw(ArgumentError(
    "You must specify values=... for a nominal parameter."))

function nominal_range(::Type{T}, field, values::AbstractVector{T}) where T
    return NominalRange{T}(field, Tuple(values))
end
