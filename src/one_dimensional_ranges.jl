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
MLJBase.transform(::SCALE, ::Val{:logminus}, x) = log(-x)
MLJBase.inverse_transform(::SCALE, ::Val{:logminus}, x) = -exp(x)
MLJBase.transform(::SCALE, ::Val{:log10}, x) = log10(x)
MLJBase.inverse_transform(::SCALE, ::Val{:log10}, x) = 10^x
MLJBase.transform(::SCALE, ::Val{:log2}, x) = log2(x)
MLJBase.inverse_transform(::SCALE, ::Val{:log2}, x) = 2^x

MLJBase.transform(::SCALE, f::Function, x) = x            # not a typo!
MLJBase.inverse_transform(::SCALE, f::Function, x) = f(x) # not a typo!

abstract type ParamRange <: MLJType end

Base.isempty(::ParamRange) = false

abstract type Boundedness end
abstract type Bounded <: Boundedness end
abstract type Unbounded <: Boundedness end 
abstract type LeftUnbounded <: Unbounded end
abstract type RightUnbounded <: Unbounded end
abstract type DoublyUnbounded <: Unbounded end

struct NumericRange{B<:Boundedness,T,D} <: ParamRange 
    field::Union{Symbol,Expr}
    lower::Union{T,Float64} # Float64 to allow for -Inf
    upper::Union{T,Float64} # Float64 to allow for Inf
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
function Base.range(model::Union{Model, Type},
                    field::Union{Symbol,Expr}; values=nothing,
                    lower=nothing, upper=nothing,
                    origin=nothing, unit=nothing, scale::D=nothing) where D 

    if model isa Model
        value = recursive_getproperty(model, field)
        T = typeof(value)
    else
        T = model
    end
    if T <: Real && values == nothing
        return numeric_range(T, D, field, lower, upper, origin, unit, scale)
    else
        return nominal_range(T, field, values)
    end
end

function numeric_range(T, D, field, lower, upper, origin, unit, scale)

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
        origin = (upper + lower)/2
    end
    if unit == nothing
        isunbounded && error("For an unbounded range you must "*
                             "specify `unit=...` to "*
                             "define a unit of scale.\nTo make the range "*
                             "bounded, specify finite `upper=...` "*
                             "and `lower=...`")
        unit = (upper - lower)/2
    end

    unit > 0 || error("`unit` must be positive. ")
    origin < upper && origin > lower ||
        error("`origin` must lie strictly between `lower` and `upper`. ")

    if lower == -Inf
        if upper == Inf
            B = DoublyUnbounded
            scale == nothing && (scale = :linear)
        else
            B = LeftUnbounded
            scale == nothing && (scale = :logminus)
        end
    else
        if upper == Inf
            B = RightUnbounded
            scale == nothing && (scale = :log)
        else
            B = Bounded
            scale == nothing && (scale = :linear)
        end
    end
    if scale isa Symbol
        NumericRange{B,T, Symbol}(field, lower, upper, origin, unit, scale)
    else
        NumericRange{B,T, D}(field, lower, upper, origin, unit, scale)
    end
end

nominal_range(T, field, values) =
    error("`values` does not have an appropriate type. ")
function nominal_range(::Type{T}, field, values::AbstractVector{T}) where T
    values === nothing && error("You must specify values=... "*
                                "for a nominal parameter. ")
    values
    return NominalRange{T}(field, Tuple(values))
end

"""
    MLJBase.scale(r::ParamRange)

Return the scale associated with a `ParamRange` object `r`. The
possible return values are: `:none` (for a `NominalRange`), `:linear`,
`:log`, `:log10`, `:log2`, or `:custom` (if `r.scale` is function).

"""
scale(r::NominalRange) = :none
scale(r::NumericRange) = :custom
scale(r::NumericRange{B,T,Symbol}) where {B<:Boundedness,T} =
    r.scale


## ITERATORS FROM A PARAMETER RANGE


"""
    MLJTuning.iterator(r::NominalRange)
    MLJTuning.iterator(r::NumericRange, n)

Return an iterator (currently a vector) for a `ParamRange` object `r`.
In the first case iteration is over all values. In the second case,
the iteration is over approximately `n` ordered values, generated as
follows:

First, exacltly `n` values are generated between `U` and `L`, with a
spacing determined by `r.scale`, where `U` and `L` are given by the
following table:

| `r.lower`   | `r.upper`  | `L`                 | `U`                 |
|-------------|------------|---------------------|---------------------|
| finite      | finite     | `r.lower`           | `r.upper`           |
| `-Inf`      | finite     | `r.upper - 2r.unit` | `r.upper`           |
| finite      | `Inf`      | `r.lower`           | `r.lower + 2r.unit  |
| `-Inf`      | `Inf`      | `r.origin - r.unit` | `r.origin + r.unit  |

If `r` isa a discrete range (`r isa
NumericRange{<:Any,<:Any,<:Integer}`) then the values are rounded,
with any duplicate values removed. Otherwise all the values are used
as is (and there are exacltly `n` of them).  """
iterator(r::NominalRange) = collect(r.values)

# top level

function iterator(r::NumericRange{<:Bounded,T},
                  n::Int) where {T<:Real}
    L = r.lower
    U = r.upper
    return iterator(T, L, U, r.scale, n)
end

function iterator(r::NumericRange{<:LeftUnbounded,T},
                  n::Int) where {T<:Real}
    L = r.upper - 2r.unit
    U = r.upper
    return iterator(T, L, U, r.scale, n)
end

function iterator(r::NumericRange{<:RightUnbounded,T},
                  n::Int) where {T<:Real}
    L = r.lower
    U = r.lower + 2r.unit
    return iterator(T, L, U, r.scale, n)
end

function iterator(r::NumericRange{<:DoublyUnbounded,T},
                  n::Int) where {T<:Real}
    L = r.origin - r.unit
    U = r.origin + r.unit
    return iterator(T, L, U, r.scale, n)
end

# middle level

iterator(::Type{<:Real}, L, U, s, n) =
    iterator(L, U, s, n)

function iterator(I::Type{<:Integer}, L, U, s, n)
    raw = iterator(L, U, s, n)
    rounded = map(x -> round(I, x), raw)
    return unique(rounded)
end

# base level

function iterator(L, U, s, n)
    transformed = range(transform(Scale, scale(s), L),
                stop=transform(Scale, scale(s), U),
                        length=n)
    inverse_transformed = map(transformed) do value
        inverse_transform(Scale, scale(s), value)
    end
    return inverse_transformed
end

