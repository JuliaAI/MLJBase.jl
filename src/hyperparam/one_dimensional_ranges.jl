## PARAMETER RANGES

abstract type ParamRange{T} <: MLJType end

Base.isempty(::ParamRange) = false

abstract type Boundedness end

abstract type   Bounded <: Boundedness end
abstract type Unbounded <: Boundedness end

abstract type   LeftUnbounded <: Unbounded end
abstract type  RightUnbounded <: Unbounded end
abstract type DoublyUnbounded <: Unbounded end

struct NumericRange{T,B<:Boundedness,D} <: ParamRange{T}
    field::Union{Symbol,Expr}
    lower::Union{T,Float64}     # Float64 to allow for -Inf
    upper::Union{T,Float64}     # Float64 to allow for Inf
    origin::Float64
    unit::Float64
    scale::D
end

struct NominalRange{T,N} <: ParamRange{T}
    field::Union{Symbol,Expr}
    values::NTuple{N,T}
end

function Base.show(stream::IO,
                   ::MIME"text/plain",
                   r::ParamRange{T}) where T
    if r.field isa Expr
        fstr = ":($(r.field))"
    else
        fstr = ":$(r.field)"
    end
    repr = "$(typeof(r).name)($T, $fstr, ... )"
    print(stream, repr)
    return nothing
end

"""
    r = range(model, :hyper; values=nothing)

Define a one-dimensional `NominalRange` object for a field `hyper` of
`model`. Note that `r` is not directly iterable but `iterator(r)`
is. 

The behaviour of range methods depends on the type of the value of the
hyperparameter at `model` during range construction. To override (or
if `model` is not available) specify a type in place of `model`.

A nested hyperparameter is specified using dot notation. For example,
`:(atom.max_depth)` specifies the `max_depth` hyperparameter of
the submodel `model.atom`.

    r = range(model, :hyper; upper=nothing, lower=nothing,
              scale=nothing, values=nothing)

Assuming `values` is not specified, define a one-dimensional
`NumericRange` object for a `Real` field `hyper` of `model`.  Note
that `r` is not directly iteratable but `iterator(r, n)`is an iterator
of length `n`. To generate random elements from `r`, instead apply
`rand` methods to `sampler(r)`. The supported scales are `:linear`,`
:log`, `:logminus`, `:log10`, `:log2`, or a callable object.

A nested hyperparameter is specified using dot notation (see above).

If `scale` is unspecified, it is set to `:linear`, `:log`,
`:logminus`, or `:linear`, according to whether the interval `(lower,
upper)` is bounded, right-unbounded, left-unbounded, or doubly
unbounded, respectively.  Note `upper=Inf` and `lower=-Inf` are
allowed.

If `values` is specified, the other keyword arguments are ignored and
a `NominalRange` object is returned (see above).

See also: [`iterator`](@ref), [`sampler`](@ref)

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
    #typeof(lower) <: Union{Float64, T} || throw(ArgumentError("`lower` must either be `-Inf`"
    #        * "or a finite value of type $(T)."  ))
    #typeof(upper) <: Union{Float64, T} || throw(ArgumentError("`upper` must either be `Inf`"
    #        * "or a finite value of type $(T)."  ))
    scale isa Symbol && (D = Symbol)
    return NumericRange{T,B,D}(field, lower, upper, origin, unit, scale)
end

nominal_range(T, field, values) = throw(ArgumentError(
   "`$values` must be an instance of type `AbstractVector{<:$T}`."
    * (T <: Model ? "\n Perharps you forgot to instantiate model"
     * "as `$(T)()`" : "") ))

nominal_range(T, field, ::Nothing) = throw(ArgumentError(
    "You must specify values=... for a nominal parameter."  ))

function nominal_range(::Type{T}, field, values::AbstractVector{T}) where T
    return NominalRange{T,length(values)}(field, Tuple(values))
end

#specific def for T<:AbstractFloat
function nominal_range(::Type{T}, field, 
               values::AbstractVector{<:Real}) where T<: AbstractFloat
    return NominalRange{T,length(values)}(field, Tuple(values))
end
