nlevels(c::CategoricalValue) = length(levels(c.pool))
nlevels(c::CategoricalString) = length(levels(c.pool))

abstract type Found end
    struct Continuous <: Found end 
    abstract type Discrete <: Found end
        struct Multiclass{N} <: Discrete end
        abstract type OrderedFactor <: Discrete end
        struct FiniteOrderedFactor{N} <: OrderedFactor end
        struct Count <: OrderedFactor end
    struct Other <: Found end

# aliases:
const Binary = Multiclass{2}

# trait function:
scitype(x) = scitype(x, Val(Tables.istable(x)))

# universal fallback:
scitype(x::Any, ::Any) = Other

# scalars:
scitype(::Missing, ::Val{false}) = Missing
scitype(::Real, ::Val{false}) = Continuous
scitype(::Integer, ::Val{false}) = Count
scitype(c::CategoricalValue, ::Val{false}) =
    c.pool.ordered ? FiniteOrderedFactor{nlevels(c)} : Multiclass{nlevels(c)}
scitype(c::CategoricalString, ::Val{false}) =
    c.pool.ordered ? FiniteOrderedFactor{nlevels(c)} : Multiclass{nlevels(c)}


# arrays:
function scitype(A::AbstractArray, ::Val{false})
    ret = Union{}
    for j in eachindex(A)
        ret = Union{ret, scitype(A[j])}
    end
    return ret
end

# tables:
# TODO: slow. fiddly to fix because schema(table).eltype does
# not determine scitype.
function scitype(table, ::Val{true})
    ret = Union{}
    cols = Tables.columns(table)
    names = Tables.schema(table).names
    for name in names
        for x in getproperty(cols, name)
            ret = Union{ret,scitype(x)}
        end
    end
    return ret
end





