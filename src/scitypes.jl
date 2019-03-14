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

# universal fallback:
scitype(::Any) = Other
scitype(::Missing) = Missing
scitype(::Real) = Continuous
scitype(::Integer) = Count
scitype(c::CategoricalValue) =
    c.pool.ordered ? FiniteOrderedFactor{nlevels(c)} : Multiclass{nlevels(c)}
scitype(c::CategoricalString) = 
    c.pool.ordered ? FiniteOrderedFactor{nlevels(c)} : Multiclass{nlevels(c)}

# arrays:
function union_scitypes(A::AbstractArray)
    ret = Union{}
    for j in eachindex(A)
        ret = Union{ret, scitype(A[j])}
    end
    return ret
end

# tables and sparse tables:
function union_scitypes(table)
    ret = Union{}
    features = schema(table).names
    for ftr in features
        col = selectcols(table, ftr)
        ret = Union{ret,union_scitypes(col)}
    end
    return ret
end





