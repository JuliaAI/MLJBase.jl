nlevels(c::CategoricalValue) = length(levels(c.pool))
nlevels(c::CategoricalString) = length(levels(c.pool))

abstract type Found end
    abstract type Known <: Found end
        struct Continuous <: Known end 
        abstract type Discrete <: Known end
            struct Multiclass{N} <: Discrete end
            abstract type OrderedFactor <: Discrete end
                struct FiniteOrderedFactor{N} <: OrderedFactor end
                struct Count <: OrderedFactor end
    struct Unknown <: Found end 

# aliases:
const Other = Unknown # TODO: depreciate:
const Binary = Multiclass{2}

"""
    scitype(x)

Return the scientific type for scalar values that object `x` can
represent.

""" 
scitype(::Any) = Unknown     
scitype(::Missing) = Missing
scitype(::Real) = Continuous
scitype(::Integer) = Count
scitype(c::CategoricalValue) =
    c.pool.ordered ? FiniteOrderedFactor{nlevels(c)} : Multiclass{nlevels(c)}
scitype(c::CategoricalString) = 
    c.pool.ordered ? FiniteOrderedFactor{nlevels(c)} : Multiclass{nlevels(c)}

"""
    union_scitypes(X)

Return the union over all elements `x` of `X` of `scitype(x)`. Here
`X` can be any table, sparse table, or abstract arrray.

"""
function union_scitypes(X)
    container_type(X) in [:table, :sparse] || throw(ArgumentError)
    ret = Union{}
    features = schema(X).names
    for ftr in features
        col = selectcols(X, ftr)
        ret = Union{ret,union_scitypes(col)}
    end
    return ret
end

# arrays:
function union_scitypes(A::AbstractArray)
    ret = Union{}
    for j in eachindex(A)
        ret = Union{ret, scitype(A[j])}
    end
    return ret
end

"""
    column_scitypes_as_tuple_type(X)

Returns `Tuple{T1, T2, ..., Tn}` where `Tj` is the union of scitypes
of elements in the `jth` column of `X`. Here `X` is any table, sparse
table, or abstract matrix.

"""
function column_scitypes_as_tuple(X)
    container_type(X) in [:table, :sparse] || throw(ArgumentError)
    return Tuple{[union_scitypes(selectcols(X, c)) for c in schema(X).names]...}
end

function column_scitypes_as_tuple(X::AbstractMatrix)
    return Tuple{[union_scitypes(X[:,c]) for c in 1:size(X, 2)]...}
end
