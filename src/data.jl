CategoricalElement{U} = Union{CategoricalValue{<:Any,U},CategoricalString{U}}

"""
    classes(x)

All the categorical elements with the same pool as `x` (including `x`),
returned as a list, with an ordering consistent with the pool. Here
`x` has `CategoricalValue` or `CategoricalString` type, and
`classes(x)` is a vector of the same eltype.

Not to be confused with the levels of `x.pool` which have a
different type. In particular, while `x in classes(x)` is always
true, `x in x.pool.levels` is not true.

    julia> v = categorical([:c, :b, :c, :a])
    julia> levels(v)
    3-element Array{Symbol,1}:
     :a
     :b
     :c
    julia> classes(v[4])
    3-element Array{CategoricalValue{Symbol,UInt32},1}:
     :a
     :b
     :c

"""
function classes(x::CategoricalElement)
    p = x.pool
    return [p.valindex[p.invindex[v]] for v in p.levels]
end

# a method just for testing:
raw(x::CategoricalElement) = x.pool.index[x.level]

"""
   int(x)

The positional integer of the `CategoricalString` or `CategoricalValue`
`x`, in the ordering defined by the pool of `x`. The type of `int(x)`
is the refrence type of `x`.

Not to be confused with `x.ref`, which is unchanged by reordering of
the pool of `x`, but has the same type.

    int(X::CategoricalArray)
    int(W::Array{<:CategoricalString})
    int(W::Array{<:CategoricalValue})

Broadcasted versions of `int`.

    julia> v = categorical([:c, :b, :c, :a])
    julia> levels(v)
    3-element Array{Symbol,1}:
     :a
     :b
     :c
    julia> int(v)
    4-element Array{UInt32,1}:
     0x00000003
     0x00000002
     0x00000003
     0x00000001

See also: [`decoder`](@ref).
"""
int(x::CategoricalElement) = x.pool.order[x.pool.invindex[x]]
int(A::AbstractArray{<:CategoricalElement}) = broadcast(int, A)
# workaround for CategoricalArrays issue
# https://github.com/JuliaData/CategoricalArrays.jl/issues/199:
# function int(X::CategoricalArray)
    

struct CategoricalDecoder{T,R} # <: MLJType
    pool::CategoricalPool{T,R}
    invorder::Vector{Int}
end

"""
    d = decoder(x)

A callable object for decoding the integer representation of a
`CategoricalString` or `CategoricalValue` sharing the same pool as
`x`. (Here `x` is of one of these two types.) Specifically, one has
`d(int(y)) == y` for all `y in classes(x)`. One can also call `d` on
integer arrays, in which case `d` is broadcast over all elements.

    julia> v = categorical([:c, :b, :c, :a])
    julia> int(v)
    4-element Array{UInt32,1}:
     0x00000003
     0x00000002
     0x00000003
     0x00000001
    julia> d = decoder(v[3])
    julia> d(int(v)) == v
    true

See also: [`int`](@ref), [`classes`](@ref).

"""
decoder(element::CategoricalElement) =
    CategoricalDecoder(element.pool, sortperm(element.pool.order))
## in the next lot need to skip the missing one
# decoder(X::CategoricalArray) = CategoricalDecoder(X.pool)
# function decoder(V::Array{<:CategoricalElement})
#     isempty(V) && error("Unable to extract decoder from empty array. ")
#     return X[1]
# end

(decoder::CategoricalDecoder{T,R})(i::Integer) where {T,R} =
    CategoricalValue{T,R}(decoder.invorder[i], decoder.pool)
(decoder::CategoricalDecoder{String,R})(i::Integer) where R =
    CategoricalString{R}(decoder.invorder[i], decoder.pool)
(decoder::CategoricalDecoder)(I::AbstractArray{<:Integer}) = broadcast(decoder, I)


## TABULAR DATA

# hack for detecting JuliaDB.NDSparse tables without loading as dependency:
# isndsparse(X) = isdefined(X, :data_buffer)


## UTILITY FOR CONVERTING BETWEEN TABULAR DATA AND MATRICES

"""
    MLJBase.matrix(X)

Convert a table source `X` into an `Matrix`; or, if `X` is
a `AbstractMatrix`, return `X`. Optimized for column-based sources.

If instead X is a sparse table, then a `SparseMatrixCSC` object is
returned. The integer relabelling of column names follows the
lexicographic ordering (as indicated by `schema(X).names`).

"""
matrix(X) = matrix(Val(ScientificTypes.trait(X)), X)
matrix(::Val{:other}, X) = throw(ArgumentError)
matrix(::Val{:other}, X::AbstractMatrix) = X

function matrix(::Val{:table}, X)
    cols = Tables.columns(X) # property-accessible object
    mat = reduce(hcat, [getproperty(cols, ftr) for ftr in propertynames(cols)])
    # tightest eltype:
    return broadcast(identity, mat)
end

function matrix(::Val{:sparse}, X)
    K = keys(X)
    features = schema(X).names
    index_given_feature = Dict{Symbol,Int}()
    for j in eachindex(features)
        index_given_feature[features[j]] = j
    end
    I = [k[1] for k in K]
    J = [index_given_feature[k[2]] for k in K]
    V = [v[1] for v in values(X)]
    return sparse(I, J, V)
end

"""
    MLJBase.table(cols; prototype=cols)

Convert a named tuple of vectors `cols`, into a table. The table
type returned is the "preferred sink type" for `prototype` (see the
Tables.jl documentation).

    MLJBase.table(X::AbstractMatrix; names=nothing, prototype=nothing)

Convert an abstract matrix `X` into a table with `names` (a tuple of
symbols) as column names, or with labels `(:x1, :x2, ..., :xn)` where
`n=size(X, 2)`, if `names` is not specified.  If prototype=nothing,
then a named tuple of vectors is returned.

Equivalent to `table(cols, prototype=prototype)` where `cols` is the
named tuple of columns of `X`, with `keys(cols) = names`.

"""
function table(cols::NamedTuple; prototype=cols)
    Tables.istable(prototype) || error("prototype is not tabular.")
    return Tables.materializer(prototype)(cols)
end
function table(X::AbstractMatrix; names=nothing, prototype=nothing)
    if names == nothing
        _names = tuple([Symbol(:x, j) for j in 1:size(X, 2)]...)
    else
        _names = names
    end
    cols = NamedTuple{_names}(tuple([X[:,j] for j in 1:size(X, 2)]...))
    _prototype = (prototype == nothing ? cols : prototype)
    return table(cols; prototype=_prototype)
end


## UNIFIED API FOR ACCESSING TABLES, MATRICES AND VECTORS

"""
    selectrows(X, r)

Select single or multiple rows from any table, sparse table, or
abstract vector `X`.  If `X` is tabular, the object returned is a
table of the preferred sink type of `typeof(X)`, even a single row is
selected.

"""
selectrows(X, r) = selectrows(Val(ScientificTypes.trait(X)), X, r)
selectrows(::Val{:other}, X, r) = throw(ArgumentError)

"""
    selectcols(X, c)

Select single or multiple columns from any table or sparse table
`X`. If `c` is an abstract vector of integers or symbols, then the
object returned is a table of the preferred sink type of
`typeof(X)`. If `c` is a *single* integer or column, then a `Vector`
or `CategoricalVector` is returned.

"""
selectcols(X, c) = selectcols(Val(ScientificTypes.trait(X)), X, c)
selectcols(::Val{:other}, X, c) = throw(ArgumentError)

"""
    select(X, r, c)

Select element of a table or sparse table at row `r` and column
`c`. In the case of sparse data where the key `(r, c)`, zero or
`missing` is returned, depending on the value type.

See also: [`selectrows`](@ref), [`selectcols`](@ref).

"""
select(X, r, c) = select(Val(ScientificTypes.trait(X)), X, r, c)
select(::Val{:other}, X, r, c) = throw(ArgumentError)

"""
    nrows(X)

Return the number of rows in a table, sparse table, or abstract vector.

"""
nrows(X) = nrows(Val(ScientificTypes.trait(X)), X)
nrows(::Val{:other}, X) = throw(ArgumentError)


# project named tuple onto a tuple with only specified `labels` or indices:
project(t::NamedTuple, labels::AbstractArray{Symbol}) = NamedTuple{tuple(labels...)}(t)
project(t::NamedTuple, label::Colon) = t
project(t::NamedTuple, label::Symbol) = project(t, [label,])
project(t::NamedTuple, indices::AbstractArray{<:Integer}) =
    NamedTuple{tuple(keys(t)[indices]...)}(tuple([t[i] for i in indices]...))
project(t::NamedTuple, i::Integer) = project(t, [i,])

# multiple columns:
function selectcols(::Val{:table}, X, c::Union{Colon, AbstractArray{I}}) where I<:Union{Symbol,Integer}
    cols = Tables.columntable(X) # named tuple of vectors
    newcols = project(cols, c)
    return Tables.materializer(X)(newcols)
end

# single column:
function selectcols(::Val{:table}, X, c::I) where I<:Union{Symbol,Integer}
    cols = Tables.columntable(X) # named tuple of vectors
    return cols[c]
end

# multiple rows (using columntable):
function selectrows(::Val{:table}, X, r::Union{Colon,AbstractVector{<:Integer}})
    cols = Tables.columntable(X)
    new_cols = NamedTuple{keys(cols)}(tuple([c[r] for c in values(cols)]...))
    return Tables.materializer(X)(new_cols)
end

# single row (using columntable):
function selectrows(::Val{:table}, X, r::Integer)
    cols = Tables.columntable(X)
    new_cols = NamedTuple{keys(cols)}(tuple([c[r:r] for c in values(cols)]...))
    return Tables.materializer(X)(new_cols)
end

## ALTERNATIVE CODE FOR PREVIOUS TWO FUNCTIONS.
## ROWTABLE SELECTION OF ROWS INSTEAD OF COLUMNTABLE SELECTION
# # multiple rows:
# function selectrows(::Val{:table}, X, r::Union{Colon,AbstractVector{<:Integer}})
#     rows = Tables.rowtable(X) # vector of named tuples
#     return Tables.materializer(X)(rows[r])
# end

# # single row:
# function selectrows(::Val{:table}, X, r::Integer)
#     rows = Tables.rowtable(X) # vector of named tuples
#     return Tables.materializer(X)([rows[r]])
# end

select(::Val{:table}, X, r::Integer, c::Symbol) = selectcols(selectrows(X, r), c)[1]
select(::Val{:table}, X, r::Integer, c) = selectcols(selectrows(X, r), c)
select(::Val{:table}, X, r, c::Symbol) = selectcols(X, c)[r]
select(::Val{:table}, X, r, c) = selectcols(selectrows(X, r), c)

nrows(::Val{:table}, X) = schema(X).nrows


## ACCESSORS FOR ABSTRACT VECTORS

selectrows(::Val{:other}, v::AbstractVector, r) = v[r]
nrows(::Val{:other}, v::AbstractVector) = length(v)
selectrows(::Val{:other}, v::CategoricalVector, r) = @inbounds v[r]

## to be replaced (not used anywhere):
## ACCESSORS FOR JULIA NDSPARSE ARRAYS (N=2)

# nrows(::Val{:sparse}, X) = maximum([r[1] for r in keys(X)])
# function select(::Val{:sparse}, X, r::Integer, c::Symbol)
#     try
#         X[r,c][1]
#     catch exception
#         exception isa KeyError || throw(exception)
#         missing
#     end
# end
# select(::Val{:sparse}, X, r::AbstractVector{<:Integer}, c::Symbol) = [select(X, s, c) for s in r]
# select(::Val{:sparse}, X, ::Colon, c::Symbol) = [select(X, s, c) for s in 1:nrows(X)]
# selectrows(::Val{:sparse}, X, r::Integer) = X[r:r,:]
# selectrows(::Val{:sparse}, X, r) = X[r,:]
# selectcols(::Val{:sparse}, X, c::Symbol) = select(X, :, c)
# selectcols(::Val{:sparse}, X, c::AbstractVector{Symbol}) = X[:,sort(c)]
# selectcols(::Val{:sparse}, X, ::Colon) = X
# select(::Val{:sparse}, X, r::Integer, c::AbstractVector{Symbol}) = X[r,sort(c)]
# select(::Val{:sparse}, X, r::Integer, ::Colon) = X[r,:]
# select(::Val{:sparse}, X, r, c) = X[r,sort(c)]

