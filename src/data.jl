## CATEGORICAL ARRAY DECODER UTILITY

"""
    reconstruct(A)

For reconstructing categorical arrays from their elements alone. Here
`A` is of type `Array{T}` where `T` is a subtype of
`CategoricalString` or `CategoricalValue`. The function `reconstruct` has
the property that `reconstruct(broadcast(identity, A)) == A`, whenever `A`
is a `CategoricalArray`. In other words, `reconstruct` is a left-inverse
for the function `A -> broadcast(identity, A)` that strips a
CategoricalArray of its "categorical wrapper".

"""
function reconstruct(A::Array{<:CategoricalValue{T},N}) where {T,N}
    !isempty(A) || error("Cannot reconstruct an empty array")
    proto_element = A[1]
    pool = A[1].pool
    refs = map(x -> x.level, A)
    return CategoricalArray{T,N}(refs, pool)
end
function reconstruct(A::Array{<:CategoricalString,N}) where {T,N}
    !isempty(A) || error("Cannot reconstruct an empty array")
    proto_element = A[1]
    pool = A[1].pool
    refs = map(x -> x.level, A)
    return CategoricalArray{String,N}(refs, pool)
end


"""
    CategoricalDecoder(C::CategoricalArray)
    CategoricalDecoder(C::CategoricalArray, eltype, start_at_zero=false)

Construct a decoder for transforming a `CategoricalArray{T}` object
into an ordinary array, and for re-encoding similar arrays back into a
`CategoricalArray{T}` object having the same `pool` (and, in
particular, the same levels) as `C`. If `eltype` is not specified then
the element type of the transformed array is `T`. Otherwise, the
element type is `eltype` and the elements are conversions to `eltype`
of the internal (unsigned integer) `ref`s of the `CategoricalArray`,
shifted backwards by one if `start_at_zero=false`. One must have
`eltype <: Real`.

If `eltype = Bool`, then `start_at_zero` is ignored.

    transform(decoder::CategoricalDecoder, C::CategoricalArray)

Transform `C` into an ordinary `Array`.

    inverse_transform(decoder::CategoricalDecoder, A::Array)

Transform an array `A` suitably compatible with `decoder` into a
`CategoricalArray` having the same `pool` as `C`.

    levels(decoder::CategoricalDecoder)
    levels_seen(decoder::CategoricaDecoder)

Return, respectively, all levels in pool of the categorical vector `C`
used to construct `decoder` (ie, `levels(C)`), and just those levels
explicitly appearing as entries of `C` (ie, `unique(C)`).

### Example

````
julia> using CategoricalArrays
julia> C = categorical(["a" "b"; "a" "c"])
2×2 CategoricalArray{String,2,UInt32}:
 "a"  "b"
 "a"  "c"

julia> decoder = MLJBase.CategoricalDecoder(C, eltype=Float64);
julia> A = transform(decoder, C)
2×2 Array{Float64,2}:
 1.0  2.0
 1.0  3.0

julia> inverse_transform(decoder, A[1:1,:])
1×2 CategoricalArray{String,2,UInt32}:
 "a"  "b"

julia> levels(ans)
3-element Array{String,1}:
 "a"
 "b"
 "c"
````

"""
struct CategoricalDecoder{I<:Real,B,V,N,R<:Integer,C}

    # I = output eltype if not using original type (junk otherwise)
    # B is boolean, whether to use original type or I
    # N is the dimension of the array to be encoded/decoded

    pool::CategoricalPool{V,R,C}
    levels_seen::Vector{V}
    start_at_zero::Bool

end

CategoricalArrays.levels(d::CategoricalDecoder) = levels(d.pool)
levels_seen(d::CategoricalDecoder) = d.levels_seen

# constructors:
# using original type:
CategoricalDecoder(X::CategoricalArray{T,N,R,V,C}) where {T,N,R,V,C} = 
    CategoricalDecoder{R,true,V,N,R,C}(X.pool, unique(X), false) # the first `R` will never be used

# using specified type:
function CategoricalDecoder(X::CategoricalArray{T,N,R,V,C},
                            eltype,
                            start_at_zero=false) where {T,N,R,V,C}
    
    eltype <: Real || error("eltype must be a subtype of Real.")
    if eltype <: Integer
        if eltype  <: Unsigned && start_at_zero
            error("Integer relabeling cannot start at zero if eltype <: Unsigned. ")
        elseif eltype <: Bool
            start_at_zero = true
        end
    end
    
    return CategoricalDecoder{eltype,false,V,N,R,C}(X.pool, unique(X), start_at_zero)
end

# using original type:
transform(decoder::CategoricalDecoder, C::CategoricalArray) = collect(C)

# using specified type:
transform(decoder::CategoricalDecoder{I,false}, C::CategoricalArray) where I =
    broadcast(C.refs) do element
        ref = convert(I, element - decoder.start_at_zero)
    end

# using original type:
function inverse_transform(decoder::CategoricalDecoder{I,true,V,N}, A::Array) where {I,V,N}
    refs = broadcast(A) do element
        get(decoder.pool, element)
    end
    return CategoricalArray{V,N}(refs, decoder.pool)
end

# using specified type:
function inverse_transform(decoder::CategoricalDecoder{I,false,V,N,R}, A::Array) where {I,V,N,R}
    refs = broadcast(A) do element
        round(R, element + decoder.start_at_zero)
    end
    return CategoricalArray{V,N}(refs, decoder.pool)
end

# special boolean case:
function inverse_transform(decoder::CategoricalDecoder{I,false,V,N,R}, A::Union{Array{Bool},BitArray}) where {I,V,N,R}
    refs = broadcast(A) do element
        round(R, element + decoder.start_at_zero)
    end
    return CategoricalArray{V,N}(refs, decoder.pool)
end


## TABULAR DATA

# HACK: Tables.istable is too permissive (see Tables.jl #74) so we
# must define our local version.
isindexedtable(X) = isdefined(X, :cardinality)
isndsparse(X) = isdefined(X, :data_buffer) 
istable(X) = Tables.istable(X) & (Tables.rowaccess(X) || Tables.columnaccess(X)) ||
                                  isindexedtable(X) 

"""
    container_type(X)

Return `:table`, `:sparse`, or `:other`, according to whether `X` is a
supported table format, a supported sparse table format, or something
else.

The first two formats, together abstract vectors, support the
`MLJBase` accessor methods `selectrows`, `selectcols`, `select`,
`nrows`, `schema`, and `union_scitypes`.

"""
function container_type(X)
    if istable(X)
        return :table
    elseif isndsparse(X)
        return :sparse
    else
        return :other
    end
end


## UTILITY FOR CONVERTING BETWEEN TABULAR DATA AND MATRICES

"""
    MLJBase.matrix(X)

Convert a table source `X` into an `Matrix`; or, if `X` is
a `AbstractMatrix`, return `X`. Optimized for column-based sources.

If instead X is a sparse table, then a `SparseMatrixCSC` object is
returned. The integer relabelling of column names follows the
lexicographic ordering (as indicated by `schema(X).names`).

"""
matrix(X) = matrix(Val(container_type(X)), X)
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
    istable(prototype) || error("prototype is not tabular.")
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
selectrows(X, r) = selectrows(Val(container_type(X)), X, r)
selectrows(::Val{:other}, X, r) = throw(ArgumentError)

"""
    selectcols(X, c)

Select single or multiple columns from any table or sparse table
`X`. If `c` is an abstract vector of integers or symbols, then the
object returned is a table of the preferred sink type of
`typeof(X)`. If `c` is a *single* integer or column, then a `Vector`
or `CategoricalVector` is returned.

"""
selectcols(X, c) = selectcols(Val(container_type(X)), X, c)
selectcols(::Val{:other}, X, c) = throw(ArgumentError)

"""
    select(X, r, c)

Select element of a table or sparse table at row `r` and column
`c`. In the case of sparse data where the key `(r, c)`, zero or
`missing` is returned, depending on the value type.

See also: selectrows, selectcols

"""
select(X, r, c) = select(Val(container_type(X)), X, r, c)
select(::Val{:other}, X, r, c) = throw(ArgumentError)

"""
    schema(X)

Returns a struct with properties `names`, `types`
with the obvious meanings. Here `X` is any table or sparse table.

"""
schema(X) = schema(Val(container_type(X)), X)
schema(::Val{:other}, X) = throw(ArgumentError)

"""
    nrows(X)

Return the number of rows in a table, sparse table, or abstract vector.

"""
nrows(X) = nrows(Val(container_type(X)), X)
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

function schema(::Val{:table}, X)
    istable(X) || throw(ArgumentError)
    if !Tables.columnaccess(X)
        return Tables.schema(Tables.rows(X))
    else
        return Tables.schema(Tables.columns(X))
    end
end

function nrows(::Val{:table}, X)
    if !Tables.columnaccess(X)
        return length(collect(X))
    else
        cols = Tables.columntable(X)
        !isempty(cols) || return 0
        return length(cols[1])
    end
end


## ACCESSORS FOR ABSTRACT VECTORS

selectrows(::Val{:other}, v::AbstractVector, r) = v[r]
nrows(::Val{:other}, v::AbstractVector) = length(v)
selectrows(::Val{:other}, v::CategoricalVector, r) = @inbounds v[r]


## ACCESSORS FOR JULIA NDSPARSE ARRAYS (N=2)

nrows(::Val{:sparse}, X) = maximum([r[1] for r in keys(X)])
function select(::Val{:sparse}, X, r::Integer, c::Symbol)
    try
        X[r,c][1]
    catch exception
        exception isa KeyError || throw(exception)
        missing
    end
end
select(::Val{:sparse}, X, r::AbstractVector{<:Integer}, c::Symbol) = [select(X, s, c) for s in r]
select(::Val{:sparse}, X, ::Colon, c::Symbol) = [select(X, s, c) for s in 1:nrows(X)]
selectrows(::Val{:sparse}, X, r::Integer) = X[r:r,:]
selectrows(::Val{:sparse}, X, r) = X[r,:]
selectcols(::Val{:sparse}, X, c::Symbol) = select(X, :, c)
selectcols(::Val{:sparse}, X, c::AbstractVector{Symbol}) = X[:,sort(c)]
selectcols(::Val{:sparse}, X, ::Colon) = X
select(::Val{:sparse}, X, r::Integer, c::AbstractVector{Symbol}) = X[r,sort(c)]
select(::Val{:sparse}, X, r::Integer, ::Colon) = X[r,:]
select(::Val{:sparse}, X, r, c) = X[r,sort(c)]

function schema(::Val{:sparse}, X)
    names = sort(unique([r[2] for r in keys(X)]))
    types = [eltype(selectcols(X, name)) for name in names]
    return Tables.Schema(names, types)
end

