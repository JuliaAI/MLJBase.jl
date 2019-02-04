## CATEGORICAL ARRAY DECODER UTILITY

"""
    CategoricalDecoder(C::CategoricalArray; eltype=nothing)

Construct a decoder for transforming a `CategoricalArray{T}` object
into an ordinary array, and for re-encoding similar arrays back into a
`CategoricalArray{T}` object having the same `pool` (and, in
particular, the same levels) as `C`. If `eltype` is not specified then
the element type of the transformed array is `T`. Otherwise, 
element type is `eltype` and the elements are promotions of the
internal (integer) `ref`s of the `CategoricalArray`. One
must have `R <: eltype <: Real` where `R` is the reference type of the
`CategoricalArray` (usually `UInt32`).

    transform(decoder::CategoricalDecoder, C::CategoricalArray)

Transform `C` into an ordinary `Array`.

    inverse_transform(decoder::CategoricalDecoder, A::Array)

Transform an array `A` suitably compatible with `decoder` into a
`CategoricalArray` having the same `pool` as `C`.

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

"""
struct CategoricalDecoder{I<:Real,T,N,R<:Integer}  # I the output eltype
    pool::CategoricalPool{T,R} # abstract type, not optimal
    use_original_type::Bool
    CategoricalDecoder{I,T,N,R}(X::CategoricalArray{T,N,R}, use_original_type) where {I,T,N,R}  =
        new(X.pool, use_original_type)
end

function CategoricalDecoder(X::CategoricalArray{T,N,R}; eltype=nothing) where {T,N,R}
    if eltype ==  nothing
        eltype = R # any integer type will do here; not used
        use_original_type = true
    else
        use_original_type = false
    end
    return CategoricalDecoder{eltype,T,N,R}(X, use_original_type)
end

function transform(decoder::CategoricalDecoder{I,T,N,R}, C::CategoricalArray) where {I,T,N,R}
    if decoder.use_original_type
        return collect(C)
    else
        return broadcast(C.refs) do element
            ref = convert(I, element)
        end
    end
end

function inverse_transform(decoder::CategoricalDecoder{I,T,N,R}, A::Array{J}) where {I,T,N,R,J<:Union{I,T}}
    if decoder.use_original_type
        refs = broadcast(A) do element
            decoder.pool.invindex[element]
        end
    else
        refs = broadcast(A) do element
            round(R, element)
        end
    end
    return CategoricalArray{T,N}(refs, decoder.pool)

end


## UTILITY FOR CONVERTING BETWEEN TABULAR DATA AND MATRICES

""""
    MLJBase.matrix(X)

Convert a generic table source `X` into an `Matrix`; or, if `X` is
a `AbstractMatrix`, return `X`. Optimized for column-based sources.

"""
function matrix(X)

    Tables.istable(X) || error("Argument is not tabular.")
    cols = Tables.columns(X) # property-accessible object

    return reduce(hcat, [getproperty(cols, ftr) for ftr in propertynames(cols)])

end
matrix(X::AbstractMatrix) = X


"""
    MLJBase.table(cols; prototype=cols)

Convert a named tuple of vectors `cols`, into a table. The table
type returned is the "preferred sink type" for `prototype` (see the
Tables.jl documentation). 

    MLJBase.table(X::AbstractMatrix; names=nothing, prototype=nothing)

Convert an abstract matrix `X` into a table with `names` (a tuple of
symbols) as column names, or with labels `(:x1, :x2, ..., :xn)` where
`n=size(X, 2)`, if `names` is not specified.  Equivalent to `table(cols,
prototype=prototype)` where `cols` is the named tuple of columns of
`X`, with `keys(cols) = names`.

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
               

## TOOLS FOR INDEXING TABLES 

struct Schema{names, eltypes}
    nrows::Int
    ncols::Int
    Schema(names, eltypes, nrows, ncols) = new{names,Tuple{eltypes...}}(nrows, ncols)
end

function Base.getproperty(s::Schema{names,eltypes}, field::Symbol) where {names,eltypes}
    if field === :names
        return names
    elseif field === :eltypes
        return Tuple(fieldtype(eltypes, i) for i = 1:fieldcount(eltypes))
    else
        return getfield(s, field)
    end
end

"""
    selectrows(X, r)

Select single or multiple rows from any object `X` for which
`Tables.istable(X)` is true.  The object returned is a table of the
preferred sink type of `typeof(X)`, even a single row is selected.

The method is overloaded to additionally work on abstract matrices
and vectors.

"""
selectrows(X, r) = selectrows(Val(Tables.istable(X)), X, r)
selectrows(::Val{false}, X, r) = error("Argument is not tabular.")

"""
    selectcols(X, c)

Select single or multiple columns from any object `X` for which
`Tables.istable(X)` is true. If `c` is an abstract vector of integers
or symbols, then the object returned is a table of the preferred sink
type of `typeof(X)`. If `c` is a *single* integer or column, then
a `Vector` or `CategoricalVector` is returned. 

The method is overloaded to additionally work on abstract matrices.

"""
selectcols(X, c) = selectcols(Val(Tables.istable(X)), X, c)
selectcols(::Val{false}, X, c) = error("Argument is not tabular.")

"""
    schema(X)

Returns a struct with properties `names`, `eltypes`, `nrows, `ncols`,
with the obvious meanings. Here `X` is any object for which
`Tables.istable(X)` is true, an abstract matrix, or vector. 
"""
schema(X) = schema(Val(Tables.istable(X)), X)
schema(::Val{false}, X) = error("Argument is not tabular.")

# project named tuple onto a tuple with only specified `labels` or indices:
project(t::NamedTuple, labels::AbstractArray{Symbol}) = NamedTuple{tuple(labels...)}(t)
project(t::NamedTuple, label::Colon) = t
project(t::NamedTuple, label::Symbol) = project(t, [label,])
project(t::NamedTuple, indices::AbstractArray{<:Integer}) =
    NamedTuple{tuple(keys(t)[indices]...)}(tuple([t[i] for i in indices]...))
project(t::NamedTuple, i::Integer) = project(t, [i,])

# multiple columns:
function selectcols(::Val{true}, X, c::Union{Colon, AbstractArray{I}};
                  prototype=nothing) where I<:Union{Symbol,Integer}
    prototype2 = (prototype == nothing ? X : prototype)
    cols = Tables.columntable(X) # named tuple of vectors
    newcols = project(cols, c)
    return Tables.materializer(prototype2)(newcols)
end
                    
# single column:
function selectcols(::Val{true}, X, c::I;
                  prototype=nothing) where I<:Union{Symbol,Integer}
    prototype2 = (prototype == nothing ? X : prototype)
    cols = Tables.columntable(prototype2) # named tuple of vectors
    return cols[c]
end

# multiple rows:
function selectrows(::Val{true}, X::T, r::Union{Colon,AbstractVector{I}};
                  prototype=nothing) where {T,I<:Integer}
    prototype2 = (prototype == nothing ? X : prototype)
    rows = Tables.rowtable(X) # vector of named tuples
    return Tables.materializer(prototype2)(rows[r])
end

# single row:
function selectrows(::Val{true}, X::T, r::Integer;
                  prototype=nothing) where {T,I<:Integer}
    prototype2 = (prototype == nothing ? X : prototype)
    rows = Tables.rowtable(X) # vector of named tuples
    return Tables.materializer(prototype2)([rows[r]])
end

function schema(::Val{true}, X)

    row_iterator = Tables.rows(X)
    nrows = length(row_iterator)

    if nrows == 0
        ncols = 0
        names = ()
        eltypes = ()
    else
        row = first(row_iterator) # a named tuple
        ncols = length(propertynames(row))
        s = Tables.schema(X)
        names, eltypes = s.names, s.types
    end
                    
    return Schema(names, eltypes, nrows, ncols)

end

# select(df::JuliaDB.IndexedTable, ::Type{Rows}, r) = df[r]
# select(df::JuliaDB.NextTable, ::Type{Cols}, c) = select(df, c)
# select(df::JuliaDB.NextTable, ::Type{Names}) = getfields(typeof(df.columns.columns))
# select(df::JuliaDB.NextTable, ::Type{NRows}) = length(df)

selectrows(::Val{false}, A::AbstractMatrix, r; prototype=nothing) = A[r,:]
selectcols(::Val{false}, A::AbstractMatrix, c; prototype=nothing) = A[:,c]
function schema(::Val{false}, A::AbstractMatrix{T}) where T
    nrows = size(A, 1)
    ncols = size(A, 2)
    names = tuple([Symbol(:x, j) for j in 1:ncols]...)
    eltypes = tuple(fill(T, ncols)...)
    return Schema(names, eltypes, nrows, ncols)
end

selectrows(::Val{false}, v::AbstractVector, r; prototype=nothing) = v[r]
selectcols(::Val{false}, v::AbstractVector, c; prototype=nothing) = error("AbstractVectors are not column-indexable.")
schema(::Val{false}, v::AbstractVector{T}) where T = Schema((:x,), (T,), length(v), 1)
selectrows(::Val{false}, v::CategoricalArray{T,1,S} where {T,S}, r; prototype=nothing) = @inbounds v[r]
selectcols(::Val{false}, v::CategoricalArray{T,1,S} where {T,S}, c; prototype=nothing) =
    error("Categorical vectors are not column-indexable.")
schema(::Val{false}, v::CategoricalArray{T,1,S} where {T,S}) =
    schema(Val(false), collect(v))

