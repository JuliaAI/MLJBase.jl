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
    MLJBase.table(cols; prototype=DataFrames)

Convert a named tuple of vectors `cols`, into a table. The table
type returned is the "preferred sink type" for `prototype` (see the
Tables.jl documentation), which is generally the type of `prototype`
itself, or a named tuple of vectors (in which case `cols` itself is
returned).

    MLJBase.table(X; prototype=DataFrames())

Convert an abstract matrix `X` into a table with column names, `(:x1,
:x2, ..., :xn)` where `n=size(X, 2)`.  Equivalent to `table(cols,
prototype=prototype)` where `cols` is the named tuple of columns of
`X`, with `keys(cols) = (:x1, :x2, ..., :xn)`.

"""
function table(cols::NamedTuple; prototype=DataFrames.DataFrame())
    Tables.istable(prototype) || error("prototype is not tabular.")
    return Tables.materializer(prototype)(cols)
end
function table(X::AbstractMatrix; prototype=DataFrames.DataFrame())
    names = tuple([Symbol(:x, j) for j in 1:size(X, 2)]...)
    cols = NamedTuple{names}(tuple([X[:,j] for j in 1:size(X, 2)]...))
    return table(cols; prototype=prototype)

end

               

## TOOLS FOR INDEXING QUERYVERSE ITERABLE TABLES

struct Rows end
struct Cols end
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
    select(X, Rows, r)
    select(X, Cols, c)

Select single or multiple rows or columns ofrom a `X` for which
`Tables.istable(X)` is true. In the cas of columns, `c`, can be an
abstract vector of integers or symbols; if `c` is a *single* integer
or column, then a `Vector` or `CategoricalVector` is returned. In all
other cases (including a single row request) the object returned is a
table of the preferred sink type of `typeof(X)`.

    select(X, Schema)

Returns a struct with properties `names`, `eltypes`, `nrows, `ncols`,
with the obvious meanings.

The above methods are also overloaded to work on abstract matrices
and vectors, where this makes sense.

"""

select(X, args...) = select(Val(Tables.istable(X)), X, args...)
select(::Val{false}, X, args...) = error("Argument is not tabular.")

# Note: to `select` from matrices and other non-tabluar data,
# we overload the second method above.

# project named tuple onto a tuple with only specified `labels` or indices:
project(t::NamedTuple, labels::AbstractArray{Symbol}) = NamedTuple{tuple(labels...)}(t)
project(t::NamedTuple, label::Colon) = t
project(t::NamedTuple, label::Symbol) = project(t, [label,])
project(t::NamedTuple, indices::AbstractArray{<:Integer}) =
    NamedTuple{tuple(keys(t)[indices]...)}(tuple([t[i] for i in indices]...))
project(t::NamedTuple, i::Integer) = project(t, [i,])

# to select columns `c` of any tabular data `X` with `select(X, Cols, c)`:
function select(::Val{true}, X, ::Type{Cols}, c::Union{Colon, AbstractArray{I}};
                  prototype=nothing) where I<:Union{Symbol,Integer}
    prototype2 = (prototype == nothing ? X : prototype)
    cols = Tables.columntable(X) # named tuple of vectors
    newcols = project(cols, c)
    return Tables.materializer(prototype2)(newcols)
end
                    
# to select a single column `c` of any tabular data `X` with
# `select(X, Cols, c)`:
function select(::Val{true}, X::T, ::Type{Cols}, c::I;
                  prototype=nothing) where {T, I<:Union{Symbol,Integer}}
    prototype2 = (prototype == nothing ? X : prototype)
    cols = Tables.columntable(prototype2) # named tuple of vectors
    return cols[c]
end

# to select rows `r` of any tabular data `X` with `select(X, Rows, r)`:
function select(::Val{true}, X::T, ::Type{Rows}, r;
                  prototype=nothing) where T
    prototype2 = (prototype == nothing ? X : prototype)
    rows = Tables.rowtable(X) # vector of named tuples
    return Tables.materializer(prototype2)(rows[r])
end

# to get the number of nrows, ncols, feature names and eltypes of
# tabular data:
function select(::Val{true}, X, ::Type{Schema})

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

select(::Val{false}, A::AbstractMatrix, ::Type{Rows}, r; prototype=nothing) = A[r,:]
select(::Val{false}, A::AbstractMatrix, ::Type{Cols}, c; prototype=nothing) = A[:,c]
function select(::Val{false}, A::AbstractMatrix{T}, ::Type{Schema}) where T
    nrows = size(A, 1)
    ncols = size(A, 2)
    names = tuple([Symbol(:x, j) for j in 1:ncols]...)
    eltypes = tuple(fill(T, ncols)...)
    return Schema(names, eltypes, nrows, ncols)
end

select(::Val{false}, v::AbstractVector, ::Type{Rows}, r; prototype=nothing) = v[r]
select(::Val{false}, v::AbstractVector, ::Type{Cols}, c; prototype=nothing) = error("AbstractVectors are not column-indexable.")
select(::Val{false}, v::AbstractVector{T}, ::Type{Schema}) where T = Schema((:x,), (T,), length(v), 1)
select(::Val{false}, v::CategoricalArray{T,1,S} where {T,S}, ::Type{Rows}, r; prototype=nothing) = @inbounds v[r]
select(::Val{false}, v::CategoricalArray{T,1,S} where {T,S}, ::Type{Cols}, r; prototype=nothing) =
    error("Categorical vectors are not column-indexable.")
select(::Val{false}, v::CategoricalArray{T,1,S} where {T,S}, ::Type{Schema}) =
    select(Val(false), collect(v), Schema)

