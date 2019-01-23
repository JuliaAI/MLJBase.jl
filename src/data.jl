## CATEGORICAL ARRAY DECODER UTILITY

"""
    CategoricalDecoder(C::CategoricalArray; eltype=nothing)

Construct a decoder for transforming a `CategoricalArray{T}` object
into an ordinary array, and for re-encoding similar arrays back into a
`CategoricalArray{T}` object having the same `pool` (and, in
particular, the same levels) as `C`. If `eltype` is not specified then
the element type of the transformed array is `T`. Otherwise, the
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


## UTILITY FOR CONVERTING TABULAR DATA INTO MATRIX

""""
    MLJBase.matrix(X)

Convert an iteratable table source `X` into an `Matrix`; or, if `X` is
a `Matrix`, return `X`.

"""
function matrix(X)
    TableTraits.isiterabletable(X) || error("Argument is not an iterable table.")

    df = @from row in X begin
        @select row
        @collect DataFrames.DataFrame
    end
    return convert(Matrix, df)
    
end

matrix(X::Matrix) = X


## TOOLS FOR INDEXING QUERYVERSE ITERABLE TABLES

struct Rows end
struct Cols end
struct Schema
    nrows::Int
    ncols::Int
    names::Vector{Symbol}
    eltypes::Vector{DataType}
end

# for accessing tabular data:
retrieve(X, args...) = retrieve(Val(TableTraits.isiterabletable(X)), X, args...)
retrieve(::Val{false}, X, args...) = error("Argument is not an iterable table.")

# Note: to `retrieve` from matrices and other non-iterabletable data,
# we overload the second method above.

# project named tuple onto a tuple with only specified `labels` or indices:
project(t::NamedTuple, labels::AbstractArray{Symbol}) =
    NamedTuple{tuple(labels...)}(tuple([getproperty(t, n) for n in labels]...))
project(t::NamedTuple, indices::AbstractArray{<:Integer}) =
    NamedTuple{tuple(keys(t)[indices]...)}(tuple([t[i] for i in indices]...))

# to select columns `c` of any tabular data `X` with `retrieve(X, Cols, c)`:
# CURRENTLY RETURNS A DATAFRAME AND NOT THE ORIGINAL TYPE
function retrieve(::Val{true}, X::T, ::Type{Cols}, c::AbstractArray{I}) where {T,I<:Union{Symbol,Integer}}

    row_iterator = @from row in X begin
        @select project(row, c)
        @collect DataFrames.DataFrame # `T` does not work here unless typeof(X) has no type-parameters
    end
                    
end

# to select rows `r` of any tabular data `X` with `retrieve(X, Rows, c)`:
# CURRENTLY RETURNS A DATAFRAME AND NOT THE ORIGINAL TYPE
function retrieve(::Val{true}, X::T, ::Type{Rows}, r) where T

    row_iterator = @from row in X begin
        @select row
        @collect
    end
                    
    return @from row in row_iterator[r] begin
        @select row
        @collect DataFrames.DataFrame # `T` does not work here unless typeof(X) has no type-parameters
    end

end

# to get the number of nrows, ncols, feature names and eltypes of an
# tabular data:
function retrieve(::Val{true}, X, ::Type{Schema})

    TableTraits.isiterabletable(X) || error("Argument is not an iterable table.")

    row_iterator = @from row in X begin
        @select row
        @collect
    end

    nrows = length(row_iterator)
    if nrows == 0
        ncols = 0
        _names = Symbol[]
        _eltypes = DataType[]
    else
        row = first(row_iterator) # a named tuple
        ncols = length(row)
        _names = keys(row) |> collect
        _eltypes = DataType[eltype(x) for x in row]
    end
                    
    return Schema(nrows, ncols, _names, _eltypes)

end

# retrieve(df::JuliaDB.IndexedTable, ::Type{Rows}, r) = df[r]
# retrieve(df::JuliaDB.NextTable, ::Type{Cols}, c) = select(df, c)
# retrieve(df::JuliaDB.NextTable, ::Type{Names}) = getfields(typeof(df.columns.columns))
# retrieve(df::JuliaDB.NextTable, ::Type{NRows}) = length(df)

retrieve(::Val{false}, A::Matrix, ::Type{Rows}, r) = A[r,:]
retrieve(::Val{false}, A::Matrix, ::Type{Cols}, c) = A[:,c]
function retrieve(::Val{false}, A::Matrix{T}, ::Type{Schema}) where T
    nrows = size(A, 1)
    ncols = size(A, 2)
    _names = [Symbol(:x, j) for j in 1:ncols]
    _eltypes = fill(T, ncols)
    return Schema(nrows, ncols, _names, _eltypes)
end

retrieve(::Val{false}, v::Vector, ::Type{Rows}, r) = v[r]
retrieve(::Val{false}, v::Vector, ::Type{Cols}, c) = error("Abstract vectors are not column-indexable.")
retrieve(::Val{false}, v::Vector{T}, ::Type{Schema}) where T = Schema(length(v), 1, [:x], [T,])
retrieve(::Val{false}, v::CategoricalArray{T,1,S} where {T,S}, ::Type{Rows}, r) = @inbounds v[r]
retrieve(::Val{false}, v::CategoricalArray{T,1,S} where {T,S}, ::Type{Cols}, r) =
    error("Categorical vectors are not column-indexable.")
retrieve(::Val{false}, v::CategoricalArray{T,1,S} where {T,S}, ::Type{Schema}) =
    retrieve(Val(false), collect(v), Schema)

