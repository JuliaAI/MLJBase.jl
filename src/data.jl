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

julia> decoder = MLJInterface.CategoricalDecoder(C, eltype=Float64);
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


## UTILITY FOR CONVERTING INTERABLE TABLE INTO MATRIX

""""
    MLJInterface.matrix(X)

Convert an iteratable table source `X` into an `Matrix`; or, if `X` is
a `Matrix`, return `X`.

"""
function matrix(X)
    TableTraits.isiterabletable(X) || error("Argument is not an iterable table.")

    df= Query.@from row in X begin
        Query.@select row
        Query.@collect DataFrames.DataFrame
    end
    return convert(Matrix, df)

end

matrix(X::Matrix) = X


