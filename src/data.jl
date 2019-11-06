## SPLITTING DATA SETS

"""
    partition(rows::AbstractVector{Int}, fractions...; shuffle=false, rng=Random.GLOBAL_RNG)

Splits the vector `rows` into a tuple of vectors whose lengths are
given by the corresponding `fractions` of `length(rows)`. The last
fraction is not provided, as it is inferred from the preceding
ones. So, for example,

    julia> partition(1:1000, 0.2, 0.7)
    (1:200, 201:900, 901:1000)

If `rng` is an integer, then `MersenneTwister(rng)` is the random
number generator used for bagging. Otherwise some `AbstractRNG` object
is expected.

"""
function partition(rows::AbstractVector{Int}, fractions...; shuffle::Bool=false, rng=Random.GLOBAL_RNG)
    rows = collect(rows)

    if rng isa Integer
        rng = MersenneTwister(rng)
    end

    shuffle && shuffle!(rng, rows)
    rowss = []
    if sum(fractions) >= 1
        throw(DomainError)
    end
    n_patterns = length(rows)
    first = 1
    for p in fractions
        n = round(Int, p*n_patterns)
        n == 0 ? (@warn "A split has only one element"; n = 1) : nothing
        push!(rowss, rows[first:(first + n - 1)])
        first = first + n
    end
    if first > n_patterns
        @warn "Last vector in the split has only one element."
        first = n_patterns
    end
    push!(rowss, rows[first:n_patterns])
    return tuple(rowss...)
end


"""
    t1, t2, ...., tk = unnpack(table, t1, t2, ... tk; wrap_singles=false)

Split any Tables.jl compatible `table` into smaller tables (or
vectors) `t1, t2, ..., tk` by making selections *without replacement*
from the column names defined by the tests `t1`, `t2`, ...,
`tk`. A *test* is any object `t` such that `t(name)` is `true`
or `false` for each column `name::Symbol` of `table`.

Whenever a returned table contains a single column, it is converted to
a vector unless `wrap_singles=true`.

Scientific type conversions can be optionally specified (note
semicolon):

    unpack(table, t...; wrap_singles=false, col1=>scitype1, col2=>scitype2, ... )

### Example

```
julia> table = DataFrame(x=[1,2], y=['a', 'b'], z=[10.0, 20.0], w=[:A, :B])
julia> Z, XY = unpack(table, ==(:z), !=(:w);
               :x=>Continuous, :y=>Multiclass)
julia> XY
2×2 DataFrame
│ Row │ x       │ y            │
│     │ Float64 │ Categorical… │
├─────┼─────────┼──────────────┤
│ 1   │ 1.0     │ 'a'          │
│ 2   │ 2.0     │ 'b'          │

julia> Z
2-element Array{Float64,1}:
 10.0
 20.0
```
"""
function unpack(X, tests...; wrap_singles=false, pairs...)

    if isempty(pairs)
        Xfixed = X
    else
        Xfixed = ScientificTypes.coerce(X, pairs...)
    end

    unpacked = Any[]
    names_left = schema(Xfixed).names |> collect
    history = ""
    counter = 1
    for c in tests
        names = filter(c, names_left)
        filter!(!in(names), names_left)
        history *= "selection $counter: $names\n remaining: $names_left\n"
        isempty(names) &&
            error("Empty column selection encountered at selection $counter"*
                  "\n$history")
        length(names) == 1 && !wrap_singles && (names = names[1])
        push!(unpacked, selectcols(Xfixed, names))
        counter += 1
    end
    return Tuple(unpacked)
end


## DEALING WITH CATEGORICAL ELEMENTS

const CategoricalElement{U} = Union{CategoricalValue{<:Any,U},CategoricalString{U}}


"""
    classes(x)

All the categorical elements with in the same pool as `x` (including `x`),
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
classes(p::CategoricalPool) = [p.valindex[p.invindex[v]] for v in p.levels]
classes(x::CategoricalElement) = classes(x.pool)

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
int(A::AbstractArray) = broadcast(int, A)

# get the integer representation of a level given pool (private
# method):
int(pool::CategoricalPool, level) =  pool.order[pool.invindex[level]]

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

*Warning:* It is *not* true that `int(d(u)) == u` always holds.

See also: [`int`](@ref), [`classes`](@ref).
"""
decoder(element::CategoricalElement) =
    CategoricalDecoder(element.pool, sortperm(element.pool.order))

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
    MLJBase.matrix(X; transpose=false)

Convert a Tables.jl compatible table source `X` into an `Matrix`; or,
if `X` is a `AbstractMatrix`, return `X`. Optimized for column-based
sources. Rows of the table or input matrix, correspond to rows of the
output, unless `transpose=true`.
"""
matrix(X; kwargs...) = matrix(Val(ScientificTypes.trait(X)), X; kwargs...)
matrix(::Val{:other}, X; kwargs...) = throw(ArgumentError)
matrix(::Val{:other}, X::AbstractMatrix; transpose=false) =
    transpose ? permutedims(X) : X

matrix(::Val{:table}, X; kwargs...) = Tables.matrix(X; kwargs...)

# matrix(::Val{:table, X)
#     cols = Tables.columns(X) # property-accessible object
#     mat = reduce(hcat, [getproperty(cols, ftr) for ftr in propertynames(cols)])
#     # tightest eltype:
#     return broadcast(identity, mat)
# end

# function matrix(::Val{:sparse}, X)
#     K = keys(X)
#     features = schema(X).names
#     index_given_feature = Dict{Symbol,Int}()
#     for j in eachindex(features)
#         index_given_feature[features[j]] = j
#     end
#     I = [k[1] for k in K]
#     J = [index_given_feature[k[2]] for k in K]
#     V = [v[1] for v in values(X)]
#     return sparse(I, J, V)
# end

"""
    MLJBase.table(columntable; prototype=nothing)

Convert a named tuple of vectors or tuples `columntable`, into a table
of the "preferred sink type" of `prototype`. This is often the type of
`prototype` itself, when `prototype` is a sink; see the Tables.jl
documentation. If `prototype` is not specified, then a named tuple of
vectors is returned.

    MLJBase.table(A::AbstractMatrix; names=nothing, prototype=nothing)

Wrap an abstract matrix `A` as a Tables.jl compatible table with the
specified column `names` (a tuple of symbols). If `names` are not
specified, `names=(:x1, :x2, ..., :xn)` is used, where `n=size(A, 2)`.

If a `prototype` is specified, then the matrix is materialized as a
table of the preferred sink type of `prototype`, rather than
wrapped. Note that if `protottype` is *not* specified, then
`MLJ.matrix(MLJ.table(A))` is essentially a non-operation.
"""
function table(cols::NamedTuple; prototype=NamedTuple())
    Tables.istable(prototype) || error("`prototype` is not a table. ")
    if !Tables.istable(cols)
        tuple_of_vectors = Tuple([collect(v) for v in values(cols)])
        names = keys(cols)
        cols = NamedTuple{names}(tuple_of_vectors)
        Tables.istable(cols) || throw(ArgumentError(""))
    end
    return Tables.materializer(prototype)(cols)
end
function table(A::AbstractMatrix; names=nothing, prototype=nothing)
    if names == nothing
        _names = [Symbol(:x, j) for j in 1:size(A, 2)]
    else
        _names = collect(names)
    end
    matrix_table = Tables.table(A, header=_names)
    if prototype === nothing
        return matrix_table
    else
        return Tables.materializer(prototype)(matrix_table)
    end
end


## UNIFIED API FOR ACCESSING TABLES, MATRICES AND VECTORS

"""
    selectrows(X, r)

Select single or multiple rows from any table, or abstract vector `X`,
or matrix.  If `X` is tabular, the object returned is a table of the
preferred sink type of `typeof(X)`, even if only a single row is selected.

"""
selectrows(X, r) = selectrows(Val(ScientificTypes.trait(X)), X, r)
selectrows(::Val{:other}, X, r) = throw(ArgumentError)

"""
    selectcols(X, c)

Select single or multiple columns from any table or matrix `X`. If `c`
is an abstract vector of integers or symbols, then the object returned
is a table of the preferred sink type of `typeof(X)`. If `c` is a
*single* integer or column, then an `AbstractVector` is returned.

"""
selectcols(X, c) = selectcols(Val(ScientificTypes.trait(X)), X, c)
selectcols(::Val{:other}, X, c) = throw(ArgumentError)

"""
    select(X, r, c)

Select element of a table or matrix at row `r` and column
`c`. In the case of sparse data where the key `(r, c)`, zero or
`missing` is returned, depending on the value type.

See also: [`selectrows`](@ref), [`selectcols`](@ref).

"""
select(X, r, c) = select(Val(ScientificTypes.trait(X)), X, r, c)
select(::Val{:other}, X, r, c) = throw(ArgumentError)

"""
    nrows(X)

Return the number of rows in a table, abstract vector or abstract
matrix.

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
selectrows(::Val{:other}, v::CategoricalVector, r) = @inbounds v[r]

# single row selection must return a matrix!
selectrows(::Val{:other}, v::AbstractVector, r::Integer) = v[r:r]
selectrows(::Val{:other}, v::CategoricalVector, r::Integer) = @inbounds v[r:r]

nrows(::Val{:other}, v::AbstractVector) = length(v)


## ACCESSORS FOR ABSTRACT MATRICES

selectrows(::Val{:other}, A::AbstractMatrix, r) = A[r, :]
selectrows(::Val{:other}, A::CategoricalMatrix, r) = @inbounds A[r, :]

# single row selection must return a matrix!
selectrows(::Val{:other}, A::AbstractMatrix, r::Integer) = A[r:r, :]
selectrows(::Val{:other}, A::CategoricalMatrix, r::Integer) =
    @inbounds A[r:r, :]

nrows(::Val{:other}, A::AbstractMatrix) = size(A, 1)


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
