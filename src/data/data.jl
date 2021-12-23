# SPLITTING DATA SETS

# Helper function for partitioning in the non-stratified case
function _partition(rows, fractions, ::Nothing)
    # container for the row selections (head:tail)
    n_splits = length(fractions) + 1
    heads    = zeros(Int, n_splits)
    tails    = zeros(Int, n_splits)
    n_rows   = length(rows)
    head     = 1
    for (i, p) in enumerate(fractions)
        n = round(Int, p * n_rows)
        iszero(n) && (@warn "A split has only one element."; n = 1)
        # update tail
        tail = head + n - 1
        # store
        heads[i] = head
        tails[i] = tail
        # update head
        head = tail + 1
    end
    if head > n_rows
        @warn "Last vector in the split has only one element."
        head = n_rows
    end
    heads[end] = head
    tails[end] = n_rows
    return tuple((rows[h:t] for (h, t) in zip(heads, tails))...)
end

_make_numerical(v::AbstractVector) =
    throw(ArgumentError("`stratify` must have `Count`, `Continuous` "*
                        "or `Finite` element scitpye. Consider "*
                        "`coerce(stratify, Finite)`. "))
_make_numerical(v::AbstractVector{<:Union{Missing,Real}}) = v
_make_numerical(v::AbstractVector{<:Union{Missing,CategoricalValue}}) =
                                int.(v)

# Helper function for partitioning in the stratified case
function _partition(rows, fractions, raw_stratify::AbstractVector)
    stratify = _make_numerical(raw_stratify)
    length(stratify) == length(rows) ||
        throw(ArgumentError("The stratification vector must "*
                            "have as many entries as " *
                            "the rows to partition."))
    uv    = unique(stratify)
    # construct table (n_classes * idx_of_that_class)
    # NOTE use of '===' is important to handle missing.
    idxs  = [[i for i in rows if stratify[rows[i]] === v] for v in uv]

    # number of occurences of each class and proportions
    nidxs = length.(idxs)
    props = length.(idxs) ./ length(rows)

    n_splits = length(fractions) + 1
    n_rows   = length(rows)

    ns_props = round.(Int, n_rows * fractions * props')
    ns_props = vcat(ns_props, nidxs' .- sum(ns_props, dims=1))

    # warn if anything is >= 1
    if !all(e -> e > 1, ns_props)
        @warn "Some splits have a single or no representative of some class."
    end
    # container for the rows
    split_rows = []
    heads      = ones(Int, length(uv))
    for r in 1:size(ns_props, 1)
        tails   = heads .+ ns_props[r, :] .- 1
        # take chunks of the indices corresponding to the current fraction
        indices = vcat((idxs[i][heads[i]:tails[i]] for i in eachindex(uv))...)
        # rearrange by order of appearance
        indices = sort(indices)
        push!(split_rows, rows[indices])
        heads .= tails .+ 1
    end
    if !all(sl -> sl > 1, length.(split_rows))
        @warn "Some splits have a single or no representative of some class."
    end
    return tuple(split_rows...)
end

const ERR_PARTITION_UNSUPPORTED = ArgumentError(
    "Function `partition` only supports "*
    "AbstractVector, AbstractMatrix or containers implementing the "*
    "Tables interface.")
const ERR_PARTITION_DIMENSION_MISMATCH = DimensionMismatch(
    "Expected a tuple of objects with a common length. ")

__nrows(X) = Tables.istable(X) ? nrows(X) : throw(ERR_PARTITION_UNSUPPORTED)
__nrows(X::Union{AbstractMatrix,AbstractVector}) = nrows(X)

"""
    partition(X, fractions...;
              shuffle=nothing,
              rng=Random.GLOBAL_RNG,
              stratify=nothing,
              multi=false)

Splits the vector, matrix or table `X` into a tuple of objects of the
same type, whose vertical concatenation is `X`. The number of rows in
each component of the return value is determined by the
corresponding `fractions` of `length(nrows(X))`, where valid fractions
are floats between 0 and 1 whose sum is less than one. The last
fraction is not provided, as it is inferred from the preceding ones.

For "synchronized" partitioning of multiple objects, use the
`multi=true` option described below.

    julia> partition(1:1000, 0.8)
    ([1,...,800], [801,...,1000])

    julia> partition(1:1000, 0.2, 0.7)
    ([1,...,200], [201,...,900], [901,...,1000])

    julia> partition(reshape(1:10, 5, 2), 0.2, 0.4)
    ([1 6], [2 7; 3 8], [4 9; 5 10])

    X, y = make_blobs() # a table and vector
    Xtrain, Xtest = partition(X, 0.8, stratify=y)

    (Xtrain, Xtest), (ytrain, ytest) = partition((X, y), 0.8, rng=123, multi=true)


## Keywords

* `shuffle=nothing`: if set to `true`, shuffles the rows before taking
  fractions.

* `rng=Random.GLOBAL_RNG`: specifies the random number generator to be
  used, can be an integer seed. If specified, and `shuffle ===
  nothing` is interpreted as true.

* `stratify=nothing`: if a vector is specified, the partition will
  match the stratification of the given vector. In that case,
  `shuffle` cannot be `false`.

* `multi=false`: if `true` then `X` is expected to be a `tuple` of
  objects sharing a common length, which are each partitioned
  separately using the same specified `fractions` *and* the same row
  shuffling. Returns a tuple of partitions (a tuple of tuples).

"""
function partition(X, fractions::Real...;
                   shuffle::Union{Nothing,Bool}=nothing,
                   rng=Random.GLOBAL_RNG,
                   stratify::Union{Nothing,AbstractVector}=nothing,
                   multi=false)

    # check the fractions
    if !all(e -> 0 < e < 1, fractions) || sum(fractions) >= 1
        throw(DomainError(fractions,
                          "Fractions must be in (0, 1) with sum < 1."))
    end

    # determinen `n_rows`:
    if X isa Tuple && multi
        isempty(X) && return tuple(fill((), length(fractions) + 1)...)
        x = first(X)
        n_rows = __nrows(x)
        all(X[2:end]) do x
            nrows(x) === n_rows
        end || throw(ERR_PARTITION_DIMENSION_MISMATCH)
    else
        n_rows = __nrows(X)
    end

    # check the rng & adjust shuffling
    if rng isa Integer
        rng = MersenneTwister(rng)
    end
    if rng != Random.GLOBAL_RNG && shuffle === nothing
        shuffle = true
    end
    rows = collect(1:n_rows)
    shuffle !== nothing && shuffle && shuffle!(rng, rows)

    # determine the partition of `rows`:
    row_partition = _partition(rows, collect(fractions), stratify)
    return _partition(X, row_partition)
end

function _partition(X, row_partition)
#    _X = Tables.istable(X) ? X : collect(X)
    return tuple((selectrows(X, p) for p in row_partition)...)
end

_partition(X::Tuple, row_partition) =
    map(x->_partition(x, row_partition), X)


# # UNPACK

"""
    unpack(table, f1, f2, ... fk;
           wrap_singles=false,
           shuffle=false,
           rng::Union{AbstractRNG,Int,Nothing}=nothing,
           coerce_options...)

Horizontally split any Tables.jl compatible `table` into smaller
tables or vectors by making column selections determined by the
predicates `f1`, `f2`, ..., `fk`. Selection from the column names is
without replacement. A *predicate* is any object `f` such that
`f(name)` is `true` or `false` for each column `name::Symbol` of
`table`.

Returns a tuple of tables/vectors with length one greater than the
number of supplied predicates, with the last component including all
previously unselected columns.

```
julia> table = DataFrame(x=[1,2], y=['a', 'b'], z=[10.0, 20.0], w=["A", "B"])
2×4 DataFrame
 Row │ x      y     z        w
     │ Int64  Char  Float64  String
─────┼──────────────────────────────
   1 │     1  a        10.0  A
   2 │     2  b        20.0  B

Z, XY, W = unpack(table, ==(:z), !=(:w))
julia> Z
2-element Vector{Float64}:
 10.0
 20.0

julia> XY
2×2 DataFrame
 Row │ x      y
     │ Int64  Char
─────┼─────────────
   1 │     1  a
   2 │     2  b

julia> W  # the column(s) left over
2-element Vector{String}:
 "A"
 "B"
```

Whenever a returned table contains a single column, it is converted to
a vector unless `wrap_singles=true`.

If `coerce_options` are specified then `table` is first replaced
with `coerce(table, coerce_options)`. See
[`ScientificTypes.coerce`](@ref) for details.

If `shuffle=true` then the rows of `table` are first shuffled, using
the global RNG, unless `rng` is specified; if `rng` is an integer, it
specifies the seed of an automatically generated Mersenne twister. If
`rng` is specified then `shuffle=true` is implicit.

"""
function unpack(X, predicates...;
                wrap_singles=false,
                shuffle=nothing,
                rng=nothing, pairs...)

    # add a final predicate to unpack all remaining columns into to
    # the last return value:
    predicates = (predicates..., _ -> true)

    shuffle, rng = shuffle_and_rng(shuffle, rng)

    shuffle && (X = selectrows(X, Random.shuffle(rng, 1:nrows(X))))

    if isempty(pairs)
        Xfixed = X
    else
        Xfixed = coerce(X, pairs...)
    end

    unpacked = Any[]
    names_left = schema(Xfixed).names |> collect

    for c in predicates
        names = filter(c, names_left)
        filter!(!in(names), names_left)
        length(names) == 1 && !wrap_singles && (names = names[1])
        push!(unpacked, selectcols(Xfixed, names))
    end
    return Tuple(unpacked)
end


## RESTRICTING TO A FOLD

struct FoldRestrictor{i,N}
    f::NTuple{N,Vector{Int}}
end
(r::FoldRestrictor{i})(X) where i = selectrows(X, (r.f)[i])

"""
    restrict(X, folds, i)

The restriction of `X`, a vector, matrix or table, to the `i`th fold
of `folds`, where `folds` is a tuple of vectors of row indices.

The method is curried, so that `restrict(folds, i)` is the operator
on data defined by `restrict(folds, i)(X) = restrict(X, folds, i)`.

### Example

    folds = ([1, 2], [3, 4, 5],  [6,])
    restrict([:x1, :x2, :x3, :x4, :x5, :x6], folds, 2) # [:x3, :x4, :x5]

See also [`corestrict`](@ref)

"""
restrict(f::NTuple{N}, i) where N = FoldRestrictor{i,N}(f)
restrict(X, f, i) = restrict(f, i)(X)


## RESTRICTING TO A FOLD COMPLEMENT


"""
    complement(folds, i)

The complement of the `i`th fold of `folds` in the concatenation of
all elements of `folds`. Here `folds` is a vector or tuple of integer
vectors, typically representing row indices or a vector, matrix or
table.

    complement(([1,2], [3,], [4, 5]), 2) # [1 ,2, 4, 5]

"""
complement(f, i) = reduce(vcat, collect(f)[Not(i)])

struct FoldComplementRestrictor{i,N}
    f::NTuple{N,Vector{Int}}
end
(r::FoldComplementRestrictor{i})(X) where i =
    selectrows(X, complement(r.f, i))

"""
    corestrict(X, folds, i)

The restriction of `X`, a vector, matrix or table, to the *complement*
of the `i`th fold of `folds`, where `folds` is a tuple of vectors of
row indices.

The method is curried, so that `corestrict(folds, i)` is the operator
on data defined by `corestrict(folds, i)(X) = corestrict(X, folds, i)`.

### Example

    folds = ([1, 2], [3, 4, 5],  [6,])
    corestrict([:x1, :x2, :x3, :x4, :x5, :x6], folds, 2) # [:x1, :x2, :x6]

"""
corestrict(f::NTuple{N}, i) where N = FoldComplementRestrictor{i,N}(f)
corestrict(X, f, i) = corestrict(f, i)(X)

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


## TRANSFORMING BETWEEN CATEGORICAL ELEMENTS AND RAW VALUES

MLJModelInterface.transform(
    e::Union{CategoricalArray,CategoricalValue,CategoricalPool},
    arg) = CategoricalDistributions.transform(e, arg)


## SKIPPING MISSING AND NAN: skipinvalid

_isnan(x) = false
_isnan(x::Number) = isnan(x)

skipnan(x) = Iterators.filter(!_isnan, x)

"""
    skipinvalid(itr)

Return an iterator over the elements in `itr` skipping `missing` and
`NaN` values. Behaviour is similar to [`skipmissing`](@ref).

    skipinvalid(A, B)

For vectors `A` and `B` of the same length, return a tuple of vectors
`(A[mask], B[mask])` where `mask[i]` is `true` if and only if `A[i]`
and `B[i]` are both valid (non-`missing` and non-`NaN`). Can also
called on other iterators of matching length, such as arrays, but
always returns a vector. Does not remove `Missing` from the element
types if present in the original iterators.

"""
skipinvalid(v) = v |> skipmissing |> skipnan

isinvalid(x) = ismissing(x) || _isnan(x)

function skipinvalid(yhat, y)
    mask = .!(isinvalid.(yhat) .| isinvalid.(y))
    return yhat[mask], y[mask]
end

# TODO: refactor balanced accuracy to get rid of these:

function _skipinvalid(yhat, y, w::Arr)
    mask = .!(isinvalid.(yhat) .| isinvalid.(y))
    return yhat[mask], y[mask], w[mask]
end

function _skipinvalid(yhat, y, w::Union{Nothing,AbstractDict})
    mask = .!(isinvalid.(yhat) .| isinvalid.(y))
    return yhat[mask], y[mask], w
end
