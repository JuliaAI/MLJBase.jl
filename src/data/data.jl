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

"""
    partition(X, fractions...;
              shuffle=nothing, rng=Random.GLOBAL_RNG, stratify=nothing)

Splits the vector or matrix `X` into a tuple of vectors or matrices
whose vertical concatentation is `X`. The number of rows in each
componenent of the return value is determined by the corresponding
`fractions` of `length(nrows(X))`, where valid fractions are in (0,1)
and sum up to less than one. The last fraction is not provided, as it
is inferred from the preceding ones.

`X` can also be any object which implements the `Tables.jl` interface
according to `Tables.istable`.

So, for example,

    julia> partition(1:1000, 0.8)
    ([1,...,800], [801,...,1000])

    julia> partition(1:1000, 0.2, 0.7)
    ([1,...,200], [201,...,900], [901,...,1000])

    julia> partition(reshape(1:10, 5, 2), 0.2, 0.4)
    ([1 6], [2 7; 3 8], [4 9; 5 10])

## Keywords

* `shuffle=nothing`:        if set to  `true`, shuffles the rows before taking fractions.
* `rng=Random.GLOBAL_RNG`:  specifies the random number generator to be used, can be an integer
                            seed. If specified, and `shuffle === nothing` is interpreted as true.
* `stratify=nothing`:       if a vector is specified, the partition will match the stratification
                            of the given vector. In that case, `shuffle` cannot be `false`.
"""
function partition(X, fractions...; kwargs...)
    if X isa AbstractMatrix || Tables.istable(X)
        # Generic method for all matrices and tables.  Partition its rows and
        # apply `selectrows` to each partition.
        return tuple((selectrows(X, p) for p in partition(1:nrows(X), fractions...; kwargs...))...)
    else
        throw(ArgumentError("Function `partition` only supports AbstractVector, " *
                            "AbstractMatrix or containers implementing the Tables interface."))
    end
end

function partition(rows::AbstractVector, fractions::Real...;
                   shuffle::Union{Nothing,Bool}=nothing, rng=Random.GLOBAL_RNG,
                   stratify::Union{Nothing,AbstractVector}=nothing)
    # if rows is a unitrange, collect it
    rows = collect(rows)
    # check the fractions
    if !all(e -> 0 < e < 1, fractions) || sum(fractions) >= 1
        throw(DomainError(fractions, "Fractions must be in (0, 1) with sum < 1."))
    end
    # check the rng & adjust shuffling
    if rng isa Integer
        rng = MersenneTwister(rng)
    end
    if rng != Random.GLOBAL_RNG && shuffle === nothing
        shuffle = true
    end
    shuffle !== nothing && shuffle && shuffle!(rng, rows)
    return _partition(rows, collect(fractions), stratify)
end

"""

    t1, t2, ...., tk = unpack(table, f1, f2, ... fk;
                               wrap_singles=false,
                               shuffle=false,
                               rng::Union{AbstractRNG,Int,Nothing}=nothing)



Split any Tables.jl compatible `table` into smaller tables (or
vectors) `t1, t2, ..., tk` by making selections *without replacement*
from the column names defined by the filters `f1`, `f2`, ...,
`fk`. A *filter* is any object `f` such that `f(name)` is `true`
or `false` for each column `name::Symbol` of `table`.

Whenever a returned table contains a single column, it is converted to
a vector unless `wrap_singles=true`.

Scientific type conversions can be optionally specified (note
semicolon):

    unpack(table, t...; wrap_singles=false, col1=>scitype1, col2=>scitype2, ... )

If `shuffle=true` then the rows of `table` are first shuffled, using
the global RNG, unless `rng` is specified; if `rng` is an integer, it
specifies the seed of an automatically generated Mersenne twister. If
`rng` is specified then `shuffle=true` is implicit.

### Example

```
julia> table = DataFrame(x=[1,2], y=['a', 'b'], z=[10.0, 20.0], w=["A", "B"])
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
function unpack(X, tests...;
                wrap_singles=false,
                shuffle=nothing,
                rng=nothing, pairs...)

    shuffle, rng = shuffle_and_rng(shuffle, rng)

    shuffle && (X = selectrows(X, Random.shuffle(rng, 1:nrows(X))))

    if isempty(pairs)
        Xfixed = X
    else
        Xfixed = coerce(X, pairs...)
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

_err_missing_class(c) =  throw(DomainError(
    "Value `$c` not in pool"))


function transform_(pool, x)
    ismissing(x) && return missing
    x in levels(pool) || _err_missing_class(x)
    return pool[get(pool, x)]
end

transform_(pool, X::AbstractArray) = broadcast(x -> transform_(pool, x), X)

"""
    transform(e::Union{CategoricalElement,CategoricalArray,CategoricalPool},  X)

Transform the specified object `X` into a categorical version, using
the pool contained in `e`. Here `X` is a raw value (an element of
`levels(e)`) or an `AbstractArray` of such values.

```julia
v = categorical(["x", "y", "y", "x", "x"])
julia> transform(v, "x")
CategoricalValue{String,UInt32} "x"

julia> transform(v[1], ["x" "x"; missing "y"])
2×2 CategoricalArray{Union{Missing, Symbol},2,UInt32}:
 "x"       "x"
 missing   "y"

"""
MLJModelInterface.transform(e::Union{CategoricalArray, CategoricalValue},
                            arg) = transform_(CategoricalArrays.pool(e), arg)
MLJModelInterface.transform(e::CategoricalPool, arg) =
    transform_(e, arg)
