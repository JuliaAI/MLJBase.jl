abstract type ResamplingStrategy <: MLJType end
show_as_constructed(::Type{<:ResamplingStrategy}) = true

# resampling strategies are `==` if they have the same type and their
# field values are `==`:
function ==(s1::S, s2::S) where S <: ResamplingStrategy
    return all(getfield(s1, fld) == getfield(s2, fld) for fld in fieldnames(S))
end

# fallbacks for method to be implemented by each new strategy:
train_test_pairs(s::ResamplingStrategy, rows, X, y, w) =
    train_test_pairs(s, rows, X, y)
train_test_pairs(s::ResamplingStrategy, rows, X, y) =
    train_test_pairs(s, rows, y)
train_test_pairs(s::ResamplingStrategy, rows, y) =
    train_test_pairs(s, rows)

# Helper to interpret rng, shuffle in case either is `nothing` or if
# `rng` is an integer:
function shuffle_and_rng(shuffle, rng)
    if rng isa Integer
        rng = MersenneTwister(rng)
    end

    if shuffle === nothing
        shuffle = ifelse(rng===nothing, false, true)
    end

    if rng === nothing
        rng = Random.GLOBAL_RNG
    end

    return shuffle, rng
end

# ----------------------------------------------------------------
# InSample

"""
    in_sample = InSample()

Instantiate an `InSample` resampling strategy, for use in `evaluate!`, `evaluate` and in
tuning. In this strategy the train and test sets are the same, and consist of all
observations specified by the `rows` keyword argument. If `rows` is not specified, all
supplied rows are used.

# Example

```julia
using MLJBase, MLJModels

X, y = make_blobs()  # a table and a vector
model = ConstantClassifier()
train, test = partition(eachindex(y), 0.7)  # train:test = 70:30
```

Compute in-sample (training) loss:

```julia
evaluate(model, X, y, resampling=InSample(), rows=train, measure=brier_loss)
```

Compute the out-of-sample loss:

```julia
evaluate(model, X, y, resampling=[(train, test),], measure=brier_loss)
```

Or equivalently:

```julia
evaluate(model, X, y, resampling=Holdout(fraction_train=0.7), measure=brier_loss)
```

"""
struct InSample <: ResamplingStrategy end

train_test_pairs(::InSample, rows) = [(rows, rows),]

# ----------------------------------------------------------------
# Holdout

"""
    holdout = Holdout(; fraction_train=0.7, shuffle=nothing, rng=nothing)

Instantiate a `Holdout` resampling strategy, for use in `evaluate!`, `evaluate` and in
tuning.

```julia
train_test_pairs(holdout, rows)
```

Returns the pair `[(train, test)]`, where `train` and `test` are
vectors such that `rows=vcat(train, test)` and
`length(train)/length(rows)` is approximatey equal to fraction_train`.

Pre-shuffling of `rows` is controlled by `rng` and `shuffle`. If `rng`
is an integer, then the `Holdout` keyword constructor resets it to
`MersenneTwister(rng)`. Otherwise some `AbstractRNG` object is
expected.

If `rng` is left unspecified, `rng` is reset to `Random.GLOBAL_RNG`,
in which case rows are only pre-shuffled if `shuffle=true` is
specified.

"""
struct Holdout <: ResamplingStrategy
    fraction_train::Float64
    shuffle::Bool
    rng::Union{Int,AbstractRNG}

    function Holdout(fraction_train, shuffle, rng)
        0 < fraction_train < 1 ||
            error("`fraction_train` must be between 0 and 1.")
        return new(fraction_train, shuffle, rng)
    end
end

# Keyword Constructor:
Holdout(; fraction_train::Float64=0.7, shuffle=nothing, rng=nothing) =
    Holdout(fraction_train, shuffle_and_rng(shuffle, rng)...)

function train_test_pairs(holdout::Holdout, rows)

    train, test = partition(rows, holdout.fraction_train,
                            shuffle=holdout.shuffle, rng=holdout.rng)
    return [(train, test),]

end

# ----------------------------------------------------------------
# Cross-validation (vanilla)

"""
    cv = CV(; nfolds=6,  shuffle=nothing, rng=nothing)

Cross-validation resampling strategy, for use in `evaluate!`,
`evaluate` and tuning.

```julia
train_test_pairs(cv, rows)
```

Returns an `nfolds`-length iterator of `(train, test)` pairs of
vectors (row indices), where each `train` and `test` is a sub-vector
of `rows`. The `test` vectors are mutually exclusive and exhaust
`rows`. Each `train` vector is the complement of the corresponding
`test` vector. With no row pre-shuffling, the order of `rows` is
preserved, in the sense that `rows` coincides precisely with the
concatenation of the `test` vectors, in the order they are
generated. The first `r` test vectors have length `n + 1`, where `n, r
= divrem(length(rows), nfolds)`, and the remaining test vectors have
length `n`.

Pre-shuffling of `rows` is controlled by `rng` and `shuffle`. If `rng`
is an integer, then the `CV` keyword constructor resets it to
`MersenneTwister(rng)`. Otherwise some `AbstractRNG` object is
expected.

If `rng` is left unspecified, `rng` is reset to `Random.GLOBAL_RNG`,
in which case rows are only pre-shuffled if `shuffle=true` is
explicitly specified.

"""
struct CV <: ResamplingStrategy
    nfolds::Int
    shuffle::Bool
    rng::Union{Int,AbstractRNG}
    function CV(nfolds, shuffle, rng)
        nfolds > 1 || throw(ArgumentError("Must have nfolds > 1. "))
        return new(nfolds, shuffle, rng)
    end
end

# Constructor with keywords
CV(; nfolds::Int=6,  shuffle=nothing, rng=nothing) =
    CV(nfolds, shuffle_and_rng(shuffle, rng)...)

function train_test_pairs(cv::CV, rows)

    n_obs = length(rows)
    n_folds = cv.nfolds

    if cv.shuffle
        rows=shuffle!(cv.rng, collect(rows))
    end

    n, r = divrem(n_obs, n_folds)

    if n < 1
        throw(ArgumentError(
            """Insufficient data for $n_folds-fold cross-validation.
            Try reducing nfolds. """
        ))
    end

    m = n + 1 # number of observations in first r folds

    itr1 = Iterators.partition( 1 : m*r , m)
    itr2 = Iterators.partition( m*r+1 : n_obs , n)
    test_folds = Iterators.flatten((itr1, itr2))

    return map(test_folds) do test_indices
        test_rows = rows[test_indices]

        train_rows = vcat(
            rows[ 1 : first(test_indices)-1 ],
            rows[ last(test_indices)+1 : end ]
        )

        (train_rows, test_rows)
    end
end


# ----------------------------------------------------------------
# Cross-validation (TimeSeriesCV)
"""
    tscv = TimeSeriesCV(; nfolds=4)

Cross-validation resampling strategy, for use in `evaluate!`,
`evaluate` and tuning, when observations are chronological and not
expected to be independent.

```julia
train_test_pairs(tscv, rows)
```

Returns an `nfolds`-length iterator of `(train, test)` pairs of
vectors (row indices), where each `train` and `test` is a sub-vector
of `rows`. The rows are partitioned sequentially into `nfolds + 1`
approximately equal length partitions, where the first partition is the first
train set, and the second partition is the first test set. The second
train set consists of the first two partitions, and the second test set
consists of the third partition, and so on for each fold.

The first partition (which is the first train set) has length `n + r`,
where `n, r = divrem(length(rows), nfolds + 1)`, and the remaining partitions
(all of the test folds) have length `n`.

# Examples

```julia-repl
julia> MLJBase.train_test_pairs(TimeSeriesCV(nfolds=3), 1:10)
3-element Vector{Tuple{UnitRange{Int64}, UnitRange{Int64}}}:
 (1:4, 5:6)
 (1:6, 7:8)
 (1:8, 9:10)

julia> model = (@load RidgeRegressor pkg=MultivariateStats verbosity=0)();

julia> data = @load_sunspots;

julia> X = (lag1 = data.sunspot_number[2:end-1],
            lag2 = data.sunspot_number[1:end-2]);

julia> y = data.sunspot_number[3:end];

julia> tscv = TimeSeriesCV(nfolds=3);

julia> evaluate(model, X, y, resampling=tscv, measure=rmse, verbosity=0)
┌───────────────────────────┬───────────────┬────────────────────┐
│ _.measure                 │ _.measurement │ _.per_fold         │
├───────────────────────────┼───────────────┼────────────────────┤
│ RootMeanSquaredError @753 │ 21.7          │ [25.4, 16.3, 22.4] │
└───────────────────────────┴───────────────┴────────────────────┘
_.per_observation = [missing]
_.fitted_params_per_fold = [ … ]
_.report_per_fold = [ … ]
_.train_test_rows = [ … ]
```
"""
struct TimeSeriesCV <: ResamplingStrategy
    nfolds::Int
    function TimeSeriesCV(nfolds)
        nfolds > 0 || throw(ArgumentError("Must have nfolds > 0. "))
        return new(nfolds)
    end
end

# Constructor with keywords
TimeSeriesCV(; nfolds::Int=4) = TimeSeriesCV(nfolds)

function train_test_pairs(tscv::TimeSeriesCV, rows)
    if rows != sort(rows)
        @warn "TimeSeriesCV is being applied to `rows` not in sequence. "
    end

    n_obs = length(rows)
    n_folds = tscv.nfolds

    m, r = divrem(n_obs, n_folds + 1)

    if m < 1
        throw(ArgumentError(
            "Insufficient data for $n_folds-fold " *
            "time-series cross-validation.\n" *
            "Try reducing nfolds. "
        ))
    end

    test_folds = Iterators.partition( m+r+1 : n_obs , m)

    return map(test_folds) do test_indices
        train_indices = 1 : first(test_indices)-1
        rows[train_indices], rows[test_indices]
    end
end

# ----------------------------------------------------------------
# Cross-validation (stratified; for `Finite` targets)

"""
    stratified_cv = StratifiedCV(; nfolds=6,
                                   shuffle=false,
                                   rng=Random.GLOBAL_RNG)

Stratified cross-validation resampling strategy, for use in
`evaluate!`, `evaluate` and in tuning. Applies only to classification
problems (`OrderedFactor` or `Multiclass` targets).

```julia
train_test_pairs(stratified_cv, rows, y)
```
Returns an `nfolds`-length iterator of `(train, test)` pairs of
vectors (row indices) where each `train` and `test` is a sub-vector of
`rows`. The `test` vectors are mutually exclusive and exhaust
`rows`. Each `train` vector is the complement of the corresponding
`test` vector.

Unlike regular cross-validation, the distribution of the levels of the
target `y` corresponding to each `train` and `test` is constrained, as
far as possible, to replicate that of `y[rows]` as a whole.

The stratified `train_test_pairs` algorithm is invariant to label renaming.
For example, if you run `replace!(y, 'a' => 'b', 'b' => 'a')` and then re-run
`train_test_pairs`, the returned `(train, test)` pairs will be the same.

Pre-shuffling of `rows` is controlled by `rng` and `shuffle`. If `rng`
is an integer, then the `StratifedCV` keywod constructor resets it to
`MersenneTwister(rng)`. Otherwise some `AbstractRNG` object is
expected.

If `rng` is left unspecified, `rng` is reset to `Random.GLOBAL_RNG`,
in which case rows are only pre-shuffled if `shuffle=true` is
explicitly specified.

"""
struct StratifiedCV <: ResamplingStrategy
    nfolds::Int
    shuffle::Bool
    rng::Union{Int,AbstractRNG}
    function StratifiedCV(nfolds, shuffle, rng)
        nfolds > 1 || throw(ArgumentError("Must have nfolds > 1. "))
        return new(nfolds, shuffle, rng)
    end
end

# Constructor with keywords
StratifiedCV(; nfolds::Int=6,  shuffle=nothing, rng=nothing) =
       StratifiedCV(nfolds, shuffle_and_rng(shuffle, rng)...)

# Description of the stratified CV algorithm:
#
# There are algorithms that are conceptually somewhat simpler than this
# algorithm, but this algorithm is O(n) and is invariant to relabelling
# of the target vector.
#
# 1) Use countmap() to get the count for each level.
#
# 2) Use unique() to get the order in which the levels appear. (Steps 1
# and 2 could be combined if countmap() used an OrderedDict.)
#
# 3) For y = ['b', 'c', 'a', 'b', 'b', 'b', 'c', 'c', 'c', 'a', 'a', 'a'],
# the levels occur in the order ['b', 'c', 'a'], and each level has a count
# of 4. So imagine a table like this:
#
# b b b b c c c c a a a a
# 1 2 3 1 2 3 1 2 3 1 2 3
#
# This table ensures that the levels are smoothly spread across the test folds.
# In other words, where one level leaves off, the next level picks up. So,
# for example, as the 'c' levels are encountered, the corresponding row indices
# are added to folds [2, 3, 1, 2], in that order. The table above is
# partitioned by y-level and put into a dictionary `fold_lookup` that maps
# levels to the corresponding array of fold indices.
#
# 4) Iterate i from 1 to length(rows). For each i, look up the corresponding
# level, i.e. `level = y[rows[i]]`. Then use `popfirst!(fold_lookup[level])`
# to find the test fold in which to put the i-th element of `rows`.
#
# 5) Concatenate the appropriate test folds together to get the train
# indices for each `(train, test)` pair.

function train_test_pairs(stratified_cv::StratifiedCV, rows, y)

    st = scitype(y)

    if stratified_cv.shuffle
        rows=shuffle!(stratified_cv.rng, collect(rows))
    end

    n_folds = stratified_cv.nfolds
    n_obs = length(rows)
    obs_per_fold = div(n_obs, n_folds)

    y_included = y[rows]
    level_count_dict = countmap(y_included)

    # unique() preserves the order of appearance of the levels.
    # We need this so that the results are invariant to renaming of the levels.
    y_levels = unique(y_included)
    level_count = [level_count_dict[level] for level in y_levels]

    fold_cycle = collect(Iterators.take(Iterators.cycle(1:n_folds), n_obs))

    lasts = cumsum(level_count)
    firsts = [1; lasts[1:end-1] .+ 1]

    level_fold_indices = (fold_cycle[f:l] for (f, l) in zip(firsts, lasts))
    fold_lookup = Dict(y_levels .=> level_fold_indices)

    folds = [Int[] for _ in 1:n_folds]
    for fold in folds
        sizehint!(fold, obs_per_fold)
    end

    for i in 1:n_obs
        level = y_included[i]
        fold_index = popfirst!(fold_lookup[level])
        push!(folds[fold_index], rows[i])
    end

    [(complement(folds, i), folds[i]) for i in 1:n_folds]
end

