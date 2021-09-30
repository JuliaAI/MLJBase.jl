const PREDICT_OPERATIONS_STRING = begin
    strings = map(PREDICT_OPERATIONS) do op
        "`"*string(op)*"`"
    end
    join(strings, ", ", ", or ")
end
const ERR_WEIGHTS_REAL =
    ArgumentError("`weights` must be a `Real` vector. ")
const ERR_WEIGHTS_LENGTH =
    DimensionMismatch("`weights` and target "*
                      "have different lengths. ")
const ERR_WEIGHTS_DICT =
    ArgumentError("`class_weights` must be a "*
                  "dictionary with `Real` values. ")
const ERR_WEIGHTS_CLASSES =
    DimensionMismatch("The keys of `class_weights` "*
                      "are not the same as the levels of the "*
                      "target, `y`. Do `levels(y)` to check levels. ")
const ERR_OPERATION_MEASURE_MISMATCH = DimensionMismatch(
    "The number of operations and the number of measures are different. ")
const ERR_INVALID_OPERATION = ArgumentError(
    "Invalid `operation` or `operations`. "*
    "An operation must be one of these: $PREDICT_OPERATIONS_STRING. ")
_ambiguous_operation(model, measure) =
    "`prediction_type($measure) == $(prediction_type(measure))` but "*
    "`prediction_type($model) == $(prediction_type(model))`."
err_ambiguous_operation(model, measure) = ArgumentError(
    _ambiguous_operation(model, measure)*
    "\nUnable to deduce an appropriate operation for $measure. "*
    "Explicitly specify `operation=...` or `operations=...`. ")
err_incompatible_prediction_types(model, measure) = ArgumentError(
    _ambiguous_operation(model, measure))

# ==================================================================
## MODEL TYPES THAT CAN BE EVALUATED

# not exported:
const Measurable = Union{Supervised,
                         ProbabilisticSupervisedDetector,
                         ProbabilisticUnsupervisedDetector,
                         DeterministicSupervisedDetector,
                         DeterministicUnsupervisedDetector}


# ==================================================================
## RESAMPLING STRATEGIES

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
# Holdout

"""
    holdout = Holdout(; fraction_train=0.7,
                         shuffle=nothing,
                         rng=nothing)

Holdout resampling strategy, for use in `evaluate!`, `evaluate` and in
tuning.

    train_test_pairs(holdout, rows)

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

    train_test_pairs(cv, rows)

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
            """Inusufficient data for $n_folds-fold cross-validation.
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

train_test_pairs(tscv, rows)

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
            "Inusufficient data for $n_folds-fold " *
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

    train_test_pairs(stratified_cv, rows, y)

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
is an integer, then the `StratifedCV` keyword constructor resets it to
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
    if !(st <: AbstractArray{<:Finite})
        throw(ArgumentError(
            "Supplied target has scitype $st but stratified " *
            "cross-validation applies only to classification problems. "
        ))
    end

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

# ================================================================
## EVALUATION RESULT TYPE

const PerformanceEvaluation = NamedTuple{(:measure,
                                          :measurement,
                                          :operation,
                                          :per_fold,
                                          :per_observation,
                                          :fitted_params_per_fold,
                                          :report_per_fold,
                                          :train_test_rows)}

# pretty printing:
round3(x) = x
round3(x::AbstractFloat) = round(x, sigdigits=3)

_short(v) = v
_short(v::Vector{<:Real}) = MLJBase.short_string(v)
_short(v::Vector) = string("[", join(_short.(v), ", "), "]")
_short(::Missing) = missing

function Base.show(io::IO, ::MIME"text/plain", e::PerformanceEvaluation)
    data = hcat(e.measure, round3.(e.measurement), e.operation,
                [round3.(v) for v in e.per_fold])
    header = ["measure", "measurement", "operation", "per_fold"]
    println(io, "PerformanceEvaluation object "*
            "with these fields:")
    println(io, "  measure, measurement, operation, per_fold,\n"*
            "  per_observation, fitted_params_per_fold,\n"*
            "  report_per_fold, train_test_pairs")
    println(io, "Extract:")
    show_color = MLJBase.SHOW_COLOR
    color_off()
    PrettyTables.pretty_table(io, data, header;
                              header_crayon=PrettyTables.Crayon(bold=false),
                              alignment=:l)
    show_color ? color_on() : color_off()
end

function Base.show(io::IO, e::PerformanceEvaluation)
    summary = Tuple(round3.(e.measurement))
    print(io, "PerformanceEvaluation$summary")
end

# ===============================================================
## EVALUATION METHODS

# ---------------------------------------------------------------
# Helpers

function actual_rows(rows, N, verbosity)
    unspecified_rows = (rows === nothing)
    _rows = unspecified_rows ? (1:N) : rows
    if !unspecified_rows && verbosity > 0
        @info "Creating subsamples from a subset of all rows. "
    end
    return _rows
end

function _check_measure(measure, operation, model, y)

    T = scitype(y)

    T == Unknown && (return true)
    target_scitype(measure) == Unknown && (return true)
    prediction_type(measure) == :unknown && (return true)

    avoid = "\nTo override measure checks, set check_measure=false. "

    T <: target_scitype(measure) ||
        throw(ArgumentError(
            "\nscitype of target = $T but target_scitype($measure) = "*
            "$(target_scitype(measure))."*avoid))

    incompatible = model isa Probabilistic &&
        operation == predict &&
        prediction_type(measure) != :probabilistic

    if incompatible
        if target_scitype(measure) <:
            AbstractVector{<:Union{Missing,Finite}}
            suggestion = "\nPerhaps you want to set `operation="*
                "predict_mode` or need to "*
                "specify multiple operations, "*
                "one for each measure. "
        elseif target_scitype(measure) <:
            AbstractVector{<:Union{Missing,Continuous}}
            suggestion = "\nPerhaps you want to set `operation="*
                "predict_mean` or `operation=predict_median`, or "*
                "specify multiple operations, "*
                "one for each measure. "
        else
            suggestion = ""
        end
        throw(ArgumentError(
            "\n$model <: Probabilistic but prediction_type($measure) = "*
            ":$(prediction_type(measure)). "*suggestion*avoid))
    end

    model isa Deterministic && prediction_type(measure) != :deterministic &&
        throw(ArgumentError("$model <: Deterministic but "*
                            "prediction_type($measure) ="*
              ":$(prediction_type(measure))."*avoid))

    return true

end

_check_measures(measures, operations, model, y) = begin
    all(eachindex(measures)) do j
        _check_measure(measures[j], operations[j], model, y)
    end
end

function _actual_measures(measures, model)

    if measures === nothing
        candidate = default_measure(model)
        candidate ===  nothing && error("You need to specify measure=... ")
        _measures = [candidate, ]
    elseif !(measures isa AbstractVector)
        _measures = [measures, ]
    else
        _measures = measures
    end

    return _measures

end

function _check_weights(weights, nrows)
    weights isa AbstractVector{<:Real} ||
        throw(ERR_WEIGHTS_REAL)
    length(weights) == nrows ||
        throw(ERR_WEIGHTS_LENGTH)
    return true
end

function _check_class_weights(weights, levels)
    weights isa AbstractDict{<:Any,<:Real} ||
        throw(ERR_WEIGHTS_DICT)
    Set(levels) == Set(keys(weights)) ||
        throw(ERR_WEIGHTS_CLASSES)
    return true
end

function _check_weights_measures(weights,
                                 class_weights,
                                 measures,
                                 mach,
                                 operations,
                                 verbosity,
                                 check_measure)

    if check_measure || !(weights isa Nothing) || !(class_weights isa Nothing)
        y = mach.args[2]()
    end

    check_measure && _check_measures(measures, operations, mach.model, y)

    weights isa Nothing || _check_weights(weights, nrows(y))

    class_weights isa Nothing ||
        _check_class_weights(class_weights, levels(y))

end

# here `operation` is what the user has specified, and `nothing` if
# not specified:
_actual_operations(operation, measures, args...) =
    _actual_operations(fill(operation, length(measures)), measures, args...)
function _actual_operations(operation::AbstractVector, measures, args...)
    length(measures) === length(operation) ||
        throw(ERR_OPERATION_MEASURE_MISMATCH)
    all(operation) do op
        op in eval.(PREDICT_OPERATIONS)
    end || throw(ERR_INVALID_OPERATION)
    return operation
end
function _actual_operations(operation::Nothing,
                            measures,  # vector of measures
                            model,
                            verbosity)
    map(measures) do m

        prediction_type = MLJBase.prediction_type(m)
        target_scitype = MLJBase.target_scitype(m)

        if prediction_type === :unknown
            return predict
        end

        if MLJBase.prediction_type(model) === :probabilistic
            if prediction_type === :probabilistic
                return predict
            elseif prediction_type === :deterministic
                if target_scitype <: AbstractArray{<:Union{Missing,Finite}}
                    return predict_mode
                elseif target_scitype <:
                    AbstractArray{<:Union{Missing,Continuous,Count}}
                    return predict_mean
                else
                    throw(err_ambiguous_operation(model, m))
                end
            else
                throw(err_ambiguous_operation(model, m))
            end
        elseif MLJBase.prediction_type(model) === :deterministic
            if prediction_type === :probabilistic
                throw(err_incompatible_prediction_types(model, m))
            elseif prediction_type === :deterministic
                return predict
            else
                throw(err_ambiguous_operation(model, m))
            end
        else
            if prediction_type === :interval
                return predict
            else
                throw(err_ambiguous_operation(model, m))
            end
        end
    end
end

function _warn_about_unsupported(trait, str, measures, weights, verbosity)
    if verbosity >= 0 && weights !== nothing
        unsupported = filter(measures) do m
            !trait(m)
        end
        if !isempty(unsupported)
            unsupported_as_string = string(unsupported[1])
            unsupported_as_string *=
                reduce(*, [string(", ", m) for m in unsupported[2:end]])
                @warn "$str weights ignored in evaluations of the following"*
            " measures, as unsupported: \n$unsupported_as_string "
        end
    end
end

function _process_accel_settings(accel::CPUThreads)
    if accel.settings === nothing
        nthreads = Threads.nthreads()
        _accel =  CPUThreads(nthreads)
    else
      typeof(accel.settings) <: Signed ||
      throw(ArgumentError("`n`used in `acceleration = CPUThreads(n)`must" *
                        "be an instance of type `T<:Signed`"))
      accel.settings > 0 ||
            throw(error("Can't create $(accel.settings) tasks)"))
      _accel = accel
    end
    return _accel
end

_process_accel_settings(accel::Union{CPU1,CPUProcesses}) = accel

#fallback
_process_accel_settings(accel) =  throw(ArgumentError("unsupported" *
                            " acceleration parameter`acceleration = $accel` "))

# --------------------------------------------------------------
# User interface points: `evaluate!` and `evaluate`

"""
    evaluate!(mach,
              resampling=CV(),
              measure=nothing,
              rows=nothing,
              weights=nothing,
              class_weights=nothing,
              operation=nothing,
              repeats=1,
              acceleration=default_resource(),
              force=false,
              verbosity=1,
              check_measure=true)

Estimate the performance of a machine `mach` wrapping a supervised
model in data, using the specified `resampling` strategy (defaulting
to 6-fold cross-validation) and `measure`, which can be a single
measure or vector.

Do `subtypes(MLJ.ResamplingStrategy)` to obtain a list of available
resampling strategies. If `resampling` is not an object of type
`MLJ.ResamplingStrategy`, then a vector of pairs (of the form
`(train_rows, test_rows)` is expected. For example, setting

    resampling = [(1:100), (101:200)),
                   (101:200), (1:100)]

gives two-fold cross-validation using the first 200 rows of data.

The type of operation (`predict`, `predict_mode`, etc) to be
associated with `measure` is automatically inferred from measure
traits where possible. For example, `predict_mode` will be used for a
`Multiclass` target, if `model` is probabilistic but `measure` is
deterministic. The operations applied can be inspected from the
`operation` field of the object returned. Alternatively, operations
can be explicitly specified using `operation=...`. If `measure` is a
vector, then `operation` must be a single operation, which will be
associated with all measures, or a vector of the same length as
`measure`.

The resampling strategy is applied repeatedly (Monte Carlo resampling)
if `repeats > 1`. For example, if `repeats = 10`, then `resampling =
CV(nfolds=5, shuffle=true)`, generates a total of 50 `(train, test)`
pairs for evaluation and subsequent aggregation.

If `resampling isa MLJ.ResamplingStrategy` then one may optionally
restrict the data used in evaluation by specifying `rows`.

An optional `weights` vector may be passed for measures that support
sample weights (`MLJ.supports_weights(measure) == true`), which is
ignored by those that don't. These weights are not to be confused with
any weights `w` bound to `mach` (as in `mach = machine(model, X,
y, w)`). To pass these to the performance evaluation measures you must
explictly specify `weights=w` in the `evaluate!` call.

Additionally, optional `class_weights` dictionary may be passed
for measures that support class weights
(`MLJ.supports_class_weights(measure) == true`), which is
ignored by those that don't. These weights are not to be confused with
any weights `class_w` bound to `mach` (as in `mach = machine(model, X,
y, class_w)`). To pass these to the performance evaluation measures you
must explictly specify `class_weights=w` in the `evaluate!` call.

User-defined measures are supported; see the manual for details.

If no measure is specified, then `default_measure(mach.model)` is
used, unless this default is `nothing` and an error is thrown.

The `acceleration` keyword argument is used to specify the compute resource (a
subtype of `ComputationalResources.AbstractResource`) that will be used to
accelerate/parallelize the resampling operation.

Although evaluate! is mutating, `mach.model` and `mach.args` are
untouched.

### Summary of key-word arguments

- `resampling` - resampling strategy (default is `CV(nfolds=6)`)

- `measure`/`measures` - measure or vector of measures (losses, scores, etc)

- `rows` - vector of observation indices from which both train and
  test folds are constructed (default is all observations)

- `weights` - per-sample weights for measures that support them (not
  to be confused with weights used in training)

- `class_weights` - dictionary of per-class weights for use with
  measures that support these, in classification problems (not to be
  confused with per-sample `weights` or with class weights used in
  training)

- `operation`/`operations` - One of $PREDICT_OPERATIONS_STRING, or a
  vector of these of the same length as
  `measure`/`measures`. Automatically inferred if left unspecified.

- `repeats` - default is 1; set to a higher value for repeated
  (Monte Carlo) resampling

 - `acceleration` - parallelization option; currently supported
   options are instances of `CPU1` (single-threaded computation)
   `CPUThreads` (multi-threaded computation) and `CPUProcesses`
   (multi-process computation); default is `default_resource()`.

- `force` - default is `false`; set to `true` for force cold-restart
  of each training event

- `verbosity` level, an integer defaulting to 1.

- `check_measure` - default is `true`


### Return value

A property-accessible object of type `PerformanceEvaluation` with
these properties:

- `measure`: the vector of specified measures

- `measurement`: the corresponding measurements, aggregated across the
  test folds using the aggregation method defined for each measure (do
  `aggregation(measure)` to inspect)

- `operation`: for each measure, the operation applied; one of:
  $PREDICT_OPERATIONS_STRING.

- `per_fold`: a vector of vectors of individual test fold evaluations
  (one vector per measure)

- `per_observation`: a vector of vectors of individual observation
  evaluations of those measures for which
  `reports_each_observation(measure)` is true, which is otherwise
  reported `missing`

-`fitted_params_per_fold`: a vector containing `fitted pamarms(mach)` for each
  machine `mach` trained during resampling.

- `report_per_fold`: a vector containing `report(mach)` for each
   machine `mach` training in resampling

"""
function evaluate!(mach::Machine{<:Measurable};
                   resampling=CV(),
                   measures=nothing,
                   measure=measures,
                   weights=nothing,
                   class_weights=nothing,
                   operations=nothing,
                   operation=operations,
                   acceleration=default_resource(),
                   rows=nothing,
                   repeats=1,
                   force=false,
                   check_measure=true,
                   verbosity=1) where M

    # this method just checks validity of options, preprocess the
    # weights, measures, operations, and dispatches a
    # strategy-specific `evaluate!`

    repeats > 0 || error("Need `repeats > 0`. ")

    if resampling isa TrainTestPairs
        if rows !== nothing
            error("You cannot specify `rows` unless `resampling "*
                  "isa MLJ.ResamplingStrategy` is true. ")
        end
        if repeats != 1 && verbosity > 0
            @warn "repeats > 1 not supported unless "*
            "`resampling <: ResamplingStrategy. "
        end
    end

    _measures = _actual_measures(measure, mach.model)

    _operations = _actual_operations(operation,
                                     _measures,
                                     mach.model,
                                     verbosity)

    _check_weights_measures(weights,
                            class_weights,
                            _measures,
                            mach,
                            _operations,
                            verbosity,
                            check_measure)

    _warn_about_unsupported(supports_weights,
                            "Sample", _measures, weights, verbosity)
    _warn_about_unsupported(supports_class_weights,
                            "Class", _measures, class_weights, verbosity)

    _acceleration= _process_accel_settings(acceleration)

    evaluate!(mach, resampling, weights, class_weights, rows, verbosity,
              repeats, _measures, _operations, _acceleration, force)

end

"""
    evaluate(model, data...; cache=true, kw_options...)

Equivalent to `evaluate!(machine(model, data..., cache=cache);
wk_options...)`.  See the machine version `evaluate!` for the complete
list of options.

"""
evaluate(model::Measurable, args...; cache=true, kwargs...) =
    evaluate!(machine(model, args...; cache=cache); kwargs...)

# -------------------------------------------------------------------
# Resource-specific methods to distribute a function parameterized by
# fold number `k` over processes/threads.

# Here `func` is always going to be `fit_and_extract_on_fold`; see later

function _evaluate!(func, mach, ::CPU1, nfolds, verbosity)
    p = Progress(nfolds,

                 dt = 0,
                 desc = "Evaluating over $nfolds folds: ",
                 barglyphs = BarGlyphs("[=> ]"),
                 barlen = 25,
                 color = :yellow)

    ret = mapreduce(vcat, 1:nfolds) do k
        r = func(mach, k)
        verbosity < 1 || begin
            p.counter += 1
            ProgressMeter.updateProgress!(p)
        end
        return [r, ]
    end

    return zip(ret...) |> collect

end

function _evaluate!(func, mach, ::CPUProcesses, nfolds, verbosity)

    local ret
    @sync begin
        p = Progress(nfolds,
                     dt = 0,
                     desc = "Evaluating over $nfolds folds: ",
                     barglyphs = BarGlyphs("[=> ]"),
                     barlen = 25,
                     color = :yellow)
        channel = RemoteChannel(()->Channel{Bool}(min(1000, nfolds)), 1)
        # printing the progress bar
        verbosity < 1 || @async begin
            while take!(channel)
                p.counter +=1
                ProgressMeter.updateProgress!(p)
            end
        end


        ret = @distributed vcat for k in 1:nfolds
            r = func(mach, k)
            verbosity < 1 || begin
                put!(channel, true)
            end
            [r, ]
        end

        verbosity < 1 || put!(channel, false)

    end

    return zip(ret...) |> collect
end

@static if VERSION >= v"1.3.0-DEV.573"

# determines if an instantiated machine caches data:
_caches_data(::Machine{M, C}) where {M, C} = C

function _evaluate!(func, mach, accel::CPUThreads, nfolds, verbosity)

    nthreads = Threads.nthreads()

    if nthreads == 1
        return _evaluate!(func, mach, CPU1(), nfolds, verbosity)
    end
    ntasks = accel.settings
    partitions = chunks(1:nfolds, ntasks)

    p = Progress(nfolds,
                 dt = 0,
                 desc = "Evaluating over $nfolds folds: ",
                 barglyphs = BarGlyphs("[=> ]"),
                 barlen = 25,
                 color = :yellow)
    ch = Channel{Bool}(min(1000, length(partitions)))

    results = Vector(undef, length(partitions))

    @sync begin
        # printing the progress bar
        verbosity < 1 || @async begin
            while take!(ch)
                p.counter +=1
                ProgressMeter.updateProgress!(p)
            end
        end
        clean!(mach.model)
        #One tmach for each task:
        machines = vcat(mach, [
           machine(mach.model, mach.args...; cache = _caches_data(mach))
           for _ in 2:length(partitions)
        ])

        @sync for (i, parts) in enumerate(partitions)
            Threads.@spawn begin
                results[i] = mapreduce(vcat, parts) do k
                    r = func(machines[i], k)
                    verbosity < 1 || put!(ch, true)
                    [r, ]
                end

            end
        end

        verbosity < 1 || put!(ch, false)

    end

    ret = reduce(vcat, results)

    return zip(ret...) |> collect

end

end
# ------------------------------------------------------------
# Core `evaluation` method, operating on train-test pairs

const AbstractRow = Union{AbstractVector{<:Integer}, Colon}
const TrainTestPair = Tuple{AbstractRow,AbstractRow}
const TrainTestPairs = AbstractVector{<:TrainTestPair}

# helper:
_feature_dependencies_exist(measures) =
    !all(m->!(is_feature_dependent(m)), measures)

# helper:
function measure_specific_weights(measure, weights, class_weights, test)
    supports_weights(measure) && supports_class_weights(measure) &&
        error("Encountered a measure that simultaneously supports "*
              "(per-sample) weights and class weights. ")
    if supports_weights(measure)
        weights === nothing && return nothing
        return weights[test]
    end
    supports_class_weights(measure) && return class_weights
    return nothing
end

# Evaluation when `resampling` is a TrainTestPairs (CORE EVALUATOR):
function evaluate!(mach::Machine, resampling, weights,
                   class_weights, rows, verbosity, repeats,
                   measures, operations, acceleration, force)

    # Note: `rows` and `repeats` are ignored here

    resampling isa TrainTestPairs ||
        error("`resampling` must be an "*
              "`MLJ.ResamplingStrategy` or tuple of pairs "*
              "of the form `(train_rows, test_rows)`")

    X = mach.args[1]()
    y = mach.args[2]()

    nfolds = length(resampling)

    nmeasures = length(measures)

    feature_dependencies_exist = _feature_dependencies_exist(measures)

    function fit_and_extract_on_fold(mach, k)
        train, test = resampling[k]
        fit!(mach; rows=train, verbosity=verbosity - 1, force=force)
        # build a dictionary of predictions keyed on the operations
        # that appear (`predict`, `predict_mode`, etc):
        yhat_given_operation =
            Dict(op=>op(mach, rows=test) for op in unique(operations))
        if feature_dependencies_exist
            Xtest = selectrows(X, test)
        else
            Xtest = nothing
        end
        ytest = selectrows(y, test)

        measurements =  map(measures, operations) do m, op
            wtest = measure_specific_weights(m,
                                             weights,
                                             class_weights,
                                             test)
            value(m, yhat_given_operation[op], Xtest, ytest, wtest)
        end

        fp = fitted_params(mach)
        r = report(mach)
        return (measurements, fp, r)
    end

    if acceleration isa CPUProcesses
        if verbosity > 0
            @info "Distributing evaluations " *
                  "among $(nworkers()) workers."
        end
    end
     if acceleration isa CPUThreads
        if verbosity > 0
                @info "Performing evaluations " *
                      "using $(Threads.nthreads()) threads."
        end
    end

    measurements_vector_of_vectors, fitted_params_per_fold, report_per_fold  =
        _evaluate!(fit_and_extract_on_fold,
                   mach,
                   acceleration,
                   nfolds,
                   verbosity)

    measurements_flat = vcat(measurements_vector_of_vectors...)

    # in the following rows=folds, columns=measures:
    measurements_matrix = permutedims(
        reshape(collect(measurements_flat), (nmeasures, nfolds)))

    # measurements for each observation:
    per_observation = map(1:nmeasures) do k
        m = measures[k]
        if reports_each_observation(m)
            measurements_matrix[:,k]
        else
            missing
        end
    end

    # measurements for each fold:
    per_fold = map(1:nmeasures) do k
        m = measures[k]
        if reports_each_observation(m)
            broadcast(MLJBase.aggregate, per_observation[k], [m,])
        else
            measurements_matrix[:,k]
        end
    end

    # overall aggregates:
    per_measure = map(1:nmeasures) do k
        m = measures[k]
        MLJBase.aggregate(per_fold[k], m)
    end

    ret = (measure                = measures,
           measurement            = per_measure,
           operation              = operations,
           per_fold               = per_fold,
           per_observation        = per_observation,
           fitted_params_per_fold = fitted_params_per_fold |> collect,
           report_per_fold        = report_per_fold |> collect,
           train_test_rows       = resampling)

    return ret

end

# ----------------------------------------------------------------
# Evaluation when `resampling` is a ResamplingStrategy

function evaluate!(mach::Machine, resampling::ResamplingStrategy,
                   weights, class_weights, rows, verbosity, repeats, args...)

    train_args = Tuple(a() for a in mach.args)
    y = train_args[2]

    _rows = actual_rows(rows, nrows(y), verbosity)

    repeated_train_test_pairs =
        vcat([train_test_pairs(resampling, _rows, train_args...)
              for i in 1:repeats]...)

    return evaluate!(mach,
                     repeated_train_test_pairs,
                     weights,
                     class_weights,
                     nothing,
                     verbosity,
                     repeats,
                     args...)

end

# ====================================================================
## RESAMPLER - A MODEL WRAPPER WITH `evaluate` OPERATION

"""
    resampler = Resampler(model=ConstantRegressor(),
                          resampling=CV(),
                          measure=nothing,
                          weights=nothing,
                          class_weights=nothing
                          operation=predict,
                          repeats = 1,
                          acceleration=default_resource(),
                          check_measure=true)

Resampling model wrapper, used internally by the `fit` method of
`TunedModel` instances and `IteratedModel` instances. See
[`evaluate!](@ref) for options. Not intended for general use.

Given a machine `mach = machine(resampler, args...)` one obtains a
performance evaluation of the specified `model`, performed according
to the prescribed `resampling` strategy and other parameters, using
data `args...`, by calling `fit!(mach)` followed by
`evaluate(mach)`.

On subsequent calls to `fit!(mach)` new train/test pairs of row
indices are only regenerated if `resampling`, `repeats` or `cache`
fields of `resampler` have changed. The evolution of an RNG field of
`resampler` does *not* constitute a change (`==` for `MLJType` objects
is not sensitive to such changes; see [`is_same_except'](@ref)).

If there is single train/test pair, then warm-restart behavior of the
wrapped model `resampler.model` will extend to warm-restart behaviour
of the wrapper `resampler`, with respect to mutations of the wrapped
model.

The sample `weights` are passed to the specified performance measures
that support weights for evaluation. These weights are not to be
confused with any weights bound to a `Resampler` instance in a
machine, used for training the wrapped `model` when supported.

The sample `class_weights` are passed to the specified performance
measures that support per-class weights for evaluation. These weights
are not to be confused with any weights bound to a `Resampler` instance
in a machine, used for training the wrapped `model` when supported.

"""
mutable struct Resampler{S} <: Model
    model
    resampling::S # resampling strategy
    measure
    weights::Union{Nothing,AbstractVector{<:Real}}
    class_weights::Union{Nothing, AbstractDict{<:Any, <:Real}}
    operation
    acceleration::AbstractResource
    check_measure::Bool
    repeats::Int
    cache::Bool
end

# Some traits are markded as `missing` because we cannot determine
# them from from the type because we have removed `M` (for "model"} as
# a `Resampler` type parameter. See
# https://github.com/JuliaAI/MLJTuning.jl/issues/141#issue-951221466

StatisticalTraits.is_wrapper(::Type{<:Resampler}) = true
StatisticalTraits.supports_weights(::Type{<:Resampler}) = missing
StatisticalTraits.supports_class_weights(::Type{<:Resampler}) = missing
StatisticalTraits.is_pure_julia(::Type{<:Resampler}) = true

function MLJModelInterface.clean!(resampler::Resampler)
    warning = ""
    if resampler.measure === nothing && resampler.model !== nothing
        measure = default_measure(resampler.model)
        if measure === nothing
            error("No default measure known for $(resampler.model). "*
                  "You must specify measure=... ")
        else
            warning *= "No `measure` specified. "*
            "Setting `measure=$measure`. "
        end
    end

    return warning
end

function Resampler(; model=nothing,
                   resampling=CV(),
                   measure=nothing,
                   weights=nothing,
                   class_weights=nothing,
                   operation=predict,
                   acceleration=default_resource(),
                   check_measure=true,
                   repeats=1,
                   cache=true)

    resampler = Resampler(model,
                          resampling,
                          measure,
                          weights,
                          class_weights,
                          operation,
                          acceleration,
                          check_measure,
                          repeats, cache)
    message = MLJModelInterface.clean!(resampler)
    isempty(message) || @warn message

    return resampler

end

function MLJModelInterface.fit(resampler::Resampler, verbosity::Int, args...)

    mach = machine(resampler.model, args...; cache=resampler.cache)

    _measures = _actual_measures(resampler.measure, resampler.model)

    _operations = _actual_operations(resampler.operation,
                                     _measures,
                                     resampler.model,
                                     verbosity)

    _check_weights_measures(resampler.weights,
                            resampler.class_weights,
                            _measures,
                            mach,
                            _operations,
                            verbosity,
                            resampler.check_measure)

    _acceleration = _process_accel_settings(resampler.acceleration)

    e = evaluate!(mach,
                  resampler.resampling,
                  resampler.weights,
                  resampler.class_weights,
                  nothing,
                  verbosity - 1,
                  resampler.repeats,
                  _measures,
                  _operations,
                  _acceleration,
                  false)

    fitresult = (machine = mach, evaluation = e)
    cache = (resampler = deepcopy(resampler),
             acceleration = _acceleration)
    report =(evaluation = e, )

    return fitresult, cache, report

end

# helper to update the model in a machine

# when the machine's existing model and the new model have same type:
function _update!(mach::Machine{M}, model::M) where M
    mach.model = model
    return mach
end

# when the types are different, we need a new machine:
_update!(mach, model) = machine(model, mach.args...)

function MLJModelInterface.update(resampler::Resampler,
                                  verbosity::Int,
                                  fitresult,
                                  cache,
                                  args...)

    old_resampler, acceleration = cache

    # if we need to generate new train/test pairs, or data caching
    # option has changed, then fit from scratch:
    if resampler.resampling != old_resampler.resampling ||
        resampler.repeats != old_resampler.repeats ||
        resampler.cache != old_resampler.cache
        return MLJModelInterface.fit(resampler, verbosity, args...)
    end

    mach, e = fitresult
    train_test_rows = e.train_test_rows

    measures = e.measure
    operations = e.operation

    # update the model:
    mach2 = _update!(mach, resampler.model)

    # re-evaluate:
    e = evaluate!(mach2,
                  train_test_rows,
                  resampler.weights,
                  resampler.class_weights,
                  nothing,
                  verbosity - 1,
                  resampler.repeats,
                  measures,
                  operations,
                  acceleration,
                  false)

    report = (evaluation = e, )
    fitresult = (machine=mach2, evaluation=e)
    cache = (resampler = deepcopy(resampler),
             acceleration = acceleration)

    return fitresult, cache, report

end

# The input and target scitypes cannot be determined from the type
# because we have removed `M` (for "model") as a `Resampler` type
# parameter. See
# https://github.com/JuliaAI/MLJTuning.jl/issues/141#issue-951221466

StatisticalTraits.input_scitype(::Type{<:Resampler}) = Unknown
StatisticalTraits.target_scitype(::Type{<:Resampler}) = Unknown
StatisticalTraits.package_name(::Type{<:Resampler}) = "MLJBase"
StatisticalTraits.load_path(::Type{<:Resampler}) = "MLJBase.Resampler"

fitted_params(::Resampler, fitresult) = fitresult

evaluate(resampler::Resampler, fitresult) = fitresult.evaluation
function evaluate(machine::Machine{<:Resampler})
    if isdefined(machine, :fitresult)
        return evaluate(machine.model, machine.fitresult)
    else
        throw(error("$machine has not been trained."))
    end
end
