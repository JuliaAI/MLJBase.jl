# # LOCAL TYPE ALIASES

const AbstractRow = Union{AbstractVector{<:Integer}, Colon}
const TrainTestPair = Tuple{AbstractRow, AbstractRow}
const TrainTestPairs = Union{
    Tuple{Vararg{TrainTestPair}},
    AbstractVector{<:TrainTestPair},
}


# # ERROR MESSAGES

const PREDICT_OPERATIONS_STRING = begin
    strings = map(PREDICT_OPERATIONS) do op
        "`"*string(op)*"`"
    end
    join(strings, ", ", ", or ")
end
const PROG_METER_DT = 0.1
const ERR_WEIGHTS_LENGTH =
    DimensionMismatch("`weights` and target have different lengths. ")
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
    "`$measure` does not support a `model` with "*
    "`prediction_type(model) == :$(prediction_type(model))`. "
err_incompatible_prediction_types(model, measure) = ArgumentError(
    _ambiguous_operation(model, measure)*
    "If your model is truly making probabilistic predictions, try explicitly "*
    "specifiying operations. For example, for "*
    "`measures = [area_under_curve, accuracy]`, try "*
    "`operations=[predict, predict_mode]`. ")
const LOG_AVOID = "\nTo override measure checks, set check_measure=false. "
const LOG_SUGGESTION1 =
    "\nPerhaps you want to set `operation="*
    "predict_mode` or need to "*
    "specify multiple operations, "*
    "one for each measure. "
const LOG_SUGGESTION2 =
    "\nPerhaps you want to set `operation="*
    "predict_mean` or `operation=predict_median`, or "*
    "specify multiple operations, "*
    "one for each measure. "
ERR_MEASURES_OBSERVATION_SCITYPE(measure, T_measure, T) = ArgumentError(
    "\nobservation scitype of target = `$T` but ($measure) only supports "*
        "`$T_measure`."*LOG_AVOID
)
ERR_MEASURES_PROBABILISTIC(measure, suggestion) = ArgumentError(
    "The model subtypes `Probabilistic`, and so is not supported by "*
        "`$measure`. $suggestion"*LOG_AVOID
)
ERR_MEASURES_DETERMINISTIC(measure) = ArgumentError(
    "The model subtypes `Deterministic`, "*
        "and so is not supported by `$measure`. "*LOG_AVOID
)

err_ambiguous_operation(model, measure) = ArgumentError(
    _ambiguous_operation(model, measure)*
    "\nUnable to infer an appropriate operation for `$measure`. "*
    "Explicitly specify `operation=...` or `operations=...`. "*
    "Possible value(s) are: $PREDICT_OPERATIONS_STRING. "
)

const ERR_UNSUPPORTED_PREDICTION_TYPE = ArgumentError(
    """

    The `prediction_type` of your model needs to be one of: `:deterministic`,
    `:probabilistic`, or `:interval`. Does your model implement one of these operations:
    $PREDICT_OPERATIONS_STRING? If so, you can try explicitly specifying `operation=...`
    or `operations=...` (and consider posting an issue to have the model review it's
    definition of `MLJModelInterface.prediction_type`). Otherwise, performance
    evaluation is not supported.

   """
)

const ERR_NEED_TARGET = ArgumentError(
   """

    To evaluate a model's performance you must provide a target variable `y`, as in
    `evaluate(model, X, y; options...)` or

        mach = machine(model, X, y)
        evaluate!(mach; options...)

    """
)

const ERR_BAD_RESAMPLING_OPTION = ArgumentError(
    "`resampling` must be an "*
        "`MLJ.ResamplingStrategy` or a vector (or tuple) of tuples "*
        "of the form `(train_rows, test_rows)`"
)

const ERR_EMPTY_RESAMPLING_OPTION = ArgumentError(
    "`resampling` cannot be emtpy. It must be an "*
        "`MLJ.ResamplingStrategy` or a vector (or tuple) of tuples "*
        "of the form `(train_rows, test_rows)`"
)


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

# ================================================================
## EVALUATION RESULT TYPE

abstract type AbstractPerformanceEvaluation <: MLJType end

"""
    PerformanceEvaluation <: AbstractPerformanceEvaluation

Type of object returned by [`evaluate`](@ref) (for models plus data) or
[`evaluate!`](@ref) (for machines). Such objects encode estimates of the performance
(generalization error) of a supervised model or outlier detection model, and store other
information ancillary to the computation.

If [`evaluate`](@ref) or [`evaluate!`](@ref) is called with the `compact=true` option,
then a [`CompactPerformanceEvaluation`](@ref) object is returned instead.

When `evaluate`/`evaluate!` is called, a number of train/test pairs ("folds") of row
indices are generated, according to the options provided, which are discussed in the
[`evaluate!`](@ref) doc-string. Rows correspond to observations. The generated train/test
pairs are recorded in the `train_test_rows` field of the `PerformanceEvaluation` struct,
and the corresponding estimates, aggregated over all train/test pairs, are recorded in
`measurement`, a vector with one entry for each measure (metric) recorded in `measure`.

When displayed, a `PerformanceEvaluation` object includes a value under the heading
`1.96*SE`, derived from the standard error of the `per_fold` entries. This value is
suitable for constructing a formal 95% confidence interval for the given
`measurement`. Such intervals should be interpreted with caution. See, for example, [Bates
et al.  (2021)](https://arxiv.org/abs/2104.00673).

### Fields

These fields are part of the public API of the `PerformanceEvaluation` struct.

- `model`: model used to create the performance evaluation. In the case a
    tuning model, this is the best model found.

- `measure`: vector of measures (metrics) used to evaluate performance

- `measurement`: vector of measurements - one for each element of `measure` - aggregating
  the performance measurements over all train/test pairs (folds). The aggregation method
  applied for a given measure `m` is
  `StatisticalMeasuresBase.external_aggregation_mode(m)` (commonly `Mean()` or `Sum()`)

- `operation` (e.g., `predict_mode`): the operations applied for each measure to generate
  predictions to be evaluated. Possibilities are: $PREDICT_OPERATIONS_STRING.

- `per_fold`: a vector of vectors of individual test fold evaluations (one vector per
  measure). Useful for obtaining a rough estimate of the variance of the performance
  estimate.

- `per_observation`: a vector of vectors of vectors containing individual per-observation
  measurements: for an evaluation `e`, `e.per_observation[m][f][i]` is the measurement for
  the `i`th observation in the `f`th test fold, evaluated using the `m`th measure.  Useful
  for some forms of hyper-parameter optimization. Note that an aggregregated measurement
  for some measure `measure` is repeated across all observations in a fold if
  `StatisticalMeasures.can_report_unaggregated(measure) == true`. If `e` has been computed
  with the `per_observation=false` option, then `e_per_observation` is a vector of
  `missings`.

- `fitted_params_per_fold`: a vector containing `fitted params(mach)` for each machine
  `mach` trained during resampling - one machine per train/test pair. Use this to extract
  the learned parameters for each individual training event.

- `report_per_fold`: a vector containing `report(mach)` for each machine `mach` training
  in resampling - one machine per train/test pair.

- `train_test_rows`: a vector of tuples, each of the form `(train, test)`, where `train`
  and `test` are vectors of row (observation) indices for training and evaluation
  respectively.

- `resampling`: the user-specified resampling strategy to generate the train/test pairs
  (or literal train/test pairs if that was directly specified).

- `repeats`: the number of times the resampling strategy was repeated.

See also [`CompactPerformanceEvaluation`](@ref).

"""
struct PerformanceEvaluation{M,
                             Measure,
                             Measurement,
                             Operation,
                             PerFold,
                             PerObservation,
                             FittedParamsPerFold,
                             ReportPerFold,
                             R} <: AbstractPerformanceEvaluation
    model::M
    measure::Measure
    measurement::Measurement
    operation::Operation
    per_fold::PerFold
    per_observation::PerObservation
    fitted_params_per_fold::FittedParamsPerFold
    report_per_fold::ReportPerFold
    train_test_rows::TrainTestPairs
    resampling::R
    repeats::Int
end

"""
    CompactPerformanceEvaluation <: AbstractPerformanceEvaluation

Type of object returned by [`evaluate`](@ref) (for models plus data) or
[`evaluate!`](@ref) (for machines) when called with the option `compact = true`. Such
objects have the same structure as the [`PerformanceEvaluation`](@ref) objects returned by
default, except that the following fields are omitted to save memory:
`fitted_params_per_fold`, `report_per_fold`, `train_test_rows`.

For more on the remaining fields, see [`PerformanceEvaluation`](@ref).

"""
struct CompactPerformanceEvaluation{M,
                             Measure,
                             Measurement,
                             Operation,
                             PerFold,
                             PerObservation,
                             R} <: AbstractPerformanceEvaluation
    model::M
    measure::Measure
    measurement::Measurement
    operation::Operation
    per_fold::PerFold
    per_observation::PerObservation
    resampling::R
    repeats::Int
end

compactify(e::CompactPerformanceEvaluation) = e
compactify(e::PerformanceEvaluation) = CompactPerformanceEvaluation(
    e.model,
    e.measure,
    e.measurement,
    e.operation,
    e.per_fold,
    e. per_observation,
    e.resampling,
    e.repeats,
)

# pretty printing:
round3(x) = x
round3(x::AbstractFloat) = round(x, sigdigits=3)

const SE_FACTOR = 1.96 # For a 95% confidence interval.

_standard_error(v::AbstractVector{<:Real}) = SE_FACTOR*std(v) / sqrt(length(v) - 1)
_standard_error(v) = "N/A"

function _standard_errors(e::AbstractPerformanceEvaluation)
    measure = e.measure
    length(e.per_fold[1]) == 1 && return [nothing]
    std_errors = map(_standard_error, e.per_fold)
    return std_errors
end

# to address #874, while preserving the display worked out in #757:
_repr_(f::Function) = repr(f)
_repr_(x) = repr("text/plain", x)

# helper for row labels: _label(1) ="A", _label(2) = "B", _label(27) = "BA", etc
const alphabet = Char.(65:90)
_label(i) = map(digits(i - 1, base=26)) do d alphabet[d + 1] end |> join |> reverse

function Base.show(io::IO, ::MIME"text/plain", e::AbstractPerformanceEvaluation)
    _measure = [_repr_(m) for m in e.measure]
    _measurement = round3.(e.measurement)
    _per_fold = [round3.(v) for v in e.per_fold]
    _sterr = round3.(_standard_errors(e))
    row_labels = _label.(eachindex(e.measure))

    # Define header and data for main table

    data = hcat(_measure, e.operation, _measurement)
    header = ["measure", "operation", "measurement"]
    if length(row_labels) > 1
        data = hcat(row_labels, data)
        header =["", header...]
    end

    if e isa PerformanceEvaluation
        println(io, "PerformanceEvaluation object "*
            "with these fields:")
        println(io, "  model, measure, operation,\n"*
            "  measurement, per_fold, per_observation,\n"*
            "  fitted_params_per_fold, report_per_fold,\n"*
            "  train_test_rows, resampling, repeats")
    else
        println(io, "CompactPerformanceEvaluation object "*
            "with these fields:")
        println(io, "  model, measure, operation,\n"*
            "  measurement, per_fold, per_observation,\n"*
            "  train_test_rows, resampling, repeats")
    end

    println(io, "Extract:")
    show_color = MLJBase.SHOW_COLOR[]
    color_off()
    PrettyTables.pretty_table(
        io,
        data;
        header,
        header_crayon=PrettyTables.Crayon(bold=false),
        alignment=:l,
        linebreaks=true,
    )

    # Show the per-fold table if needed:

    if length(first(e.per_fold)) > 1
        show_sterr = any(!isnothing, _sterr)
        data2 = hcat(_per_fold, _sterr)
        header2 = ["per_fold", "1.96*SE"]
        if length(row_labels) > 1
            data2 = hcat(row_labels, data2)
            header2 =["", header2...]
        end
        PrettyTables.pretty_table(
            io,
            data2;
            header=header2,
            header_crayon=PrettyTables.Crayon(bold=false),
            alignment=:l,
            linebreaks=true,
        )
    end
    show_color ? color_on() : color_off()
end

_summary(e) = Tuple(round3.(e.measurement))
Base.show(io::IO, e::PerformanceEvaluation) =
    print(io, "PerformanceEvaluation$(_summary(e))")
Base.show(io::IO, e::CompactPerformanceEvaluation) =
    print(io, "CompactPerformanceEvaluation$(_summary(e))")



# ===============================================================
## USER CONTROL OF DEFAULT LOGGING

const DOC_DEFAULT_LOGGER =
"""

The default logger is used in calls to [`evaluate!`](@ref) and [`evaluate`](@ref), and
in the constructors `TunedModel` and `IteratedModel`, unless the `logger` keyword is
explicitly specified.

!!! note

    Prior to MLJ v0.20.7 (and MLJBase 1.5) the default logger was always `nothing`.

"""

"""
    default_logger()

Return the current value of the default logger for use with supported machine learning
tracking platforms, such as [MLflow](https://mlflow.org/docs/latest/index.html).

$DOC_DEFAULT_LOGGER

When MLJBase is first loaded, the default logger is `nothing`.

"""
default_logger() = DEFAULT_LOGGER[]

"""
    default_logger(logger)

Reset the default logger.

# Example

Suppose an [MLflow](https://mlflow.org/docs/latest/index.html) tracking service is running
on a local server at `http://127.0.0.1:500`. Then in every `evaluate` call in which
`logger` is not specified, the peformance evaluation is
automatically logged to the service, as here:

```julia
using MLJ
logger = MLJFlow.Logger("http://127.0.0.1:5000/api")
default_logger(logger)

X, y = make_moons()
model = ConstantClassifier()
evaluate(model, X, y, measures=[log_loss, accuracy)])
```

"""
function default_logger(logger)
    DEFAULT_LOGGER[] = logger
end


# ===============================================================
## EVALUATION METHODS

# ---------------------------------------------------------------
# Helpers

# to fill out predictions in the case of density estimation ("cone" construction):
fill_if_needed(yhat, X, n) = yhat
fill_if_needed(yhat, X::Nothing, n) = FillArrays.Fill(yhat, n)

function actual_rows(rows, N, verbosity)
    unspecified_rows = (rows === nothing)
    _rows = unspecified_rows ? (1:N) : rows
    if !unspecified_rows && verbosity > 0
        @info "Creating subsamples from a subset of all rows. "
    end
    return _rows
end

function _check_measure(measure, operation, model, y)

    # get observation scitype:
    T = MLJBase.guess_observation_scitype(y)

    # get type supported by measure:
    T_measure = StatisticalMeasuresBase.observation_scitype(measure)

    T == Unknown && (return true)
    T_measure == Union{} && (return true)
    isnothing(StatisticalMeasuresBase.kind_of_proxy(measure)) && (return true)


    T <: T_measure || throw(ERR_MEASURES_OBSERVATION_SCITYPE(measure, T_measure, T))

    incompatible = model isa Probabilistic &&
        operation == predict &&
        StatisticalMeasuresBase.kind_of_proxy(measure) != LearnAPI.Distribution()

    if incompatible
        if T <: Union{Missing,Finite}
            suggestion = LOG_SUGGESTION1
        elseif T <: Union{Missing,Infinite}
            suggestion = LOG_SUGGESTION2
        else
            suggestion = ""
        end
        throw(ERR_MEASURES_PROBABILISTIC(measure, suggestion))
    end

    model isa Deterministic &&
        StatisticalMeasuresBase.kind_of_proxy(measure) != LearnAPI.Point() &&
        throw(ERR_MEASURES_DETERMINISTIC(measure))

    return true

end

function _check_measures(measures, operations, model, y)
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

    # wrap in `robust_measure` to allow unsupported weights to be silently treated as
    # uniform when invoked; `_check_measure` will throw appropriate warnings unless
    # explicitly suppressed.
    return StatisticalMeasuresBase.robust_measure.(_measures)

end

function _check_weights(weights, nrows)
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

        # `kind_of_proxy` is the measure trait corresponding to `prediction_type` model
        # trait. But it's values are instances of LearnAPI.KindOfProxy, instead of
        # symbols:
        #
        # `LearnAPI.Point()`        ~ `:deterministic` (`model isa Deterministic`)
        # `LearnAPI.Distribution()` ~ `:probabilistic` (`model isa Deterministic`)
        #
        kind_of_proxy = StatisticalMeasuresBase.kind_of_proxy(m)

        # `observation_type` is the measure trait which we need to match the model
        # `target_scitype` but the latter refers to the whole target `y`, not a single
        # observation.
        #
        # One day, models will have their own `observation_scitype`
        observation_scitype = StatisticalMeasuresBase.observation_scitype(m)

        # One day, models will implement LearnAPI and will get their own `kind_of_proxy`
        # trait replacing `prediction_type` and `observation_scitype` trait replacing
        # `target_scitype`.

        isnothing(kind_of_proxy) && (return predict)

        if MLJBase.prediction_type(model) === :probabilistic
            if kind_of_proxy === LearnAPI.Distribution()
                return predict
            elseif kind_of_proxy === LearnAPI.Point()
                if observation_scitype <: Union{Missing,Finite}
                    return predict_mode
                elseif observation_scitype <:Union{Missing,Infinite}
                    return predict_mean
                else
                    throw(err_ambiguous_operation(model, m))
                end
            else
                throw(err_ambiguous_operation(model, m))
            end
        elseif MLJBase.prediction_type(model) === :deterministic
            if kind_of_proxy === LearnAPI.Distribution()
                throw(err_incompatible_prediction_types(model, m))
            elseif kind_of_proxy === LearnAPI.Point()
                return predict
            else
                throw(err_ambiguous_operation(model, m))
            end
        elseif MLJBase.prediction_type(model) === :interval
            if kind_of_proxy === LearnAPI.ConfidenceInterval()
                return predict
            else
                throw(err_ambiguous_operation(model, m))
            end
        else
            throw(ERR_UNSUPPORTED_PREDICTION_TYPE)
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

_process_accel_settings(accel::Union{CPU1, CPUProcesses}) = accel

#fallback
_process_accel_settings(accel) =  throw(ArgumentError("unsupported" *
                            " acceleration parameter`acceleration = $accel` "))

# --------------------------------------------------------------
# User interface points: `evaluate!` and `evaluate`

const RESAMPLING_STRATEGIES = subtypes(ResamplingStrategy)
const RESAMPLING_STRATEGIES_LIST =
    join(
        map(RESAMPLING_STRATEGIES) do s
             name = split(string(s), ".") |> last
             "`$name`"
        end,
        ", ",
        " and ",
    )

"""
    log_evaluation(logger, performance_evaluation)

Log a performance evaluation to `logger`, an object specific to some logging platform,
such as mlflow. If `logger=nothing` then no logging is performed.  The method is called at
the end of every call to `evaluate/evaluate!` using the logger provided by the `logger`
keyword argument.

# Implementations for new logging platforms

Julia interfaces to workflow logging platforms, such as mlflow (provided by the
MLFlowClient.jl interface) should overload `log_evaluation(logger::LoggerType,
performance_evaluation)`, where `LoggerType` is a platform-specific type for logger
objects. For an example, see the implementation provided by the MLJFlow.jl package.

"""
log_evaluation(logger, performance_evaluation) = nothing

"""
    evaluate!(mach; resampling=CV(), measure=nothing, options...)

Estimate the performance of a machine `mach` wrapping a supervised model in data, using
the specified `resampling` strategy (defaulting to 6-fold cross-validation) and `measure`,
which can be a single measure or vector. Returns a [`PerformanceEvaluation`](@ref)
object.

Available resampling strategies are $RESAMPLING_STRATEGIES_LIST. If `resampling` is not an
instance of one of these, then a vector of tuples of the form `(train_rows, test_rows)`
is expected. For example, setting

```julia
resampling = [(1:100, 101:200),
              (101:200, 1:100)]
```

gives two-fold cross-validation using the first 200 rows of data.

Any measure conforming to the
[StatisticalMeasuresBase.jl](https://juliaai.github.io/StatisticalMeasuresBase.jl/dev/)
API can be provided, assuming it can consume multiple observations.

Although `evaluate!` is mutating, `mach.model` and `mach.args` are not mutated.

# Additional keyword options

- `rows` - vector of observation indices from which both train and test folds are
  constructed (default is all observations)

- `operation`/`operations=nothing` - One of $PREDICT_OPERATIONS_STRING, or a vector of
  these of the same length as `measure`/`measures`. Automatically inferred if left
  unspecified. For example, `predict_mode` will be used for a `Multiclass` target, if
  `model` is a probabilistic predictor, but `measure` is expects literal (point) target
  predictions. Operations actually applied can be inspected from the `operation` field of
  the object returned.

- `weights` - per-sample `Real` weights for measures that support them (not to be confused
  with weights used in training, such as the `w` in `mach = machine(model, X, y, w)`).

- `class_weights` - dictionary of `Real` per-class weights for use with measures that
  support these, in classification problems (not to be confused
  with weights used in training, such as the `w` in `mach = machine(model, X, y, w)`).

- `repeats::Int=1`: set to a higher value for repeated (Monte Carlo)
  resampling. For example, if `repeats = 10`, then `resampling = CV(nfolds=5,
  shuffle=true)`, generates a total of 50 `(train, test)` pairs for evaluation and
  subsequent aggregation.

- `acceleration=CPU1()`: acceleration/parallelization option; can be any instance of
  `CPU1`, (single-threaded computation), `CPUThreads` (multi-threaded computation) or
  `CPUProcesses` (multi-process computation); default is `default_resource()`. These types
  are owned by ComputationalResources.jl.

- `force=false`: set to `true` to force cold-restart
  of each training event

- `verbosity::Int=1` logging level; can be negative

- `check_measure=true`: whether to screen measures for possible incompatibility with the
  model. Will not catch all incompatibilities.

- `per_observation=true`: whether to calculate estimates for individual observations; if
  `false` the `per_observation` field of the returned object is populated with
  `missing`s. Setting to `false` may reduce compute time and allocations.

- `logger=default_logger()` - a logger object for forwarding results to a machine learning
  tracking platform; see [`default_logger`](@ref) for details.

- `compact=false` - if `true`, the returned evaluation object excludes these fields:
  `fitted_params_per_fold`, `report_per_fold`, `train_test_rows`.

See also [`evaluate`](@ref), [`PerformanceEvaluation`](@ref),
[`CompactPerformanceEvaluation`](@ref).

"""
function evaluate!(
    mach::Machine;
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
    per_observation=true,
    verbosity=1,
    logger=default_logger(),
    compact=false,
    )

    # this method just checks validity of options, preprocess the
    # weights, measures, operations, and dispatches a
    # strategy-specific `evaluate!`

    length(mach.args) > 1 || throw(ERR_NEED_TARGET)

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

    _warn_about_unsupported(
        StatisticalMeasuresBase.supports_weights,
        "Sample",
        _measures,
        weights,
        verbosity,
    )
    _warn_about_unsupported(
        StatisticalMeasuresBase.supports_class_weights,
        "Class",
        _measures,
        class_weights,
        verbosity,
    )

    _acceleration= _process_accel_settings(acceleration)

    evaluate!(
        mach,
        resampling,
        weights,
        class_weights,
        rows,
        verbosity,
        repeats,
        _measures,
        _operations,
        _acceleration,
        force,
        per_observation,
        logger,
        resampling,
        compact,
    )
end

"""
    evaluate(model, data...; cache=true, options...)

Equivalent to `evaluate!(machine(model, data..., cache=cache); options...)`.
See the machine version `evaluate!` for the complete list of options.

Returns a  [`PerformanceEvaluation`](@ref) object.

See also [`evaluate!`](@ref).

"""
evaluate(model::Model, args...; cache=true, kwargs...) =
    evaluate!(machine(model, args...; cache=cache); kwargs...)

# -------------------------------------------------------------------
# Resource-specific methods to distribute a function parameterized by
# fold number `k` over processes/threads.

# Here `func` is always going to be `fit_and_extract_on_fold`; see later

function _next!(p)
    p.counter +=1
    ProgressMeter.updateProgress!(p)
end

function _evaluate!(func, mach, ::CPU1, nfolds, verbosity)
    if verbosity > 0
        p = Progress(
            nfolds,
            dt = PROG_METER_DT,
            desc = "Evaluating over $nfolds folds: ",
            barglyphs = BarGlyphs("[=> ]"),
            barlen = 25,
            color = :yellow
        )
    end

    ret = mapreduce(vcat, 1:nfolds) do k
        r = func(mach, k)
        verbosity < 1 || _next!(p)
        return [r, ]
    end

    return zip(ret...) |> collect

end

function _evaluate!(func, mach, ::CPUProcesses, nfolds, verbosity)

    local ret
    @sync begin
        if verbosity > 0
            p = Progress(
                nfolds,
                dt = PROG_METER_DT,
                desc = "Evaluating over $nfolds folds: ",
                barglyphs = BarGlyphs("[=> ]"),
                barlen = 25,
                color = :yellow
            )
            channel = RemoteChannel(()->Channel{Bool}(), 1)
        end

        # printing the progress bar
        verbosity < 1 || @async begin
            while take!(channel)
                _next!(p)
            end
        end

        ret = @distributed vcat for k in 1:nfolds
            r = func(mach, k)
            verbosity < 1 || put!(channel, true)
            [r, ]
        end

        verbosity < 1 || put!(channel, false)
    end

    return zip(ret...) |> collect
end

@static if VERSION >= v"1.3.0-DEV.573"

# determines if an instantiated machine caches data:
_caches_data(::Machine{<:Any,<:Any,C}) where C = C

function _evaluate!(func, mach, accel::CPUThreads, nfolds, verbosity)

    nthreads = Threads.nthreads()

    if nthreads == 1
        return _evaluate!(func, mach, CPU1(), nfolds, verbosity)
    end

    ntasks = accel.settings
    partitions = chunks(1:nfolds, ntasks)
    if verbosity > 0
        p = Progress(
            nfolds,
            dt = PROG_METER_DT,
            desc = "Evaluating over $nfolds folds: ",
            barglyphs = BarGlyphs("[=> ]"),
            barlen = 25,
            color = :yellow
        )
        ch = Channel{Bool}()
    end

    results = Vector(undef, length(partitions))

    @sync begin
        # printing the progress bar
        verbosity < 1 || @async begin
            while take!(ch)
                _next!(p)
            end
        end

        clean!(mach.model)
        # One tmach for each task:
        machines = vcat(
            mach,
            [
                machine(mach.model, mach.args...; cache = _caches_data(mach))
                for _ in 2:length(partitions)
            ]
        )

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

_view(::Nothing, rows) = nothing
_view(weights, rows) = view(weights, rows)

# Evaluation when `resampling` is a TrainTestPairs (CORE EVALUATOR):
function evaluate!(
    mach::Machine,
    resampling,
    weights,
    class_weights,
    rows,
    verbosity,
    repeats,
    measures,
    operations,
    acceleration,
    force,
    per_observation_flag,
    logger,
    user_resampling,
    compact,
    )

    # Note: `user_resampling` keyword argument is the user-defined resampling strategy,
    # while `resampling` is always a `TrainTestPairs`.

    # Note: `rows` and `repeats` are only passed to the final `PeformanceEvaluation`
    # object to be returned and are not otherwise used here.

    isempty(resampling) && throw(ERR_EMPTY_RESAMPLING_OPTION)
    resampling isa TrainTestPairs || throw(ERR_BAD_RESAMPLING_OPTION)

    X = mach.args[1]()
    y = mach.args[2]()
    nrows = MLJBase.nrows(y)

    nfolds = length(resampling)
    test_fold_sizes = map(resampling) do train_test_pair
        test = last(train_test_pair)
        test isa Colon && (return nrows)
        length(test)
    end

    # weights used to aggregate per-fold measurements, which depends on a measures
    # external mode of aggregation:
    fold_weights(mode) = nfolds .* test_fold_sizes ./ sum(test_fold_sizes)
    fold_weights(::StatisticalMeasuresBase.Sum) = nothing

    nmeasures = length(measures)

    function fit_and_extract_on_fold(mach, k)
        train, test = resampling[k]
        fit!(mach; rows=train, verbosity=verbosity - 1, force=force)
        ntest = MLJBase.nrows(test)
        # build a dictionary of predictions keyed on the operations
        # that appear (`predict`, `predict_mode`, etc):
        yhat_given_operation =
            Dict(op=>
            fill_if_needed(op(mach, rows=test), X, ntest)
                 for op in unique(operations))
        # Note: `fill_if_need(yhat, X, n) = yhat` in typical case that `X` is different
        # from `nothing`.

        ytest = selectrows(y, test)
        if per_observation_flag
            measurements =  map(measures, operations) do m, op
                StatisticalMeasuresBase.measurements(
                    m,
                    yhat_given_operation[op],
                    ytest,
                    _view(weights, test),
                    class_weights,
                )
            end
        else
            measurements =  map(measures, operations) do m, op
                m(
                    yhat_given_operation[op],
                    ytest,
                    _view(weights, test),
                    class_weights,
                )
            end
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
            nthreads = Threads.nthreads()
            @info "Performing evaluations " *
              "using $(nthreads) thread" * ifelse(nthreads == 1, ".", "s.")
        end
    end

    measurements_vector_of_vectors, fitted_params_per_fold, report_per_fold  =
        _evaluate!(
            fit_and_extract_on_fold,
            mach,
            acceleration,
            nfolds,
            verbosity
        )

    measurements_flat = vcat(measurements_vector_of_vectors...)

    # In the `measurements_matrix` below, rows=folds, columns=measures; each element of
    # the matrix is:
    #
    # - a vector of meausurements, one per observation within a fold, if
    # - `per_observation_flag = true`; or
    #
    # - a single measurment for the whole fold, if `per_observation_flag = false`.
    #
    measurements_matrix = permutedims(
        reshape(collect(measurements_flat), (nmeasures, nfolds))
    )

    # measurements for each observation:
    per_observation = if per_observation_flag
       map(1:nmeasures) do k
           measurements_matrix[:,k]
       end
    else
        fill(missing, nmeasures)
    end

    # measurements for each fold:
    per_fold = if per_observation_flag
        map(1:nmeasures) do k
            m = measures[k]
            mode = StatisticalMeasuresBase.external_aggregation_mode(m)
            map(per_observation[k]) do v
                StatisticalMeasuresBase.aggregate(v; mode)
            end
        end
    else
        map(1:nmeasures) do k
            measurements_matrix[:,k]
        end
    end

    # overall aggregates:
    per_measure = map(1:nmeasures) do k
        m = measures[k]
        mode = StatisticalMeasuresBase.external_aggregation_mode(m)
        StatisticalMeasuresBase.aggregate(
            per_fold[k];
            mode,
            weights=fold_weights(mode),
        )
    end

    evaluation = PerformanceEvaluation(
        mach.model,
        measures,
        per_measure,
        operations,
        per_fold,
        per_observation,
        fitted_params_per_fold |> collect,
        report_per_fold |> collect,
        resampling,
        user_resampling,
        repeats
    )
    log_evaluation(logger, evaluation)

    compact && return compactify(evaluation)
    return evaluation
end

# ----------------------------------------------------------------
# Evaluation when `resampling` is a ResamplingStrategy

function evaluate!(mach::Machine, resampling::ResamplingStrategy,
                   weights, class_weights, rows, verbosity, repeats, args...)

    train_args = Tuple(a() for a in mach.args)
    y = train_args[2]

    _rows = actual_rows(rows, nrows(y), verbosity)

    repeated_train_test_pairs =
        vcat(
            [train_test_pairs(resampling, _rows, train_args...) for i in 1:repeats]...
        )

    evaluate!(
        mach,
        repeated_train_test_pairs,
        weights,
        class_weights,
        nothing,
        verbosity,
        repeats,
        args...
    )
end

# ====================================================================
## RESAMPLER - A MODEL WRAPPER WITH `evaluate` OPERATION

"""
    resampler = Resampler(
        model=ConstantRegressor(),
        resampling=CV(),
        measure=nothing,
        weights=nothing,
        class_weights=nothing
        operation=predict,
        repeats = 1,
        acceleration=default_resource(),
        check_measure=true,
        per_observation=true,
        logger=default_logger(),
        compact=false,
    )

*Private method.* Use at own risk.

Resampling model wrapper, used internally by the `fit` method of `TunedModel` instances
and `IteratedModel` instances. See [`evaluate!`](@ref) for meaning of the options. Not
intended for use by general user, who will ordinarily use [`evaluate!`](@ref) directly.

Given a machine `mach = machine(resampler, args...)` one obtains a performance evaluation
of the specified `model`, performed according to the prescribed `resampling` strategy and
other parameters, using data `args...`, by calling `fit!(mach)` followed by
`evaluate(mach)`.

On subsequent calls to `fit!(mach)` new train/test pairs of row indices are only
regenerated if `resampling`, `repeats` or `cache` fields of `resampler` have changed. The
evolution of an RNG field of `resampler` does *not* constitute a change (`==` for
`MLJType` objects is not sensitive to such changes; see [`is_same_except`](@ref)).

If there is single train/test pair, then warm-restart behavior of the wrapped model
`resampler.model` will extend to warm-restart behaviour of the wrapper `resampler`, with
respect to mutations of the wrapped model.

The sample `weights` are passed to the specified performance measures that support weights
for evaluation. These weights are not to be confused with any weights bound to a
`Resampler` instance in a machine, used for training the wrapped `model` when supported.

The sample `class_weights` are passed to the specified performance measures that support
per-class weights for evaluation. These weights are not to be confused with any weights
bound to a `Resampler` instance in a machine, used for training the wrapped `model` when
supported.

"""
mutable struct Resampler{S, L} <: Model
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
    per_observation::Bool
    logger::L
    compact::Bool
end

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

function Resampler(
    ;model=nothing,
    resampling=CV(),
    measures=nothing,
    measure=measures,
    weights=nothing,
    class_weights=nothing,
    operations=predict,
    operation=operations,
    acceleration=default_resource(),
    check_measure=true,
    repeats=1,
    cache=true,
    per_observation=true,
    logger=default_logger(),
    compact=false,
)
    resampler = Resampler(
        model,
        resampling,
        measure,
        weights,
        class_weights,
        operation,
        acceleration,
        check_measure,
        repeats,
        cache,
        per_observation,
        logger,
        compact,
    )
    message = MLJModelInterface.clean!(resampler)
    isempty(message) || @warn message

    return resampler

end

function MLJModelInterface.fit(resampler::Resampler, verbosity::Int, args...)

    mach = machine(resampler.model, args...; cache=resampler.cache)

    _measures = _actual_measures(resampler.measure, resampler.model)

    _operations = _actual_operations(
        resampler.operation,
        _measures,
        resampler.model,
        verbosity
    )

    _check_weights_measures(
        resampler.weights,
        resampler.class_weights,
        _measures,
        mach,
        _operations,
        verbosity,
        resampler.check_measure
    )

    _acceleration = _process_accel_settings(resampler.acceleration)

    # the value of `compact` below is always `false`, because we need
    # `e.train_test_rows` in `update`. (If `resampler.compact=true`, then
    # `evaluate(resampler, ...)` returns the compactified version of the current
    # `PerformanceEvaluation` object.)
    e = evaluate!(
        mach,
        resampler.resampling,
        resampler.weights,
        resampler.class_weights,
        nothing,
        verbosity - 1,
        resampler.repeats,
        _measures,
        _operations,
        _acceleration,
        false,
        resampler.per_observation,
        resampler.logger,
        resampler.resampling,
        false, # compact
    )

    fitresult = (machine = mach, evaluation = e)
    cache = (
        resampler = deepcopy(resampler),
        acceleration = _acceleration
    )
    report = (evaluation = e, )

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

function MLJModelInterface.update(
    resampler::Resampler,
    verbosity::Int,
    fitresult,
    cache,
    args...
)
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

    # since `resampler.model` could have changed, so might the actual measures and
    # operations that should be passed to the (low level) `evaluate!`:
    measures = _actual_measures(resampler.measure, resampler.model)
    operations = _actual_operations(
        resampler.operation,
        measures,
        resampler.model,
        verbosity
    )

    # update the model:
    mach2 = _update!(mach, resampler.model)

    # re-evaluate:
    e = evaluate!(
        mach2,
        train_test_rows,
        resampler.weights,
        resampler.class_weights,
        nothing,
        verbosity - 1,
        resampler.repeats,
        measures,
        operations,
        acceleration,
        false,
        resampler.per_observation,
        resampler.logger,
        resampler.resampling,
        false # we use `compact=false`; see comment in `fit` above
    )
    report = (evaluation = e, )
    fitresult = (machine=mach2, evaluation=e)
    cache = (
        resampler = deepcopy(resampler),
        acceleration = acceleration
    )
    return fitresult, cache, report

end

# Some traits are marked as `missing` because we cannot determine
# them from from the type because we have removed `M` (for "model"} as
# a `Resampler` type parameter. See
# https://github.com/JuliaAI/MLJTuning.jl/issues/141#issue-951221466

StatisticalTraits.is_wrapper(::Type{<:Resampler}) = true
StatisticalTraits.supports_weights(::Type{<:Resampler}) = missing
StatisticalTraits.supports_class_weights(::Type{<:Resampler}) = missing
StatisticalTraits.is_pure_julia(::Type{<:Resampler}) = true
StatisticalTraits.constructor(::Type{<:Resampler}) = Resampler
StatisticalTraits.input_scitype(::Type{<:Resampler}) = Unknown
StatisticalTraits.target_scitype(::Type{<:Resampler}) = Unknown
StatisticalTraits.package_name(::Type{<:Resampler}) = "MLJBase"
StatisticalTraits.load_path(::Type{<:Resampler}) = "MLJBase.Resampler"

fitted_params(::Resampler, fitresult) = fitresult

evaluate(resampler::Resampler, fitresult) = resampler.compact ?
    compactify(fitresult.evaluation) : fitresult.evaluation

function evaluate(machine::Machine{<:Resampler})
    if isdefined(machine, :fitresult)
        return evaluate(machine.model, machine.fitresult)
    else
        throw(error("$machine has not been trained."))
    end
end
