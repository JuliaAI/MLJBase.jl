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

# Keyword Constructor
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
        nfolds > 1 || error("Must have nfolds > 1. ")
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
    n > 0 || error("Inusufficient data for $n_folds-fold cross-validation.\n"*
                   "Try reducing nfolds. ")

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
        nfolds > 1 || error("Must have nfolds > 1. ")
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
    st <: AbstractArray{<:Finite} ||
        error("Supplied target has scitpye $st but stratified "*
              "cross-validation applies only to classification problems. ")

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
                                          :per_fold,
                                          :per_observation,
                                          :fitted_params_per_fold,
                                          :report_per_fold)}

# pretty printing:
round3(x) = x
round3(x::AbstractFloat) = round(x, sigdigits=3)

_short(v) = v
_short(v::Vector{<:Real}) = MLJBase.short_string(v)
_short(v::Vector) = string("[", join(_short.(v), ", "), "]")
_short(::Missing) = missing

function Base.show(io::IO, ::MIME"text/plain", e::PerformanceEvaluation)
    data = hcat(e.measure, round3.(e.measurement),
                [round3.(v) for v in e.per_fold])
    header = ["_.measure", "_.measurement", "_.per_fold"]
    PrettyTables.pretty_table(io, data, header;
                              header_crayon=PrettyTables.Crayon(bold=false),
                              alignment=:l)
    println(io, "_.per_observation = $(_short(e.per_observation))")
    println(io, "_.fitted_params_per_fold = [ … ]")
    println(io, "_.report_per_fold = [ … ]")
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
    unspecified_rows ||
        @info "Creating subsamples from a subset of all rows. "
    return _rows
end

function _check_measure(model, measure, y, operation, override)

    override && (return nothing)

    T = scitype(y)

    T == Unknown && (return nothing)
    target_scitype(measure) == Unknown && (return nothing)
    prediction_type(measure) == :unknown && (return nothing)

    avoid = "\nTo override measure checks, set check_measure=false. "

    T <: target_scitype(measure) ||
        throw(ArgumentError(
            "\nscitype of target = $T but target_scitype($measure) = "*
            "$(target_scitype(measure))."*avoid))

    if model isa Probabilistic
        if operation == predict
            if prediction_type(measure) != :probabilistic
                suggestion = ""
                if target_scitype(measure) <: Finite
                    suggestion = "\nPerhaps you want to set operation="*
                    "predict_mode. "
                elseif target_scitype(measure) <: Continuous
                    suggestion = "\nPerhaps you want to set operation="*
                    "predict_mean or operation=predict_median. "
                else
                    suggestion = ""
                end
                throw(ArgumentError(
                   "\n$model <: Probabilistic but prediction_type($measure) = "*
                      ":$(prediction_type(measure)). "*suggestion*avoid))
            end
        end
    end

    model isa Deterministic && prediction_type(measure) != :deterministic &&
        throw(ArgumentError("$model <: Deterministic but "*
                            "prediction_type($measure) ="*
              ":$(prediction_type(measure))."*avoid))

    return nothing

end

function _process_weights_measures(weights, measures, mach,
                                   operation, verbosity, check_measure)

    if measures === nothing
        candidate = default_measure(mach.model)
        candidate ===  nothing && error("You need to specify measure=... ")
        _measures = [candidate, ]
    elseif !(measures isa AbstractVector)
        _measures = [measures, ]
    else
        _measures = measures
    end

    y = mach.args[2]()

    [ _check_measure(mach.model, m, y, operation, !check_measure)
     for m in _measures ]

    if weights != nothing
        weights isa AbstractVector{<:Real} ||
            throw(ArgumentError("`weights` must be a `Real` vector."))
        length(weights) == nrows(y) ||
            throw(DimensionMismatch("`weights` and target "*
                                    "have different lengths. "))
        _weights = weights
    elseif  length(mach.args) == 3
        verbosity < 1 ||
            @info "Passing machine sample weights to any supported measures. "
        _weights = mach.args[3]()
    else
        _weights = weights
    end

    return _weights, _measures

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
            throw(error("Can't create $(acceleration.settings) tasks)"))
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
              operation=predict,
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

The resampling strategy is applied repeatedly (Monte Carlo resampling)
if `repeats > 1`. For example, if `repeats = 10`, then `resampling =
CV(nfolds=5, shuffle=true)`, generates a total of 50 `(train, test)`
pairs for evaluation and subsequent aggregation.

If `resampling isa MLJ.ResamplingStrategy` then one may optionally
restrict the data used in evaluation by specifying `rows`.

An optional `weights` vector may be passed for measures that support
sample weights (`MLJ.supports_weights(measure) == true`), which is
ignored by those that don't.

*Important:* If `mach` already wraps sample weights `w` (as in `mach =
machine(model, X, y, w)`) then these weights, which are used for
*training*, are automatically passed to the measures for
evaluation. However, for evaluation purposes, any `weights` specified
as a keyword argument will take precedence over `w`.

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

- `weights` - per-sample weights for training and measures; see
  important note above

- `operation` - `predict`, `predict_mean`, `predict_mode` or
  `predict_median`; `predict` is the default but cannot be used with a
  deterministic measure if `model isa Probabilistic`

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

- `measurements`: the corresponding measurements, aggregated across the
  test folds using the aggregation method defined for each measure (do
  `aggregation(measure)` to inspect)

- `per_fold`: a vector of vectors of individual test fold evaluations
  (one vector per measure)

- `per_observation`: a vector of vectors of individual observation
  evaluations of those measures for which
  `reports_each_observation(measure)` is true, which is otherwise
  reported `missing`.

See also [`evaluate`](@ref)

"""
function evaluate!(mach::Machine{<:Supervised};
                   resampling=CV(),
                   measures=nothing,
                   measure=measures,
                   weights=nothing,
                   operation=predict,
                   acceleration=default_resource(),
                   rows=nothing,
                   repeats=1,
                   force=false,
                   check_measure=true,
                   verbosity=1)

    # this method just checks validity of options, preprocess the
    # weights and measures, and dispatches a strategy-specific
    # `evaluate!`

    repeats > 0 || error("Need n > 0. ")

    if resampling isa TrainTestPairs
        if rows !== nothing
            error("You cannot specify `rows` unless `resampling "*
                  "isa MLJ.ResamplingStrategy` is true. ")
        end
        if repeats != 1 && verbosity > 0
            @warn "repeats > 1 not supported unless "*
            "`resampling<:ResamplingStrategy. "
        end
    end

    _weights, _measures =
        _process_weights_measures(weights, measure, mach,
                                  operation, verbosity, check_measure)

    if verbosity >= 0 && weights !== nothing
        unsupported = filter(_measures) do m
            !supports_weights(m)
        end
        if !isempty(unsupported)
            unsupported_as_string = string(unsupported[1])
            unsupported_as_string *=
                reduce(*, [string(", ", m) for m in unsupported[2:end]])
                @warn "Sample weights ignored in evaluations of the following"*
            " measures, as unsupported: \n$unsupported_as_string "
        end
    end

    _acceleration= _process_accel_settings(acceleration)

    evaluate!(mach, resampling, _weights, rows, verbosity, repeats,
                   _measures, operation, _acceleration, force)

end

"""
    evaluate(model, X, y; measure=nothing, options...)
    evaluate(model, X, y, w; measure=nothing, options...)

Evaluate the performance of a supervised model `model` on input data
`X` and target `y`, optionally specifying sample weights `w` for
training, where supported. The same weights are passed to measures
that support sample weights, unless this behaviour is overridden by
explicitly specifying the option `weights=...`.

See the machine version `evaluate!` for the complete list of options.

"""
evaluate(model::Supervised, args...; kwargs...) =
    evaluate!(machine(model, args...); kwargs...)

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
        machines = [mach, [machine(mach.model, mach.args...) for
                           _ in 2:length(partitions)]...]
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

# Evaluation when resampling is a TrainTestPairs (CORE EVALUATOR):
function evaluate!(mach::Machine, resampling, weights,
                   rows, verbosity, repeats,
                   measures, operation, acceleration, force)

    # Note: `rows` and `repeats` are ignored here

    resampling isa TrainTestPairs ||
        error("`resampling` must be an "*
              "`MLJ.ResamplingStrategy` or tuple of pairs "*
              "of the form `(train_rows, test_rows)`")

    X = mach.args[1]()
    y = mach.args[2]()

    nfolds = length(resampling)

    nmeasures = length(measures)

    function fit_and_extract_on_fold(mach, k)
        train, test = resampling[k]
        fit!(mach; rows=train, verbosity=verbosity - 1, force=force)
        Xtest = selectrows(X, test)
        ytest = selectrows(y, test)
        if weights == nothing
            wtest = nothing
        else
            wtest = weights[test]
        end
        yhat = operation(mach, Xtest)

        measurements =  [value(m, yhat, Xtest, ytest, wtest)
                         for m in measures]
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
           per_fold               = per_fold,
           per_observation        = per_observation,
           fitted_params_per_fold = fitted_params_per_fold |> collect,
           report_per_fold        = report_per_fold |> collect)

    return ret

end

# ----------------------------------------------------------------
# Evaluation when `resampling` is a ResamplingStrategy

function evaluate!(mach::Machine, resampling::ResamplingStrategy,
                   weights, rows, verbosity, repeats, args...)

    train_args = Tuple(a() for a in mach.args)
    y = train_args[2]

    _rows = actual_rows(rows, length(y), verbosity)

    repeated_train_test_pairs =
        vcat([train_test_pairs(resampling, _rows, train_args...)
              for i in 1:repeats]...)

    return evaluate!(mach,
                     repeated_train_test_pairs,
                     weights,
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
                          operation=predict,
                          repeats = 1,
                          acceleration=default_resource(),
                          check_measure=true)

Resampling model wrapper, used internally by the `fit` method of
`TunedModel` instances. See [`evaluate!](@ref) for options. Not
intended for general use.

Given a machine `mach = machine(resampler, args...)` one obtains a
performance evaluation of the specified `model`, performed according
to the prescribed `resampling` strategy and other parameters, using
data `args...`, by calling `fit!(mach)` followed by
`evaluate(mach)`. The advantage over using `evaluate(model, X, y)` is
that the latter call always calls `fit` on the `model` but
`fit!(mach)` only calls `update` after the first call.

The sample `weights` are passed to the specified performance
measures that support weights for evaluation.

*Important:* If `weights` are left unspecified, then any weight vector
`w` used in constructing the resampler machine, as in
`resampler_machine = machine(resampler, X, y, w)` (which is then used
in *training* the model) will also be used in evaluation.

"""
mutable struct Resampler{S,M<:Union{Supervised,Nothing}} <: Supervised
    model::M
    resampling::S # resampling strategy
    measure
    weights::Union{Nothing,AbstractVector{<:Real}}
    operation
    acceleration::AbstractResource
    check_measure::Bool
    repeats::Int
end

MLJBase.is_wrapper(::Type{<:Resampler}) = true
MLJBase.supports_weights(::Type{<:Resampler{<:Any,M}}) where M =
    supports_weights(M)
MLJBase.is_pure_julia(::Type{<:Resampler}) = true

function MLJBase.clean!(resampler::Resampler)
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

function Resampler(; model=nothing, resampling=CV(),
            measure=nothing, weights=nothing, operation=predict,
            acceleration=default_resource(), check_measure=true, repeats=1)

    resampler = Resampler(model, resampling, measure, weights, operation,
                          acceleration, check_measure, repeats)
    message = MLJBase.clean!(resampler)
    isempty(message) || @warn message

    return resampler

end

function MLJBase.fit(resampler::Resampler, verbosity::Int, args...)

    mach = machine(resampler.model, args...)

    weights, measures =
        _process_weights_measures(resampler.weights, resampler.measure,
                                  mach, resampler.operation,
                                  verbosity, resampler.check_measure)

    _acceleration = _process_accel_settings(resampler.acceleration)

    fitresult = evaluate!(mach, resampler.resampling,
                          weights, nothing, verbosity - 1, resampler.repeats,
                          measures, resampler.operation,
                          _acceleration, false)
    cache = (mach, deepcopy(resampler.resampling))
    report = NamedTuple()

    return fitresult, cache, report

end

# in special case of non-shuffled, non-repeated holdout, we can reuse
# the underlying model's machine, provided the training_fraction has
# not changed:
function MLJBase.update(resampler::Resampler{Holdout},
                        verbosity::Int, fitresult, cache, args...)

    old_mach, old_resampling = cache

    reusable = !resampler.resampling.shuffle &&
        resampler.repeats == 1 &&
        old_resampling.fraction_train ==
        resampler.resampling.fraction_train

    if reusable
        mach = old_mach
    else
        mach = machine(resampler.model, args...)
        cache = (mach, deepcopy(resampler.resampling))
    end

    weights, measures =
        _process_weights_measures(resampler.weights, resampler.measure,
                                  mach, resampler.operation,
                                  verbosity, resampler.check_measure)

    _acceleration = _process_accel_settings(resampler.acceleration)

    mach.model = resampler.model
    fitresult = evaluate!(mach, resampler.resampling,
                          weights, nothing, verbosity - 1, resampler.repeats,
                          measures, resampler.operation,
                          _acceleration, false)


    report = NamedTuple

    return fitresult, cache, report

end

MLJBase.input_scitype(::Type{<:Resampler{S,M}}) where {S,M} =
    MLJBase.input_scitype(M)
MLJBase.target_scitype(::Type{<:Resampler{S,M}}) where {S,M} =
    MLJBase.target_scitype(M)
MLJBase.package_name(::Type{<:Resampler}) = "MLJBase"

MLJBase.load_path(::Type{<:Resampler}) = "MLJBase.Resampler"

evaluate(resampler::Resampler, fitresult) = fitresult

function evaluate(machine::Machine{<:Resampler})
    if isdefined(machine, :fitresult)
        return evaluate(machine.model, machine.fitresult)
    else
        throw(error("$machine has not been trained."))
    end
end
