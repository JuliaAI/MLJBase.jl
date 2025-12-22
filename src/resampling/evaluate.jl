# ---------------------------------------------------------------
# Helpers

function machine_and_tag(mach::Machine)
    tag = string(name(mach.model), "-", rand(100:999))
    return mach, tag
end
machine_and_tag(pair::Pair{String,<:Machine}) = last(pair), first(pair)

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

In place of `mach`, one can use `tag_string => mach`, or a vector of either of these forms,
to return a vector of performance evaluation objects.

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

# Examples

Setup:

```julia
using MLJ
X, y = make_moons(rng=123) # a table and a vector
model = ConstantClassifier()
mach = machine(model, X, y)
```

Perform a simple evaluation on a holdout set, against accuracy and area under the ROC
curve:

```julia
evaluate!(mach; resampling=Holdout(fraction_train=0.8), measure=[accuracy, auc])
```

Perform Monte Carlo cross-validation, with 2 folds and 5 repeats, against area Brier score:

```julia
evaluate!(mach; resampling=CV(nfolds=2, rng=123), repeats=5,  measures=brier_score)
```

Evaluate on explicitly specified train-test pairs, against cross entropy, and tag the
result, "explicit folds":

```julia
e = evaluate!(
    "explicit folds" => mach;
    resampling=[(1:140, 141:150), (11:150, 1:10)],
    measure=log_loss,
)
show(e)
# PerformanceEvaluation("explicit folds", 0.708 ± 0.0328)
```

Evaluate multiple machines:

```julia
@load KNNClassifier pkg=NearestNeighborModels
mach1 = machine(ConstantClassifier(), X, y)
mach2 = machine(KNNClassifier(), X , y)
evaluate!(["const" => mach1, "knn" => mach2])
# 2-element Vector{...}
#  PerformanceEvaluation("const", 0.698 ± 0.0062)
#  PerformanceEvaluation("knn", 2.22e-16 ± 0.0)
```

See also [`evaluate`](@ref), [`PerformanceEvaluation`](@ref),
[`CompactPerformanceEvaluation`](@ref).

"""
function evaluate!(
    mach_or_pair::Union{Machine,Pair{String,<:Machine}};
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

    mach, tag = machine_and_tag(mach_or_pair)

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
        tag,
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

# multiple machine evaluations:
evaluate!(
    machines_or_pairs::AbstractVector{<:Union{Machine,Pair{String,<:Machine}}};
    kwargs...,
) = [evaluate!(x; kwargs...) for x in machines_or_pairs]

"""
    evaluate(model, data...; cache=true, options...)

Equivalent to `evaluate!(machine(model, data..., cache=cache); options...)`.
See the machine version `evaluate!` for the complete list of options.

Returns a  [`PerformanceEvaluation`](@ref) object.

In place of `model`, one can use `tag_string => model`, or a vector of either of these
forms, to return a vector of performance evaluation objects.

# Examples

Setup:

```julia
using MLJ
X, y = make_moons(rng=123) # a table and a vector
model = ConstantClassifier()
```

Perform a simple evaluation on a holdout set, against accuracy and area under the ROC
curve:

```julia
evaluate(model, X, y; resampling=Holdout(fraction_train=0.8), measure=[accuracy, auc])
```

Perform Monte Carlo cross-validation, with 2 folds and 5 repeats, against area Brier score:

```julia
evaluate(model, X, y; resampling=CV(nfolds=2, rng=123), repeats=5,  measures=brier_score)
```

Evaluate on explicitly specified train-test pairs, against cross entropy, and tag the
result, "explicit folds":

```julia
e = evaluate(
    "explicit folds" => model, X, y;
    resampling=[(1:140, 141:150), (11:150, 1:10)],
    measure=log_loss,
)
show(e)
# PerformanceEvaluation("explicit folds", 0.708 ± 0.0328)
```

Evaluate muliple models:

```julia
@load KNNClassifier pkg=NearestNeighborModels
evaluate(["const" => ConstantClassifier(), "knn" => KNNClassifier()], X , y)
# 2-element Vector{...}
#  PerformanceEvaluation("const", 0.698 ± 0.0062)
#  PerformanceEvaluation("knn", 2.22e-16 ± 0.0)
```

See also [`evaluate!`](@ref).

"""
evaluate(model::Model, args...; cache=true, kwargs...) =
    evaluate!(machine(model, args...; cache=cache); kwargs...)
evaluate(pair::Pair{String,<:Model}, args...; cache=true, kwargs...) =
    evaluate!(first(pair) => machine(last(pair), args...; cache=cache); kwargs...)

# multiple model evaluations:
evaluate(
    models_or_pairs::AbstractVector{<:Union{Model,Pair{String,<:Model}}}, args...;
    kwargs...,
) = [evaluate(x, args...; kwargs...) for x in models_or_pairs]

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

const SE_FACTOR = 1.96 # For a 95% confidence interval.

function radius_95(v::AbstractVector{<:Real})
    length(v) < 2 && return Inf
    return SE_FACTOR*std(v) / sqrt(length(v) - 1)
end
radius_95(v) = nothing

# Evaluation when `resampling` is a TrainTestPairs (CORE EVALUATOR):
function evaluate!(
    mach::Machine,
    tag,
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

    # The following is a vector with `nothing` values for each measure not returning
    # `Real` values (e.g., confmat), and `Inf` values for any other kind of measure when
    # there is only one train-test fold.
    uncertainty_radius_95 = map(radius_95, per_fold)

    evaluation = PerformanceEvaluation(
        mach.model,
        tag,
        measures,
        per_measure,
        uncertainty_radius_95,
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

function evaluate!(
    mach::Machine,
    tag,
    resampling::ResamplingStrategy,
    weights,
    class_weights,
    rows,
    verbosity,
    repeats,
    args...,
    )

    train_args = Tuple(a() for a in mach.args)
    y = train_args[2]

    _rows = actual_rows(rows, nrows(y), verbosity)

    repeated_train_test_pairs =
        vcat(
            [train_test_pairs(resampling, _rows, train_args...) for i in 1:repeats]...
        )

    evaluate!(
        mach,
        tag,
        repeated_train_test_pairs,
        weights,
        class_weights,
        nothing,
        verbosity,
        repeats,
        args...
    )
end
