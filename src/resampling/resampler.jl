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
    tag = ""

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
        tag,
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
    tag = ""
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
        tag,
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
