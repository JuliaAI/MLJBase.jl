# See network_composite_types.jl for type definitions

caches_data_by_default(::Type{<:NetworkComposite}) = false


# # PREFIT STUB

"""
    MLJBase.prefit(model, verbosity, data...)

Returns a learning network interface (see below) for a learning network with source nodes
that wrap `data`.

A user overloads `MLJBase.prefit` when exporting a learning network as a new stand-alone
model type, of which `model` above will be an instance. See the MLJ reference manual for
details.

$DOC_NETWORK_INTERFACES

"""
function prefit end

function MLJModelInterface.fit(composite::NetworkComposite, verbosity, data...)

    # fitresult is the signature of a learning network:
    fitresult = prefit(composite, verbosity, data...) |> MLJBase.Signature

    # train the network:
    greatest_lower_bound = MLJBase.glb(fitresult)
    fit!(greatest_lower_bound; verbosity, composite)

    report = MLJBase.report(fitresult)

    # for passing to `update` so changes in `composite` can be detected:
    cache = deepcopy(composite)

    return fitresult, cache, report

end

"""
    start_over(composite, old_composite, greatest_lower_bound)

**Private method.**

Return `true` if and only if `old_composite` and `composite` differ in the value of a
property that is *not* also the name of a (symbolic) model in the network with specified
`greates_lower_bound` (a "non-model" hyperparameter).

"""
function start_over(composite, old_composite, greatest_lower_bound)
    model_fields = MLJBase.models(greatest_lower_bound)
    any(propertynames(composite)) do field
        field in model_fields && return false
        old_value = getproperty(old_composite, field)
        value = getproperty(composite, field)
        value != old_value
    end
end

function MLJModelInterface.update(
    composite::NetworkComposite,
    verbosity,
    fitresult,
    old_composite,
    data...,
)
    greatest_lower_bound = MLJBase.glb(fitresult)

    start_over = MLJBase.start_over(composite, old_composite, greatest_lower_bound)
    start_over && return MLJModelInterface.fit(composite, verbosity, data...)

    # retrain the network:
    fit!(greatest_lower_bound; verbosity, composite)

    report = MLJBase.report(fitresult)

    # for passing to `update` so changes in `composite` can be detected:
    cache = deepcopy(composite)

    return fitresult, cache, report
end

function MLJModelInterface.fitted_params(composite::NetworkComposite, signature)
    greatest_lower_bound = glb(signature)
    d = machines_given_model(greatest_lower_bound)
    tuple_keyed_on_model(fitted_params, d)
end

MLJModelInterface.reporting_operations(::Type{<:NetworkComposite}) = OPERATIONS

# TODO implement `save` and `restore`
