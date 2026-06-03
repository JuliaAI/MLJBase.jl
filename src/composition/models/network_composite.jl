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
    acceleration = MLJBase.acceleration(fitresult)
    fit!(greatest_lower_bound; verbosity, composite, acceleration)

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

# Called by `fit_only!` when condition (4) (row change) fires on a composite with frozen
# descendants. Rebuilds the network on the new data via `prefit`, then transfers trained
# state from old inner machines to the new ones for any symbol whose resolved model is
# frozen. Non-frozen children stay at `state == 0` in the new network and so receive a
# fresh fit on the new sources. Frozen children short-circuit via the A/B/C checks in
# `fit_only!` because the transferred state gives them `age >= 1`.
function update_for_row_change(
    composite::NetworkComposite,
    verbosity,
    fitresult,
    cache,
    data...,
)
    old_glb = MLJBase.glb(fitresult)
    old_machs = MLJBase.machines_given_model(old_glb)

    new_fitresult = prefit(composite, verbosity, data...) |> MLJBase.Signature
    new_glb = MLJBase.glb(new_fitresult)
    new_machs = MLJBase.machines_given_model(new_glb)

    for (sym, old_ms) in old_machs
        sym in propertynames(composite) || continue
        frozen(getproperty(composite, sym)) || continue
        haskey(new_machs, sym) || continue
        new_ms = new_machs[sym]
        length(new_ms) == length(old_ms) || continue
        atomic_model = getproperty(composite, sym)
        for (old_m, new_m) in zip(old_ms, new_ms)
            isdefined(old_m, :fitresult) || continue
            new_m.fitresult = old_m.fitresult
            isdefined(old_m, :cache) && (new_m.cache = old_m.cache)
            isdefined(old_m, :report) && (new_m.report = old_m.report)
            new_m.state = old_m.state
            new_m.old_model = deepcopy(atomic_model)
            new_m.old_upstream_state = MLJBase.upstream(new_m)
        end
    end

    fit!(new_glb; verbosity, composite)

    report = MLJBase.report(new_fitresult)
    new_cache = deepcopy(composite)
    return new_fitresult, new_cache, report
end

MLJModelInterface.fitted_params(composite::NetworkComposite, signature) =
    fitted_params(signature)

MLJModelInterface.reporting_operations(::Type{<:NetworkComposite}) = OPERATIONS

# here `fitresult` has type `Signature`.
function save(model::NetworkComposite, fitresult)
    # The network includes machines with symbolic models. These machines need to be
    # replaced by serializable versions, but we cannot naively use `serializable(mach)`,
    # because the absence of the concrete model means this just returns `mach` (because
    # `save(::Symbol, fitresult)` returns `fitresult`). We need to use the special
    # `serialiable(mach, model)` instead. This is what `replace` below does, because we
    # pass it the flag `serializable=true` but we must also pass `symbol =>
    # concrete_model` replacements, which we calculate first:

    greatest_lower_bound = MLJBase.glb(fitresult)
    machines_given_model = MLJBase.machines_given_model(greatest_lower_bound)
    atomic_models = keys(machines_given_model)
    pairs = [atom => getproperty(model, atom) for atom in atomic_models]

    replace(fitresult, pairs...; serializable=true)
end

function MLJModelInterface.restore(model::NetworkComposite, serializable_fitresult)
    greatest_lower_bound = MLJBase.glb(serializable_fitresult)
    machines_given_model = MLJBase.machines_given_model(greatest_lower_bound)
    atomic_models = keys(machines_given_model)

    # the following indirectly mutates `serialiable_fiteresult`, returning it to
    # usefulness:
    for atom in atomic_models
        for mach in machines_given_model[atom]
            mach.fitresult = MLJBase.restore(getproperty(model, atom), mach.fitresult)
            mach.state = 1
        end
    end
    return serializable_fitresult
end
