const SupervisedNetwork = Union{DeterministicNetwork,ProbabilisticNetwork}
const GenericNetwork = Union{SupervisedNetwork,UnsupervisedNetwork}

# to suppress inclusion of abstract types in models():
MMI.is_wrapper(::Type{DeterministicNetwork}) = true
MMI.is_wrapper(::Type{ProbabilisticNetwork}) = true
MMI.is_wrapper(::Type{UnsupervisedNetwork}) = true


## FALL-BACKS FOR LEARNING NETWORKS EXPORTED AS MODELS

function update(model::GenericNetwork, verb::Integer, yhat, cache, args...)

    # If any `model` field has been replaced (and not just mutated)
    # then we actually need to fit rather than update (which will
    # force build of a new learning network). If `model` has been
    # created using a learning network export macro, the test used
    # below is perfect. In any other case it is at least conservative:
    network_model_ids = objectid.(models(yhat))
    fields = [getproperty(model, name) for name in fieldnames(typeof(model))]
    submodels    = filter(f->f isa Model, fields)
    submodel_ids = objectid.(submodels)
    if !issubset(submodel_ids, network_model_ids)
        return fit(model, verb, args...)
    end

    is_anonymised = cache isa NamedTuple{(:sources, :data)}

    if is_anonymised
        sources, data = cache.sources, cache.data
        for k in eachindex(sources)
            rebind!(sources[k], data[k])
        end
    end

    fit!(yhat; verbosity=verb)
    if is_anonymised
        for s in sources
            rebind!(s, nothing)
        end
    end

    return yhat, cache, nothing
end

predict(::SupervisedNetwork, fitres, Xnew)     = fitres(Xnew)

transform(::UnsupervisedNetwork, fitres, Xnew) = fitres(Xnew)

function fitted_params(yhat::Node)
    machs = machines(yhat)
    _fitted_params = NamedTuple[]
    try
        _fitted_params = [fitted_params(m) for m in machs]
    catch exception
        if exception isa UndefRefError
            error("UndefRefEror intercepted. Perhaps "*
                  "you forgot to `fit!` a machine or node?")
        else
            throw(exception)
        end
    end
    fitted_params_given_machine =
        LittleDict(machs[j] => _fitted_params[j] for j in eachindex(machs))
    return (machines=machs,
            fitted_params_given_machine=fitted_params_given_machine)
end

fitted_params(::GenericNetwork, yhat) = fitted_params(yhat)


## FOR EXPORTING LEARNING NETWORKS BY HAND

"""
    anonymize!(sources...)

Returns a named tuple `(sources=..., data=....)` whose values are the
provided source nodes and their contents respectively, and clears the
contents of those source nodes.

"""
function anonymize!(sources...)
    data = Tuple(s.data for s in sources)
    [MLJBase.rebind!(s, nothing) for s in sources]
    return (sources=sources, data=data)
end

function MLJBase.report(yhat::Node)
    machs = machines(yhat)
    reports = NamedTuple[]
    try
        reports = [report(m) for m in machs]
    catch exception
        if exception isa UndefRefError
            error("UndefRefEror intercepted. Perhaps "*
                  "you forgot to `fit!` a machine or node?")
        else
            throw(exception)
        end
    end
    report_given_machine =
        LittleDict(machs[j] => reports[j] for j in eachindex(machs))
    return (machines=machs, report_given_machine=report_given_machine)
end

# what should be returned by a fit method for an exported learning
# network:
function fitresults(yhat)
    inputs = sources(yhat, kind=:input)
    targets = sources(yhat, kind=:target)
    weights = sources(yhat, kind=:weights)

    length(inputs) == 1 ||
        error("Improperly exported supervised network does "*
              "not have a unique :input source. ")
    length(targets) < 2 ||
        error("Improperly exported network has multiple :target sources. ")
    length(weights) < 2 ||
        error("Improperly exported network has multiple :weights sources. ")

    cache = anonymize!(vcat(inputs, targets, weights)...)

    r = report(yhat)

    return yhat, cache, r
end


