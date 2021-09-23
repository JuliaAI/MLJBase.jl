## FALL-BACK METHODS FOR COMPOSITE MODELS (EXPORTED LEARNING NETWORKS)

# *Note.* Be sure to read Note 4 in src/operations.jl to see see how
# fallbacks are provided for operations acting on Composite models.

caches_data_by_default(::Type{<:Composite}) = true

# builds on `fitted_params(::NamedTuple)` defined in
# composition/learning_networks/machines.jl:
fitted_params(::Union{Composite,Surrogate}, fitresult::CompositeFitresult) =
    fitted_params(fitresult)

# legacy code (fitresult a Node):
fitted_params(::Union{Composite,Surrogate},
              fitresult::Node) =
                  fitted_params(fitresult)

function update(model::M,
                verbosity::Integer,
                fitresult::CompositeFitresult,
                cache,
                args...) where M <: Composite

    # This method falls back to `fit` to force rebuilding the
    # underlying learning network if, since the last fit:
    #
    # (i) Any hyper-parameter associated with a model in the learning network
    #     has been replaced with a new model instance (and not merely
    #     mutated), OR

    # (ii) Any OTHER hyper-parameter has changed it's value (in the sense
    # of `==`).

    # Otherwise, a "smart" fit is carried out by calling `fit!` on a
    # greatest lower bound node for nodes in the signature of the
    # underlying learning network machine. For this it is necessary to
    # temporarily "de-anonymize" the source nodes.

    network_model_names = cache.network_model_names
    old_model = cache.old_model

    glb_node = glb(values(signature(fitresult))...) # greatest lower bound

    if fallback(model, old_model, network_model_names, glb_node)
        return fit(model, verbosity, args...)
    end

    # return data to source nodes for fitting:
    sources, data = cache.sources, cache.data
    for k in eachindex(sources)
        rebind!(sources[k], data[k])
    end

    fit!(glb_node; verbosity=verbosity)
    update!(fitresult) # rebuilds the fitted_params

    # anonymize data again:
    for s in sources
        rebind!(s, nothing)
    end

    # record current model state:
    cache = (sources=cache.sources,
             data=cache.data,
             network_model_names=cache.network_model_names,
             old_model = deepcopy(model))

    return fitresult, cache, report(glb_node)

end

# helper for preceding method (where logic is explained):
function fallback(model::M, old_model, network_model_names, glb_node) where M
    # check the hyper-parameters corresponding to models:
    network_models = MLJBase.models(glb_node)
    for j in eachindex(network_models)
        name = network_model_names[j]
        name === nothing ||
            objectid(network_models[j])===objectid(getproperty(model, name)) ||
            return true
    end
    # check any other hyper-parameter:
    for name in propertynames(model)
        if !(name in network_model_names)
            old_value = getproperty(old_model, name)
            value = getproperty(model, name)
            value == old_value || return true
        end
    end
    return false
end

# legacy version of above (fitresult a Node):
function update(model::Composite,
                verbosity::Integer,
                yhat::Node,
                cache,
                args...)

    # If any `model` hyper-parameter has been replaced (and not just
    # mutated) then we actually need to fit rather than update (which
    # will force build of a new learning network). If `model` has been
    # created using a learning network export macro, the test used
    # below is perfect. In any other case it is at least conservative:
    network_model_ids = objectid.(models(yhat))
    fields = [getproperty(model, name) for name in propertynames(model)]
    submodels    = filter(f->f isa Model, fields)
    submodel_ids = objectid.(submodels)
    if !issubset(submodel_ids, network_model_ids)
        return fit(model, verbosity, args...)
    end

    is_anonymized = cache isa NamedTuple{(:sources, :data)}

    if is_anonymized
        sources, data = cache.sources, cache.data
        for k in eachindex(sources)
            rebind!(sources[k], data[k])
        end
    end

    fit!(yhat; verbosity=verbosity)
    if is_anonymized
        for s in sources
            rebind!(s, nothing)
        end
    end

    return yhat, cache, report(yhat)
end

# legacy method (replacements defined in operations.jl):
const _SC = Union{DeterministicComposite,
                  ProbabilisticComposite,
                  JointProbabilisticComposite,
                  IntervalComposite}
predict(::_SC, fitresult::Node, Xnew) = fitresult(Xnew)

# legacy method (replacements defined in operations.jl):
transform(::UnsupervisedComposite, fitresult::Node, Xnew) = fitresult(Xnew)
