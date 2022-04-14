## FALL-BACK METHODS FOR COMPOSITE MODELS (EXPORTED LEARNING NETWORKS)

# *Note.* Be sure to read Note 4 in src/operations.jl to see see how
# fallbacks are provided for operations acting on Composite models.

caches_data_by_default(::Type{<:Composite}) = true

# builds on `fitted_params(::CompositeFitresult)` defined in
# composition/learning_networks/machines.jl:
fitted_params(::Union{Composite,Surrogate}, fitresult::CompositeFitresult) =
    fitted_params(glb(fitresult))

"""
    update(model::M,
            verbosity::Integer,
            fitresult::CompositeFitresult,
            cache,
            args...) where M <: Composite

This method is an almost duplicate of `update_(mach::Machine{<:Composite}, resampled_data...; verbosity=0, kwargs...)`
The only reason it exists is to allow for a user to update a composite model without using a machine.
"""
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

    network_model_names = getfield(fitresult, :network_model_names)
    old_model = cache.old_model

    glb_node = glb(fitresult) # greatest lower bound

    if fallback(model, old_model, network_model_names, glb_node)
        return fit(model, verbosity, args...)
    else
        return update_from_glb(glb_node, model, verbosity, fitresult, cache)
    end
end

function update_from_glb(glb_node, model, verbosity, fitresult, cache)
    # return data to source nodes for fitting:
    sources, data = cache.sources, cache.data
    for k in eachindex(sources)
        rebind!(sources[k], data[k])
    end

    fit!(glb_node; verbosity=verbosity)
    # Retrieve additional report values
    report_additions_ = _call(_report_part(signature(fitresult)))

    # anonymize data again:
    for s in sources
        rebind!(s, nothing)
    end

    # record current model state:
    cache = (sources=cache.sources,
                data=cache.data,
                old_model = deepcopy(model))
    
    return (fitresult,
            cache,
            merge(report(glb_node), report_additions_))
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
