## FALL-BACK METHODS FOR COMPOSITE MODELS (EXPORTED LEARNING NETWORKS)

# *Note.* Be sure to read Note 4 in src/operations.jl to see see how
# fallbacks are provided for operations acting on Composite models.

caches_data_by_default(::Type{<:Composite}) = true

# builds on `fitted_params(::CompositeFitresult)` defined in
# composition/learning_networks/machines.jl:
fitted_params(::Union{Composite,Surrogate}, fitresult::CompositeFitresult) =
    fitted_params(glb(fitresult))

function update(model::M,
                verbosity::Integer,
                fitresult::CompositeFitresult,
                cache,
                args...) where M <: Composite

    # This method falls back to `fit` to force rebuilding of the
    # underlying learning network if, since the last fit:
    #
    # (i) Any hyper-parameter of `model` that has, as a value, a model in the network, has
    #     been replaced with a new value (and not merely mutated), OR

    # (ii) Any OTHER hyper-parameter has changed it's value (in the sense
    # of `==`).

    # Otherwise, a "smart" fit is carried out by calling `fit!` on a
    # greatest lower bound node for nodes in the signature of the
    # underlying learning network machine.

    network_model_names = getfield(fitresult, :network_model_names)

    old_model = cache.old_model
    glb = MLJBase.glb(fitresult) # greatest lower bound of network, a node

    if fallback(model, old_model, network_model_names, glb)
        return fit(model, verbosity, args...)
    end

    fit!(glb; verbosity=verbosity)

    # Retrieve additional report values
    report = MLJBase.report(fitresult)

    # record current model state:
    cache = (; old_model = deepcopy(model))

    return (fitresult,
            cache,
            report)

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
