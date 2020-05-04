## LEGACY METHODS FOR EXPORTING LEARNING NETWORKS BY HAND

# needed?
# function fitresults!(; kwargs...)
#     mach =  machine(; kwargs...) |> fit!
#     return mach.fitresult, mach.cache, mach.report
# end

# legacy method:
function fitresults(yhat::AbstractNode)
    # Base.depwarn("`fitresults(yhat::Node)` is deprecated. "*
    #              "See \"Composing Models\" section of MLJ manual "*
    #              "on preferred way to export learning networks "*
    #              "by hand. ", Base.Core.Typeof(fitresults).name.mt.name)
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

    cache = anonymize!(vcat(inputs, targets, weights))

    rep = report(yhat)

    return yhat, cache, rep
end


## FALL-BACK METHODS FOR COMPOSITE MODELS (EXPORTED LEARNING NETWORKS)

# *Note.* Be sure to read Note 4 in src/operations.jl to see see how
# fallbacks are provided for operations acting on Composite models.

fitted_params(::Composite, fitresult::NamedTuple) =
    fitted_params(glb(values(fitresult)...))

# legacy code:
fitted_params(::Composite, fitresult::Node) = fitted_params(fitresult)

function update(model::Composite,
                verb::Integer,
                fitresult::NamedTuple,
                cache,
                args...)

    # If any `model` field has been replaced (and not just mutated)
    # then we actually need to fit rather than update (which will
    # force build of a new learning network). If `model` has been
    # created using a learning network export macro, the test used
    # below is perfect. In any other case it is at least conservative.

    # greatest lower bound of all nodes delivering predictions:
    glb_node = glb(values(fitresult)...)

    network_model_ids = objectid.(models(glb_node))
    field_values =
        [getproperty(model, name) for name in fieldnames(typeof(model))]
    submodels    = filter(f->f isa Model, field_values)
    submodel_ids = objectid.(submodels)
    if !issubset(submodel_ids, network_model_ids)
        return fit(model, verb, args...)
    end

    is_anonymized = cache isa NamedTuple{(:sources, :data)}

    if is_anonymized
        sources, data = cache.sources, cache.data
        for k in eachindex(sources)
            rebind!(sources[k], data[k])
        end
    end

    fit!(glb_node; verbosity=verb)
    if is_anonymized
        for s in sources
            rebind!(s, nothing)
        end
    end

    return fitresult, cache, report(glb_node)
end

# legacy version of above (private) method:
function update(model::Composite, verb::Integer,
                yhat::Node, cache, args...)

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

    is_anonymized = cache isa NamedTuple{(:sources, :data)}

    if is_anonymized
        sources, data = cache.sources, cache.data
        for k in eachindex(sources)
            rebind!(sources[k], data[k])
        end
    end

    fit!(yhat; verbosity=verb)
    if is_anonymized
        for s in sources
            rebind!(s, nothing)
        end
    end

    return yhat, cache, report(yhat)
end

# legacy method (replacements defined in operations.jl):
predict(::SupervisedComposite, fitresult::Node, Xnew)     = fitresult(Xnew)

# legacy method (replacements defined in operations.jl):
transform(::UnsupervisedComposite, fitresult::Node, Xnew) = fitresult(Xnew)
