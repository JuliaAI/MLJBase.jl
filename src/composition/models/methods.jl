## LEGACY METHODS FOR EXPORTING LEARNING NETWORKS BY HAND

# legacy method:
function anonymize!(sources)
    data = Tuple(s.data for s in sources)
    [MLJBase.rebind!(s, nothing) for s in sources]
    return (sources=sources, data=data)
end

# legacy method:
function fitresults(yhat::AbstractNode)
    Base.depwarn("`fitresults(::Node)` is deprecated. "*
                 "Query `return!` for "*
                 "preferred way to export learning networks "*
                 "by hand. ", Base.Core.Typeof(fitresults).name.mt.name)
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

fitted_params(::Union{Composite,Surrogate},
              fitresult::NamedTuple) =
                  fitted_params(glb(values(fitresult)...))

# legacy code (fitresult a Node):
fitted_params(::Union{Composite,Surrogate},
              fitresult::Node) =
                  fitted_params(fitresult)

function update(model::Composite,
                verbosity::Integer,
                fitresult::NamedTuple,
                cache,
                args...)

    # We must rebuild the network before fitting it whenever a field
    # `fld` of `model` becomes *active*, meaning not *passive*. By
    # definition, `fld` is *passive* if:
    #
    # (1) It's `objectid` coincides with that of a model instance in
    #     the learning network, OR
    # (2) It's value is == to 

    # If any `model` field has been replaced (and not just mutated)
    # then we actually need to fit rather than update to force
    # building a *new* learning network (the existing learning network
    # still points to the old models). Similarly, if there is a field
    # that is *not* a model that changes then, we have no choice but
    # to rebuild the network to ensure it is incorporated (the
    # existing learning network cannot see the change). There
    # is no way to retain "smart fitting" in these cases.

    glb_node = glb(values(fitresult)...) # greatest lower bound in complete DAG

    network_model_ids = objectid.(models(glb_node))
    field_values =
        [getproperty(model, name) for name in fieldnames(typeof(model))]
    submodels    = filter(f->f isa Model, field_values)
    submodel_ids = objectid.(submodels)
    if !issubset(submodel_ids, network_model_ids)
        return fit(model, verbosity, args...)
    end

    # return data to source nodes for fitting:
    sources, data = cache.sources, cache.data
    for k in eachindex(sources)
        rebind!(sources[k], data[k])
    end

    fit!(glb_node; verbosity=verbosity)

    # anonymize data again:
    for s in sources
        rebind!(s, nothing)
    end


    return fitresult, cache, report(glb_node)
end

# legacy version of above (fitresult a Node):
function update(model::Composite,
                verbosity::Integer,
                yhat::Node,
                cache,
                args...)

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
predict(::SupervisedComposite, fitresult::Node, Xnew)     = fitresult(Xnew)

# legacy method (replacements defined in operations.jl):
transform(::UnsupervisedComposite, fitresult::Node, Xnew) = fitresult(Xnew)
