## THE SURROGATE MODEL EXTENSION TO COMPOSITE TYPES

#  `CompositeMachine` defined in this file is a type for special
#  machines associated with learning networks, which do *not* require
#  the network to be exported. With an exported model type undefined, they
#  are instead bound to the sole instance of `SurrogateNetwork`, a
#  type created purely for this purpose.

struct SurrogateNetwork <: Model end

const SupervisedNetwork = Union{DeterministicNetwork,ProbabilisticNetwork}
const GenericNetwork =
    Union{SurrogateNetwork,SupervisedNetwork,UnsupervisedNetwork}

# to suppress inclusion of these types in models():
MMI.is_wrapper(::Type{DeterministicNetwork}) = true
MMI.is_wrapper(::Type{ProbabilisticNetwork}) = true
MMI.is_wrapper(::Type{UnsupervisedNetwork}) = true
MMI.is_wrapper(::Type{SurrogateNetwork}) = true


## REPORTS AND FITRESULTS FOR NODES

# Exposed but not intended for public use

function report(N::Node)
    machs = machines(N)
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
report(::Source) = NamedTuple()

function MLJModelInterface.fitted_params(N::Node)
    machs = machines(N)
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
MLJModelInterface.fitted_params(S::Source) = NamedTuple()


## COMPOSITE MACHINES

# Machine like object for learning networks prior to export

mutable struct CompositeMachine{M<:SurrogateNetwork,P} <: AbstractMachine{M}
    model::M
    fitresult_in_waiting::P  # eg, (predict=yhat, )
    lub                      # least upper bound node
    fitresult
    cache
    report
    function CompositeMachine(fitresult_in_waiting::P) where P
        lub = tup(values(fitresult_in_waiting)...)
        return new{SurrogateNetwork,P}(SurrogateNetwork(),
                                       fitresult_in_waiting,
                                       lub)
    end
end

"""
    anonymize!(sources)

Returns a named tuple `(sources=..., data=....)` whose values are the
provided source nodes and their contents respectively, and clears the
contents of those source nodes.

"""

function anonymize!(sources)
    data = Tuple(s.data for s in sources)
    [MLJBase.rebind!(s, nothing) for s in sources]
    return (sources=sources, data=data)
end

"""
    machine(oper1=node1, oper2=node2, ...)

Construct a special type of machine, of type `CompositeMachine`,
representing a learning network before export. The keyword arguments
declare what nodes are called when operations (`predict`, `transform`,
etc) are called on the machine.

In addition to the operations named in the constructor, the following
ordinay machine methods extend to `CompositeMachine` instances:
`fit!`, `report`, and `fitted_params`.

`CompositeMachine` instances are used to construct `fit` return values
when exporting learning networks as stand-alone models "by hand". See
the MLJ manual for details.


### Examples

Supposing a supervised learning network's final predictions are
obtained by calling a node `yhat`, then the following code

```julia
mach = machine(predict=yhat)
fit!(mach; rows=train)
predictions = predict(mach, Xnew) # `Xnew` concrete data
```
are equivalent to

```julia
fit!(yhat, rows=train)
predictions = yhat(Xnew)
```

In a unsupervised learning network that performs clustering, for
example,in which node `Xout` delivers the output of dimension
reduction and `yhat` class labels, one can write

```julia
mach = machine(transform=Xout, predict=yhat)
fit!(mach)
transformed = transform(mach, Xnew) # `Xnew` concrete data
predictions = predict(mach, Xnew)
```

which is equivalent to

```julia
fit!(Xout)
fit!(yhat)
transformed = Xout(Xnew)
predictions = yhat(Xnew)
```

"""
machine(; pair_itr...) = CompositeMachine((; pair_itr...))

function fit!(mach::CompositeMachine; kwargs...)
    fit!(mach.lub; kwargs...)
    ss = sources(mach.lub) |> unique
    mach.fitresult = mach.fitresult_in_waiting
    mach.cache = anonymize!(ss)
    mach.report = report(mach.lub)
    return mach
end

MLJModelInterface.fitted_params(mach::CompositeMachine) =
    fitted_params(mach.lub)


## EXPORTING LEARNING NETWORKS BY HAND

function fitresults!(; kwargs...)
    mach =  machine(; kwargs...) |> fit!
    return mach.fitresult, mach.cache, mach.report
end

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


## FALL-BACK METHODS FOR LEARNING NETWORKS EXPORTED AS MODELS

# *Note.* Be sure to read Note 4 in src/operations.jl to see see how
# fallbacks are provided for operations acting on GeneralNetwork models.

fitted_params(::GenericNetwork, fitresult::NamedTuple) =
    fitted_params(tup(values(fitresult)...))

# legacy code:
fitted_params(::GenericNetwork, fitresult::Node) = fitted_params(fitresult)

function update(model::GenericNetwork,
                verb::Integer,
                fitresult::NamedTuple,
                cache,
                args...)

    # If any `model` field has been replaced (and not just mutated)
    # then we actually need to fit rather than update (which will
    # force build of a new learning network). If `model` has been
    # created using a learning network export macro, the test used
    # below is perfect. In any other case it is at least conservative.

    # least upper bound of all nodes delivering predictions:
    lub = tup(values(fitresult)...)

    network_model_ids = objectid.(models(lub))
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

    fit!(lub; verbosity=verb)
    if is_anonymized
        for s in sources
            rebind!(s, nothing)
        end
    end

    return fitresult, cache, report(lub)
end

# legacy version of above (private) method:
function update(model::GenericNetwork, verb::Integer,
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
predict(::SupervisedNetwork, fitresult::Node, Xnew)     = fitresult(Xnew)

# legacy method (replacements defined in operations.jl):
transform(::UnsupervisedNetwork, fitresult::Node, Xnew) = fitresult(Xnew)
