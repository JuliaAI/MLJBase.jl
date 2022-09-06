# # SIGNATURES

function _operation_part(signature)
    ops = filter(in(OPERATIONS), keys(signature))
    return NamedTuple{ops}(map(op->getproperty(signature, op), ops))
end
function _report_part(signature)
    :report in keys(signature) || return NamedTuple()
    return signature.report
end

_operations(signature) = keys(_operation_part(signature))

function _nodes(signature)
    return (values(_operation_part(signature))...,
            values(_report_part(signature))...)
end

function _call(nt::NamedTuple)
    _call(n) = deepcopy(n())
    _keys = keys(nt)
    _values = values(nt)
    return NamedTuple{_keys}(_call.(_values))
end

"""
    model_supertype(signature)

Return, if this can be inferred, which of `Deterministic`,
`Probabilistic` and `Unsupervised` is the appropriate supertype for a
composite model obtained by exporting a learning network with the
specified `signature`.

$DOC_SIGNATURES

If a supertype cannot be inferred, `nothing` is returned.

If the network with given `signature` is not exportable, this method
will not error but it will not a give meaningful return value either.

**Private method.**

"""
function model_supertype(signature)

    operations = _operations(signature)

    length(intersect(operations, (:predict_mean, :predict_median))) == 1 &&
        return Deterministic

    if :predict in operations
        node = signature.predict
        if node isa Source
            return Deterministic
        end
        if node.machine !== nothing
            model = node.machine.model
            model isa Deterministic && return Deterministic
            model isa Probabilistic && return Probabilistic
        end
    end

    return nothing

end


# # FITRESULTS FOR COMPOSITE MODELS

mutable struct CompositeFitresult
    signature
    glb
    network_model_names
    function CompositeFitresult(signature)
        signature_node = glb(_nodes(signature)...)
        new(signature, signature_node)
    end
end
signature(c::CompositeFitresult) = getfield(c, :signature)
glb(c::CompositeFitresult) = getfield(c, :glb)

# To accommodate pre-existing design (operations.jl) arrange
# that `fitresult.predict` returns the predict node, etc:
Base.propertynames(c::CompositeFitresult) = keys(signature(c))
Base.getproperty(c::CompositeFitresult, name::Symbol) =
    getproperty(signature(c), name)


# # LEARNING NETWORK MACHINES

surrogate(::Type{<:Deterministic})  = Deterministic()
surrogate(::Type{<:Probabilistic})  = Probabilistic()
surrogate(::Type{<:Unsupervised}) = Unsupervised()
surrogate(::Type{<:Static}) = Static()

caches_data_by_default(::Type{<:Surrogate}) = false

const ERR_MUST_PREDICT = ArgumentError(
    "You must specify at least `predict=<some node>`. ")
const ERR_MUST_TRANSFORM = ArgumentError(
    "You must specify at least `transform=<some node>`. ")
const ERR_MUST_OPERATE = ArgumentError(
    "You must specify at least one operation, as in `predict=<some node>`. ")
const ERR_MUST_SPECIFY_SOURCES = ArgumentError(
    "You must specify at least one source `Xs`, as in "*
    "`machine(surrogate_model, Xs, ...; kwargs...)`. ")
const ERR_BAD_SIGNATURE = ArgumentError(
    "Only the following keyword arguments are supported in learning network "*
    "machine constructors: `report` or one of: `$OPERATIONS`. ")
const ERR_EXPECTED_NODE_IN_SIGNATURE = ArgumentError(
    "Learning network machine constructor syntax error. "*
    "Did not enounter `Node` in place one was expected. ")

function check_surrogate_machine(::Surrogate, signature, _sources)
    isempty(_operations(signature)) && throw(ERR_MUST_OPERATE)
    isempty(_sources) && throw(ERR_MUST_SPECIFY_SOURCES)
    return nothing
end

function check_surrogate_machine(::Union{Supervised,SupervisedAnnotator},
                                 signature,
                                 _sources)
    isempty(_operations(signature)) && throw(ERR_MUST_PREDICT)
    length(_sources) > 1 || throw(err_supervised_nargs())
    return nothing
end

function check_surrogate_machine(::Union{Unsupervised},
                                 signature,
                                 _sources)
    isempty(_operations(signature)) && throw(ERR_MUST_TRANSFORM)
    length(_sources) < 2 || throw(err_unsupervised_nargs())
    return nothing
end

function machine(model::Surrogate, _sources::Source...; pair_itr...)

    # named tuple, such as `(predict=yhat, transform=W)`:
    signature = (; pair_itr...)

    # signature checks:
    isempty(_operations(signature)) && throw(ERR_MUST_OPERATE)
    for k in keys(signature)
        if k in OPERATIONS
            getproperty(signature, k) isa AbstractNode ||
                throw(ERR_EXPECTED_NODE_IN_SIGNATURE)
        elseif k === :report
            all(v->v isa AbstractNode, values(signature.report)) ||
                throw(ERR_EXPECTED_NODE_IN_SIGNATURE)
        else
            throw(ERR_BAD_SIGNATURE)
        end
    end

    check_surrogate_machine(model, signature, _sources)

    mach = Machine(model, _sources...)

    mach.fitresult = CompositeFitresult(signature)

    return mach

end

function machine(_sources::Source...; pair_itr...)

    signature = (; pair_itr...)

    T = model_supertype(signature)
    if T == nothing
        @warn "Unable to infer surrogate model type. \n"*
            "Using Deterministic(). To override specify "*
            "surrogate model, as in "*
        "`machine(Probabilistic(), ...)` or `machine(Interval(), ...)`"
        model = Deterministic()
    else
        model = surrogate(T)
    end

    return machine(model, _sources...; pair_itr...)

end

"""
    N = glb(mach::Machine{<:Union{Composite,Surrogate}})

A greatest lower bound for the nodes appearing in the signature of
`mach`.

$DOC_SIGNATURES

**Private method.**

"""
glb(mach::Machine{<:Union{Composite,Surrogate}}) = glb(mach.fitresult)

"""
    report(fitresult::CompositeFitresult)

Return a tuple combining the report from `fitresult.glb` (a `Node` report) with the
additions coming from nodes declared as report nodes in `fitresult.signature`, but without
merging the two.

$DOC_SIGNATURES

**Private method**
"""
function report(fitresult::CompositeFitresult)
    basic = report(glb(fitresult))
    additions = _call(_report_part(signature(fitresult)))
    return (; basic, additions)
end

"""
    fit!(mach::Machine{<:Surrogate};
         rows=nothing,
         acceleration=CPU1(),
         verbosity=1,
         force=false))

Train the complete learning network wrapped by the machine `mach`.

More precisely, if `s` is the learning network signature used to
construct `mach`, then call `fit!(N)`, where `N` is a greatest lower
bound of the nodes appearing in the signature (values in the signature
that are not `AbstractNode` are ignored). For example, if `s =
(predict=yhat, transform=W)`, then call `fit!(glb(yhat, W))`.

See also [`machine`](@ref)

"""
function fit!(mach::Machine{<:Surrogate}; kwargs...)
    glb = MLJBase.glb(mach)
    fit!(glb; kwargs...)
    mach.state += 1
    mach.report = Dict{Symbol,Any}(:fit => MLJBase.report(mach.fitresult))
    return mach
end

MLJModelInterface.fitted_params(mach::Machine{<:Surrogate}) =
    fitted_params(glb(mach))


# # CONSTRUCTING THE RETURN VALUE FOR A COMPOSITE FIT METHOD

logerr_identical_models(name, model) =
    "The hyperparameters $name of "*
    "$model have identical model "*
    "instances as values. "
const ERR_IDENTICAL_MODELS = ArgumentError(
    "Two distinct hyper-parameters of a "*
    "composite model that are both "*
    "associated with models in the underlying learning "*
    "network (eg, any two components of a `@pipeline` model) "*
    "cannot have identical values, although they can be `==` "*
    "(corresponding nested properties are `==`). "*
    "Consider constructing instances "*
    "separately or use `deepcopy`. ")

# Identify which properties of `model` have, as values, a model in the
# learning network wrapped by `mach`, and check that no two such
# properties have have identical values (#377). Return the property name
# associated with each model in the network (in the order appearing in
# `models(glb(mach))`) using `nothing` when the model is not
# associated with any property.
network_model_names(model::Nothing, mach::Machine{<:Surrogate}) = nothing

function network_model_names(model::M, mach::Machine{<:Surrogate}) where M<:Model

    network_model_ids = objectid.(MLJBase.models(glb(mach)))

    names = propertynames(model)

    # intialize dict to detect duplicity a la #377:
    name_given_id = Dict{UInt64,Vector{Symbol}}()

    # identify location of properties whose values are models in the
    # learning network, and build name_given_id:
    for name in names
        id = objectid(getproperty(model, name))
        if id in network_model_ids
            if haskey(name_given_id, id)
                push!(name_given_id[id], name)
            else
                name_given_id[id] = [name,]
            end
        end
    end

    # perform #377 check:
    no_duplicates = all(values(name_given_id)) do name
        length(name) == 1
    end
    if !no_duplicates
        for (id, name) in name_given_id
            if length(name) > 1
                @error logerr_identical_models(name, model)
            end
        end
        throw(ERR_IDENTICAL_MODELS)
    end

    return map(network_model_ids) do id
        if id in keys(name_given_id)
            return name_given_id[id] |> first
        else
            return nothing
        end
    end

end


"""

    return!(mach::Machine{<:Surrogate}, model, verbosity; acceleration=CPU1())

The last call in custom code defining the `MLJBase.fit` method for a
new composite model type. Here `model` is the instance of the new type
appearing in the `MLJBase.fit` signature, while `mach` is a learning
network machine constructed using `model`. Not relevant when defining
composite models using `@pipeline` (deprecated) or `@from_network`.

For usage, see the example given below. Specifically, the call does
the following:

- Determines which hyper-parameters of `model` point to model
  instances in the learning network wrapped by `mach`, for recording
  in an object called `cache`, for passing onto the MLJ logic that
  handles smart updating (namely, an `MLJBase.update` fallback for
  composite models).

- Calls `fit!(mach, verbosity=verbosity, acceleration=acceleration)`.

- Records (among other things) a copy of `model` in a variable called `cache`

- Returns `cache` and outcomes of training in an appropriate form
  (specifically, `(mach.fitresult, cache, mach.report)`; see [Adding
  Models for General
  Use](https://alan-turing-institute.github.io/MLJ.jl/dev/adding_models_for_general_use/)
  for technical details.)


### Example

The following code defines, "by hand", a new model type `MyComposite`
for composing standardization (whitening) with a deterministic
regressor:

```
mutable struct MyComposite <: DeterministicComposite
    regressor
end

function MLJBase.fit(model::MyComposite, verbosity, X, y)
    Xs = source(X)
    ys = source(y)

    mach1 = machine(Standardizer(), Xs)
    Xwhite = transform(mach1, Xs)

    mach2 = machine(model.regressor, Xwhite, ys)
    yhat = predict(mach2, Xwhite)

    mach = machine(Deterministic(), Xs, ys; predict=yhat)
    return!(mach, model, verbosity)
end
```

"""
function return!(mach::Machine{<:Surrogate},
                 model::Union{Model,Nothing},
                 verbosity;
                 acceleration=CPU1())

    network_model_names_ = network_model_names(model, mach)

    verbosity isa Nothing || fit!(mach, verbosity=verbosity, acceleration=acceleration)
    setfield!(mach.fitresult, :network_model_names, network_model_names_)

    # record the current hyper-parameter values:
    old_model = deepcopy(model)

    glb = MLJBase.glb(mach)
    cache = (; old_model)

    return mach.fitresult, cache, report_given_method(mach)[:fit]
end



###############################################################################
#####              SAVE AND RESTORE FOR COMPOSITES                        #####
###############################################################################


# Returns a new `CompositeFitresult` that is a shallow copy of the original one.
# To do so,  we build a copy of the learning network where each machine contained
# in it needs to be called `serializable` upon.
function save(model::Composite, fitresult)
    signature = MLJBase.signature(fitresult)
    operation_nodes = values(MLJBase._operation_part(signature))
    report_nodes = values(MLJBase._report_part(signature))
    W = glb(operation_nodes..., report_nodes...)
    newnode_given_old =
        IdDict{AbstractNode,AbstractNode}([old => source() for old in sources(W)])

    newsignature = copysignature!(signature, newnode_given_old; newmodel_given_old=nothing)

    newfitresult = MLJBase.CompositeFitresult(newsignature)
    setfield!(newfitresult, :network_model_names, getfield(fitresult, :network_model_names))

    return newfitresult
end


# Restores a machine of a composite model by restoring all
# submachines contained in it.
function restore!(mach::Machine{<:Composite})
    glb_node = glb(mach)
    for submach in machines(glb_node)
        restore!(submach)
    end
    mach.state = 1
    return mach
end

function setreport!(copymach::Machine{<:Composite}, mach)
    basic = report(glb(copymach.fitresult))
    additions = report_given_method(mach)[:fit].additions
    copymach.report = Dict{Symbol,Any}(:fit => (; basic, additions))
    return copymach
end
