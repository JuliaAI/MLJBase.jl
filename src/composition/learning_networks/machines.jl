## LEARNING NETWORK MACHINES

surrogate(::Type{<:Deterministic})  = Deterministic()
surrogate(::Type{<:Probabilistic})  = Probabilistic()
surrogate(::Type{<:Unsupervised}) = Unsupervised()
surrogate(::Type{<:Static}) = Static()


"""
    model_supertype(signature)

Return, if this can be deduced, which of `Deterministic`,
`Probabilistic` and `Unsupervised` is the appropriate supertype for a
composite model obtained by exporting a learning network with the
specified `signature`.

A learning network *signature* is a named tuple, such as
`(predict=yhat, transfrom=W)`, specifying what nodes of the network
are called to produce output of each operation represented by the
keys, in an exported version of the network.

If a supertype cannot be deduced, `nothing` is returned.

If the network with given `signature` is not exportable, this method
will not error but it will not a give meaningful return value either.

**Private method.**

"""
function model_supertype(signature)

    operations = keys(signature)

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

function check_surrogate_machine(::Surrogate, signature, _sources)
    isempty(signature) && throw(ERR_MUST_OPERATE)
    isempty(_sources) && throw(ERR_MUST_SPECIFY_SOURCES)
    return nothing
end

function check_surrogate_machine(::Supervised,
                                 signature,
                                 _sources)
    isempty(signature) && throw(ERR_MUST_PREDICT)
    length(_sources) > 1 || throw(err_supervised_nargs())
    return nothing
end

function check_surrogate_machine(::Unsupervised,
                                 signature,
                                 _sources)
    isempty(signature) && throw(ERR_MUST_TRANSFORM)
    length(_sources) < 2 || throw(err_unsupervised_nargs())
    return nothing
end

function machine(model::Surrogate, _sources::Source...; pair_itr...)

    # named tuple, such as `(predict=yhat, transform=W)`:
    signature = (; pair_itr...)
    for op in keys(signature)
        op in OPERATIONS || throw(ArgumentError(
            "`$op` is not an admissible operation. "))
    end

    check_surrogate_machine(model, signature, _sources)

    mach = Machine(model, _sources...)

    mach.fitresult = signature

    return mach

end

function machine(_sources::Source...; pair_itr...)

    signature = (; pair_itr...)

    isempty(signature) && throw(ERR_MUST_OPERATE)

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
    N = glb(mach::Machine{<:Surrogate})

A greatest lower bound for the nodes appearing in the signature of
`mach`.

**Private method.**

"""
glb(mach::Machine{<:Union{Composite,Surrogate}}) =
    glb(values(mach.fitresult)...)


"""
    fit!(mach::Machine{<:Surrogate};
         rows=nothing,
         acceleration=CPU1(),
         verbosity=1,
         force=false))

Train the complete learning network wrapped by the machine
`mach`.

More precisely, if `s` is the learning network signature used to
construct `mach`, then call `fit!(N)`, where `N = glb(values(s)...)`
is a greatest lower bound on the nodes appearing in the signature. For
example, if `s = (predict=yhat, transform=W)`, then call
`fit!(glb(yhat, W))`. Here `glb` is `tuple` overloaded for nodes.

See also [`machine`](@ref)

"""
function fit!(mach::Machine{<:Surrogate}; kwargs...)

    glb_node = glb(mach)
    fit!(glb_node; kwargs...)

    mach.state += 1
    mach.report = report(glb_node)
    return mach

end

MLJModelInterface.fitted_params(mach::Machine{<:Surrogate}) =
    fitted_params(glb(mach))


## CONSTRUCTING THE RETURN VALUE FOR A COMPOSITE FIT METHOD

# Identify which properties of `model` have, as values, a model in the
# learning network wrapped by `mach`, and check that no two such
# properties have have identical values (#377). Return the property name
# associated with each model in the network (in the order appearing in
# `models(glb(mach))`) using `nothing` when the model is not
# associated with any property.
function network_model_names(model::M,
                             mach::Machine{<:Surrogate}) where M<:Model

    signature = mach.fitresult
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
                @error "The hyperparameters $name of "*
                    "$model have identical model "*
                "instances as values. "
            end
        end
        throw(ArgumentError(
        "Two distinct hyper-parameters of a "*
            "composite model that are both "*
            "associated with models in the underlying learning "*
            "network (eg, any two components of a `@pipeline` model) "*
            "cannot have identical values, although they can be `==` "*
            "(corresponding nested properties are `==`). "*
            "Consider constructing instances "*
            "separately or use `deepcopy`. "))
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

    return!(mach::Machine{<:Surrogate}, model, verbosity)

The last call in custom code defining the `MLJBase.fit` method for a
new composite model type. Here `model` is the instance of the new type
appearing in the `MLJBase.fit` signature, while `mach` is a learning
network machine constructed using `model`. Not relevant when defining
composite models using `@pipeline` or `@from_network`.

For usage, see the example given below. Specificlly, the call does the
following:

- Determines which hyper-parameters of `model` point to model
  instances in the learning network wrapped by `mach`, for recording
  in an object called `cache`, for passing onto the MLJ logic that
  handles smart updating (namely, an `MLJBase.update` fallback for
  composite models).

- Calls `fit!(mach, verbosity=verbosity)`.

- Moves any data in source nodes of the learning network into `cache`
  (for data-anonymization purposes).

- Records a copy of `model` in `cache`.

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
                 verbosity)

    _network_model_names = network_model_names(model, mach)

    verbosity isa Nothing || fit!(mach, verbosity=verbosity)

    # anonymize the data:
    sources = mach.args
    data = Tuple(s.data for s in sources)
    [MLJBase.rebind!(s, nothing) for s in sources]

    # record the current hyper-parameter values:
    old_model = deepcopy(model)

    cache = (sources = sources,
             data=data,
             network_model_names=_network_model_names,
             old_model=old_model)

    return mach.fitresult, cache, mach.report

end


#legacy code:
function (mach::Machine{<:Surrogate})()
    Base.depwarn("Calling a learning network machine `mach` "*
                 "with no arguments, as in"*
                 "`mach()`, is "*
                 "deprecated and could lead "*
                 "to unexpected behaviour for `Composite` models "*
                 "with hyper-parameters that are not models. "*
                 "Instead of `fit!(mach, verbosity=verbosity); return mach()` "*
                 "use `return!(mach, model, verbosity)`, "*
                 "where `model` is the `Model` instance appearing in your "*
                 "`MLJBase.fit` signature. Query the `return!` doc-string "*
                 "for details. ",
                 nothing)

    return!(mach, nothing, nothing)
end
network_model_names(model::Nothing, mach::Machine{<:Surrogate}) =
    nothing


## DUPLICATING AND REPLACING PARTS OF A LEARNING NETWORK MACHINE

"""
    replace(mach, a1=>b1, a2=>b2, ...; empty_unspecified_sources=false)

Create a deep copy of a learning network machine `mach` but replacing
any specified sources and models `a1, a2, ...` of the original
underlying network with `b1, b2, ...`.

If `empty_unspecified_sources=true` then any source nodes not
specified are replaced with empty source nodes.

"""
function Base.replace(mach::Machine{<:Surrogate},
                      pairs::Pair...; empty_unspecified_sources=false)

    signature = mach.fitresult
    interface_nodes = values(signature)

    W = glb(interface_nodes...)

    # Note: We construct nodes of the new network as values of a
    # dictionary keyed on the nodes of the old network. Additionally,
    # there are dictionaries of models keyed on old models and
    # machines keyed on old machines. The node and machine
    # dictionaries must be built simultaneously.

    # build model dict:
    model_pairs = filter(collect(pairs)) do pair
        first(pair) isa Model
    end
    models_ = models(W)
    models_to_copy = setdiff(models_, first.(model_pairs))
    model_copy_pairs = [model=>deepcopy(model) for model in models_to_copy]
    newmodel_given_old = IdDict(vcat(model_pairs, model_copy_pairs))

    # build complete source replacement pairs:
    sources_ = mach.args # sources(W)
    specified_source_pairs = filter(collect(pairs)) do pair
        first(pair) isa Source
    end
    unspecified_sources = setdiff(sources_, first.(specified_source_pairs))
    unspecified_sources_wrapping_something =
        filter(s -> !isempty(s), unspecified_sources)
    if !isempty(unspecified_sources_wrapping_something) &&
        !empty_unspecified_sources
        @warn "No replacement specified for one or more non-empty source "*
        "nodes. Contents will be duplicated. "
    end
    if empty_unspecified_sources
        unspecified_source_pairs = [s => source() for
                                    s in unspecified_sources]
    else
        unspecified_source_pairs = [s => deepcopy(s) for
                                    s in unspecified_sources]
    end

    all_source_pairs = vcat(specified_source_pairs, unspecified_source_pairs)

    # drop source nodes from all nodes of network terminating at W:
    nodes_ = filter(nodes(W)) do N
        !(N isa Source)
    end
    isempty(nodes_) && error("All nodes in network are source nodes. ")
    # instantiate node and machine dictionaries:
    newnode_given_old =
        IdDict{AbstractNode,AbstractNode}(all_source_pairs)
    newsources = [newnode_given_old[s] for s in sources_]
    newinterface_node_given_old =
        IdDict{AbstractNode,AbstractNode}()
    newmach_given_old = IdDict{Machine,Machine}()

    # build the new network:
    for N in nodes_
       args = [newnode_given_old[arg] for arg in N.args]
         if N.machine === nothing
             newnode_given_old[N] = node(N.operation, args...)
         else
             if N.machine in keys(newmach_given_old)
                 m = newmach_given_old[N.machine]
             else
                 train_args = [newnode_given_old[arg] for arg in N.machine.args]
                 m = Machine(newmodel_given_old[N.machine.model],
                                train_args...)
                 newmach_given_old[N.machine] = m
             end
             newnode_given_old[N] = N.operation(m, args...)
         end
        if N in interface_nodes
            newinterface_node_given_old[N] = newnode_given_old[N]
        end
    end

    newinterface_nodes = Tuple(newinterface_node_given_old[N] for N in
                          interface_nodes)
    newsignature = NamedTuple{keys(signature)}(newinterface_nodes)

    return machine(mach.model, newsources...; newsignature...)

end
