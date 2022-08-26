# # SIGNATURES

const DOC_SIGNATURES =
"""
A learning network *signature* is an intermediate object defined when
a user constructs a learning network machine, `mach`. They are named
tuples whose values are the nodes consitituting interface points
between the network and the machine.  Examples are

    (predict=yhat, )
    (transform=Xsmall,)
    (predict=yhat, transform=W, report=(loss=loss_node,))

where `yhat`, `Xsmall`, `W` and `loss_node` are nodes in the network.

If a key `k` is the name of an operation (such as `:predict`,
`:predict_mode`, `:transform`, `inverse_transform`) then `k(mach, X)`
returns `n(X)` where `n` is the corresponding node value.  Each such
node must have a unique origin (`length(origins(n)) === 1`).

The only other allowed key is `:report`, whose corresponding value
must be a named tuple

    (k1=n1, k2=n2, ...)

whose keys are arbitrary, and whose values are nodes of the
network. For each such key-value pair `k=n`, the value returned by
`n()` is included in the named tuple `report(mach)`, with
corresponding key `k`. So, in the third example above,
`report(mach).loss` will return the value of `loss_node()`.

"""

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
    mach.report = MLJBase.report(mach.fitresult)
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
function network_model_names(model::M,
                             mach::Machine{<:Surrogate}) where M<:Model

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

    setfield!(mach.fitresult,
        :network_model_names,
        network_model_names(model, mach))

    return mach.fitresult, cache, mach.report

end

network_model_names(model::Nothing, mach::Machine{<:Surrogate}) =
    nothing

## DUPLICATING/REPLACING PARTS OF A LEARNING NETWORK MACHINE

"""
    copy_or_replace_machine(N::AbstractNode, newmodel_given_old, newnode_given_old)

For now, two top functions will lead to a call of this function:
`Base.replace(::Machine, ...)` and `save(::Machine, ...)`. A call from
`Base.replace` with given `newmodel_given_old` will dispatch to this
method.  A new Machine is built with training data from node N.

"""
function copy_or_replace_machine(N::AbstractNode, newmodel_given_old, newnode_given_old)
    train_args = [newnode_given_old[arg] for arg in N.machine.args]
    return Machine(newmodel_given_old[N.machine.model],
                train_args...)
end

"""
    copy_or_replace_machine(N::AbstractNode, newmodel_given_old::Nothing, newnode_given_old)

For now, two top functions will lead to a call of this function:
`Base.replace(::Machine, ...)` and `save(::Machine, ...)`. A call from
`save` will set `newmodel_given_old` to `nothing` which will then
dispatch to this method.  In this circumstance, the purpose is to make
the machine attached to node N serializable (see
`serializable(::Machine)`).

"""
function copy_or_replace_machine(
    N::AbstractNode,
    newmodel_given_old::Nothing,
    newnode_given_old
)
    m = serializable(N.machine)
    m.args = Tuple(newnode_given_old[s] for s in N.machine.args)
    return m
end

"""
    update_mappings_with_node!(
        newnode_given_old,
        newmach_given_old,
        newmodel_given_old,
        N::AbstractNode)

For Nodes that are not sources, update the appropriate mappings
between elements of the learning networks to be copied and the copy itself.
"""
function update_mappings_with_node!(
    newnode_given_old,
    newmach_given_old,
    newmodel_given_old,
    N::AbstractNode)
    args = [newnode_given_old[arg] for arg in N.args]
    if N.machine === nothing
        newnode_given_old[N] = node(N.operation, args...)
    else
        if N.machine in keys(newmach_given_old)
            m = newmach_given_old[N.machine]
        else
            m = copy_or_replace_machine(N, newmodel_given_old, newnode_given_old)
            newmach_given_old[N.machine] = m
        end
        newnode_given_old[N] = N.operation(m, args...)
    end
end

update_mappings_with_node!(
    newnode_given_old,
    newmach_given_old,
    newmodel_given_old,
    N::Source) = nothing

"""
    copysignature!(signature, newnode_given_old; newmodel_given_old=nothing)

Copies the given signature of a learning network. Contrary to Julia's convention,
this method is actually mutating `newnode_given_old` and `newmodel_given_old` and not
the first `signature` argument.

# Arguments:
- `signature`: signature of the learning network to be copied
- `newnode_given_old`: initialized mapping between nodes of the
learning network to be copied and the new one. At this stage it should
contain only source nodes.
- `newmodel_given_old`: initialized mapping between models of the
learning network to be copied and the new one. This is `nothing` if `save` was
the calling function which will result in a different behaviour of
`update_mappings_with_node!`
"""
function copysignature!(signature, newnode_given_old; newmodel_given_old=nothing)
    operation_nodes = values(_operation_part(signature))
    report_nodes = values(_report_part(signature))
    W = glb(operation_nodes..., report_nodes...)
    # Note: We construct nodes of the new network as values of a
    # dictionary keyed on the nodes of the old network. Additionally,
    # there are dictionaries of models keyed on old models and
    # machines keyed on old machines. The node and machine
    # dictionaries must be built simultaneously.

    # instantiate node and machine dictionaries:
    newoperation_node_given_old =
        IdDict{AbstractNode,AbstractNode}()
    newreport_node_given_old =
        IdDict{AbstractNode,AbstractNode}()
    newmach_given_old = IdDict{Machine,Machine}()

    # build the new network:
    for N in nodes(W)
        update_mappings_with_node!(
            newnode_given_old,
            newmach_given_old,
            newmodel_given_old,
            N
        )
        if N in operation_nodes # could be `Source`
            newoperation_node_given_old[N] = newnode_given_old[N]
        elseif N in report_nodes
            newreport_node_given_old[N] = newnode_given_old[N]
        end
    end
    newoperation_nodes = Tuple(newoperation_node_given_old[N] for N in
                          operation_nodes)
    newreport_nodes = Tuple(newreport_node_given_old[N] for N in
                            report_nodes)
    report_tuple =
        NamedTuple{keys(_report_part(signature))}(newreport_nodes)
    operation_tuple =
        NamedTuple{keys(_operation_part(signature))}(newoperation_nodes)

    newsignature = if isempty(report_tuple)
        operation_tuple
    else
        merge(operation_tuple, (report=report_tuple,))
    end

    return newsignature
end

"""
    replace(mach, a1=>b1, a2=>b2, ...; empty_unspecified_sources=false)

Create a deep copy of a learning network machine `mach` but replacing
any specified sources and models `a1, a2, ...` of the original
underlying network with `b1, b2, ...`.

If `empty_unspecified_sources=true` then any source nodes not
specified are replaced with empty source nodes, unless they wrap an
`Exception` object.

"""
function Base.replace(mach::Machine{<:Surrogate},
                      pairs::Pair...; empty_unspecified_sources=false)

    signature = MLJBase.signature(mach.fitresult)
    operation_nodes = values(_operation_part(signature))
    report_nodes = values(_report_part(signature))

    W = glb(operation_nodes..., report_nodes...)

    # Instantiate model dictionary:
    model_pairs = filter(collect(pairs)) do pair
        first(pair) isa Model
    end
    models_ = models(W)
    models_to_copy = setdiff(models_, first.(model_pairs))
    model_copy_pairs = [model=>deepcopy(model) for model in models_to_copy]
    newmodel_given_old = IdDict(vcat(model_pairs, model_copy_pairs))
    # build complete source replacement pairs:
    sources_ = sources(W)
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
    newnode_given_old =
        IdDict{AbstractNode,AbstractNode}(all_source_pairs)
    newsources = [newnode_given_old[s] for s in sources(W)]

    newsignature = copysignature!(signature, newnode_given_old, newmodel_given_old=newmodel_given_old)


    return machine(mach.model, newsources...; newsignature...)

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
    additions = mach.report.additions
    copymach.report = (; basic, additions)
end
