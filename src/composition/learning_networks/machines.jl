## LEARNING NETWORK MACHINES

# ***
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

function machine(model::Surrogate, _sources::Source...; pair_itr...)

    # named tuple, such as `(predict=yhat, transform=W)`:
    signature = (; pair_itr...)
    for op in keys(signature)
        op in OPERATIONS || throw(ArgumentError(
            "`$op` is not an admissible operation. "))
    end
    if isempty(signature)
        model isa Supervised &&
            throw(ArgumentError(
                "You must specify at least `predict=<some node>`. "))
        model isa Unsupervised &&
            throw(ArgumentError(
                "You must specify at least `transform=<some node>`. "))
    end

    if model isa Supervised
        length(_sources) in [2, 3] ||
            error("Incorrect number of source nodes specified.\n"*
                  "Use  `machine(model, X, y; ...)` or "*
                  "`machine(model, X, y, w; ...)` when "*
                  "`model isa Supervised`. ")
    elseif model isa Unsupervised
        length(_sources) == 1 ||
            error("Incorrect number of source nodes specified.\n"*
                  "Use `machine(model, X; ...)` when "*
                  "`model isa Unsupervised. ` (even if `Static`). ")
    else
        throw(DomainError)
    end

    mach = Machine(model, _sources...)

    mach.fitresult = signature

    return mach

end

function machine(sources::Source...; pair_itr...)

    signature = (; pair_itr...)

    isempty(signature) &&
        throw(ArgumentError(
            "You must specify at least `predict=<some node>` or "*
            "or `transform=<some node>`. "))

    T = model_supertype(signature)
    if T == nothing
        @warn "Unable to infer surrogate model type. \n"*
        "Using Deterministic(). To override use:\n "*
        "`machine(Probabilistic(), ...)` or `machine(Interval(), ...)`"
        model = Deterministic()
    else
        model = surrogate(T)
    end

    return machine(model, sources...; pair_itr...)

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

    signature = mach.fitresult
    glb_node = glb(values(signature)...) # greatest lower bound node
    fit!(glb_node; kwargs...)

    # mach.cache = anonymize!(mach.args)
    mach.state += 1
    mach.report = report(glb_node)
    return mach

end

# make learning network machines callable for use in manual export of
# learning networks:
function (mach::Machine{<:Surrogate})()
    # anonymize sources:
    mach.cache = anonymize!(mach.args)
    return mach.fitresult, mach.cache, mach.report
end

MLJModelInterface.fitted_params(mach::Machine{<:Surrogate}) =
    fitted_params(glb(values(mach.fitresult)...))


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
