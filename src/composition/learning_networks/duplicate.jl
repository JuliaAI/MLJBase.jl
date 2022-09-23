# `Base.replace` overloadings for use in learning networks


# # HELPERS

"""
    machine_replacement(node, newmodel_given_old, newnode_given_old, serializable)

**Private method** only called by [`update_mappings_with_node`](@ref).

If `serializable=false`, return a new machine instance by copying `node.mach` and changing
the `model` and `args` field values as derived from the provided dictionaries. In this way
the returned machine is hooked into the new learning network defined by the values of
`newnode_given_old`.

If `serializable=true`, return a serializable copy instead (namely,
`serializable(node.mach)`) and ignore the `newmodel_given_old` dictionary (no model
replacement).

See also [`serializable`](@ref).

"""
function machine_replacement(
    N::AbstractNode,
    newmodel_given_old,
    newnode_given_old,
    serializable
)
    # the `replace` called here is defined in src/machines.jl:
    mach = serializable ? MLJBase.serializable(N.machine) :
        replace(N.machine, :model => newmodel_given_old[N.machine.model])
    mach.args = Tuple(newnode_given_old[arg] for arg in N.machine.args)
    return mach
end

"""
    update_mappings_with_node!(
        newnode_given_old,
        newmach_given_old,
        newmodel_given_old,
        node::AbstractNode)

**Private method.**

This is a method called, in appropriate sequence, over each `node` a learning network
being duplicated. If `node` is not a `Source`, it updates the three dictionary arguments
which link the new network to the old one, and otherwise does nothing.

Only `_replace` calls this method.

"""
function update_mappings_with_node!(
    newnode_given_old,
    newmach_given_old,
    newmodel_given_old,
    serializable,
    N::AbstractNode,
)
    args = [newnode_given_old[arg] for arg in N.args]
    if isnothing(N.machine)
        newnode_given_old[N] = node(N.operation, args...)
    else
        if N.machine in keys(newmach_given_old)
            m = newmach_given_old[N.machine]
        else
            m = machine_replacement(N, newmodel_given_old, newnode_given_old, serializable)
            newmach_given_old[N.machine] = m
        end
        newnode_given_old[N] = N.operation(m, args...)
    end
end

update_mappings_with_node!(::Any, ::Any, ::Any, ::Any, N::Source) = nothing


# # REPLACE METHODS

const DOC_REPLACE_OPTIONS =
    """

    # Options

    - `empty_unspecified_sources=false`: If `true`, any source nodes not specified are
      replaced with empty source nodes.

    - `copy_unspecified_deeply=true`: If `false`, models or sources not listed for
      replacement are identically equal in the original and returned node.

    - `serializable=false`: If `true`, all machines in the new network are serializable.
      However, all `model` replacements are ignored, and unspecified sources are always
      replaced with empty ones.

    """

"""
    replace(node, a1=>b1, a2=>b2, ...; options...)

Recursively copy `node` and all nodes in the learning network for which it is a least
upper bound, but replacing any specified sources and models `a1, a2, ...` of that network
with `b1, b2, ...`.

$DOC_REPLACE_OPTIONS

"""
function Base.replace(W::AbstractNode, pairs::Pair...; kwargs...)
    newnode_given_old  = _replace(W, pairs...; kwargs...)
    return newnode_given_old[W]
end

"""
    replace(signature, a1=>b1, a2=>b2, ...; options...)

Copy the provided learning network signature, including the complete underlying learning
network, but replacing any specified sources and models `a1, a2, ...` of the original
underlying network with `b1, b2, ...`.

$DOC_REPLACE_OPTIONS

See also [`MLJBase.Signature`](@ref).

"""
function Base.replace(signature::Signature, pairs::Pair...; node_dict=false, kwargs...)

    # If `node_dict` is true, then we additionally return `newnode_given_old` computed
    # below.

    operation_nodes = values(MLJBase.operation_nodes(signature))
    report_nodes = values(MLJBase.report_nodes(signature))

    W = glb(operation_nodes..., report_nodes...)
    newnode_given_old = _replace(W, pairs...; kwargs...)

    # instantiate special node dictionaries:
    newoperation_node_given_old =
        IdDict{AbstractNode,AbstractNode}()
    newreport_node_given_old =
        IdDict{AbstractNode,AbstractNode}()

    # update those dictionaries based on the output of `_replace`:
    for N in Set(operation_nodes) ∪ Set(report_nodes)
        if N in operation_nodes # could be `Source`
            newoperation_node_given_old[N] = newnode_given_old[N]
        else
            k= collect(keys(newnode_given_old))
            newreport_node_given_old[N] = newnode_given_old[N]
        end
    end

    # assemble the new signature:
    newoperation_nodes = Tuple(newoperation_node_given_old[N] for N in
                          operation_nodes)
    newreport_nodes = Tuple(newreport_node_given_old[N] for N in
                            report_nodes)
    report_tuple = NamedTuple{keys(MLJBase.report_nodes(signature))}(newreport_nodes)
    operation_tuple = NamedTuple{MLJBase.operations(signature)}(newoperation_nodes)
    newsignature = if isempty(report_tuple)
        operation_tuple
    else
        merge(operation_tuple, (report=report_tuple,))
    end |> MLJBase.Signature

    node_dict || return newsignature
    return newsignature, newnode_given_old
end

"""
    replace(mach, a1=>b1, a2=>b2, ...; options...)

Return a copy the learning network machine `mach`, and it's underlying learning network,
but replacing any specified sources and models `a1, a2, ...` of the original underlying
network with `b1, b2, ...`.

$DOC_REPLACE_OPTIONS

"""
function Base.replace(mach::Machine{<:Surrogate}, pairs::Pair...; kwargs...)
    signature = MLJBase.signature(mach.fitresult) |> Signature

    newsignature, newnode_given_old =
        replace(signature, pairs...; node_dict=true, kwargs...)

    newinterface = unwrap(newsignature)

    newargs = [newnode_given_old[arg] for arg in mach.args]

    return machine(mach.model, newargs...; newinterface...)
end

# Copy the complete learning network having `W` as a greatest lower bound, executing the
# specified replacements, and return the dictionary mapping old nodes to new nodes.
function _replace(
    W::AbstractNode,
    pairs::Pair...;
    empty_unspecified_sources=false,
    copy_unspecified_deeply=true,
    serializable=false,
)

    serializable && (empty_unspecified_sources = true)
    clone(item) = copy_unspecified_deeply ? deepcopy(item) : item

    # Instantiate model dictionary:
    model_pairs = filter(collect(pairs)) do pair
        first(pair) isa Model
    end
    models_ = models(W)
    models_to_copy = setdiff(models_, first.(model_pairs))
    model_copy_pairs = [model=>clone(model) for model in models_to_copy]
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
        !empty_unspecified_sources && copy_unspecified_deeply
        @warn "No replacement specified for one or more non-empty source "*
            "nodes. Contents will be duplicated. Perhaps you want to specify "*
            "`copy_unspecified_deeply=false` to prevent copying of sources "*
            "and models, or `empty_unspecified_sources=true`."
    end
    if empty_unspecified_sources
        unspecified_source_pairs = [s => source() for
                                    s in unspecified_sources]
    else
        unspecified_source_pairs = [s => clone(s) for
                                    s in unspecified_sources]
    end
    all_source_pairs = vcat(specified_source_pairs, unspecified_source_pairs)

    # inititialization:
    newnode_given_old =  IdDict{AbstractNode,AbstractNode}(all_source_pairs)
    newmach_given_old = IdDict{Machine,Machine}()

    # build the new network:
    for N in nodes(W)
        update_mappings_with_node!(
            newnode_given_old,
            newmach_given_old,
            newmodel_given_old,
            serializable,
            N
        )
    end

    return newnode_given_old

end
