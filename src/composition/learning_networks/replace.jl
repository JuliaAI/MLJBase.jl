# # DUPLICATING LEARNING NETWORKS WITH MODEL REPLACEMENT

# needed for @from_network and for save/restore for <: Composite models

    """
    machine_replacement(N::AbstractNode, newmodel_given_old, newnode_given_old)

For now, two top functions will lead to a call of this function:
`Base.replace(::Machine, ...)` and `save(::Machine, ...)`. A call from
`Base.replace` with given `newmodel_given_old` will dispatch to this
method.  A new Machine is built with training data from node N.

"""
function machine_replacement(N::AbstractNode, newmodel_given_old, newnode_given_old)
    train_args = [newnode_given_old[arg] for arg in N.machine.args]
    return Machine(newmodel_given_old[N.machine.model],
                train_args...)
end

"""
    machine_replacement(N::AbstractNode, newmodel_given_old::Nothing, newnode_given_old)

For now, two top functions will lead to a call of this function:
`Base.replace(::Machine, ...)` and `save(::Machine, ...)`. A call from
`save` will set `newmodel_given_old` to `nothing` which will then
dispatch to this method.  In this circumstance, the purpose is to make
the machine attached to node N serializable (see
`serializable(::Machine)`).

"""
function machine_replacement(
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
            m = machine_replacement(N, newmodel_given_old, newnode_given_old)
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

