# # HELPERS

"""
    machines_given_model(node::AbstractNode)

**Private method.**

Return a dictionary of machines, keyed on model, for the all machines in the completed
learning network for which `node` is the greatest lower bound. Only  machines
bound to symbolic models are included.

"""
function machines_given_model(node::AbstractNode)
    ret = LittleDict{Symbol,Any}()
    for mach in machines(node)
        model = mach.model
        model isa Symbol || continue
        if !haskey(ret, model)
            ret[model] = Any[mach,]
        else
            push!(ret[model], mach)
        end
    end
    return ret
end

attempt_scalarize(v) = length(v) == 1 ? v[1] : v

"""
    tuple_keyed_on_model(machines_given_model, f)

**Private method.**

Given a dictionary of machine vectors, keyed on model names (symbols), broadcast `f` over
each vector, and make the result, in the returned named tuple, the value associated with
the corresponding model name as key. Singleton vector values are scalarized.

"""
function tuple_keyed_on_model(machines_given_model, f; scalarize=true)
    models = tuple(keys(machines_given_model)...)
    named_tuple_values = map(models) do model
        value = [f(m) for m in machines_given_model[model]]
        scalarize && return  attempt_scalarize(value)
        return value
    end
    return NamedTuple{models}(named_tuple_values)
end

# # SIGNATURES

const DOC_SIGNATURES =
    """

    A learning network *signature* is a named tuple declaring certain interface points in
    a learning network, to be used when "exporting" the network as a new stand-alone model
    type. Examples are

        (predict=yhat, )
        (transform=Xsmall,)
        (predict=yhat, transform=W, report=(loss=loss_node,))

    where `yhat`, `Xsmall`, `W` and `loss_node` are nodes in the network.

    The keys of the signature are always the name of an operation, such as `:predict`,
    `:predict_mode`, `:transform`, `inverse_transform`, or the special key `:report`.

    If the key is an operation, then the value must be a node `n` in the network with a
    unique origin (`length(origins(n)) === 1`). The intention of a declaration such as
    `predict=yhat` is that the exported model type implements `predict`, which, when
    applied to new data `Xnew`, should return `yhat(Xnew; composite=model)`, where `model`
    is the relevant instnace of the new exported model type.

    If the key is `:report`, then the corresponding value must be a named tuple

        (k1=n1, k2=n2, ...)

    whose keys are arbitrary, and whose values are nodes of the network. The intention of
    a declaration such as `report = (loss=loss_node,)` is to include in reports associated
    with a new exported model instance, the output of `loss_node(; composite=model)`.


"""

"""
    operation_nodes(signature)

Return the operation nodes of `signature`, as a named tuple keyed on operation names.

$DOC_SIGNATURES

"""
function operation_nodes(signature::NamedTuple)
    ops = filter(in(OPERATIONS), keys(signature))
    return NamedTuple{ops}(map(op->getproperty(signature, op), ops))
end

"""
    report_nodes(signature)

Return the report nodes of `signature`, as a named tuple keyed on operation names.

$DOC_SIGNATURES

"""
function report_nodes(signature::NamedTuple)
    :report in keys(signature) || return NamedTuple()
    return signature.report
end

"""
    operations(signature)

Return the names of all operations in `signature`.

$DOC_SIGNATURES

"""
operations(signature::NamedTuple) = keys(operation_nodes(signature))

glb(signature::NamedTuple) = glb(
    values(operation_nodes(signature))...,
    values(report_nodes(signature))...,
)


"""
    age(signature::NamedTuple)

Return the sum of the ages of all machines in the underlying network of `signature`.

$DOC_SIGNATURES

"""
age(signature::NamedTuple) = sum(age, machines(glb(signature)))

"""
    report_supplement(signature)

Generate a deep copy of the supplementary report defined by the signature, i.e., by
calling the nodes appearing as values of `signature.report` with zero arguments. Returns a
named with the keys of `signature.report`.

$DOC_SIGNATURES

"""
function report_supplement(signature::NamedTuple)
    report_nodes = MLJBase.report_nodes(signature)
    _call(node) = deepcopy(node())
    _keys = keys(report_nodes)
    _values = values(report_nodes)
    return NamedTuple{_keys}(_call.(_values))
end

"""
    report(signature)

Generate a report for the learning network associated with `signature`, including the
supplementary report.

See also [`MLJBase.report_supplement`](@ref).

$DOC_SIGNATURES

"""
function report(signature::NamedTuple)
    greatest_lower_bound = glb(signature)
    supplement = report_supplement(signature)
    d = machines_given_model(greatest_lower_bound)
    internal = tuple_keyed_on_model(d, mach -> report_given_method(mach), scalarize=false)
    merge(internal, supplement)
end
