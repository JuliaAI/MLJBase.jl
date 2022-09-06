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
