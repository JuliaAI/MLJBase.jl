# a signature is just a thin wrapper for what the user knows as "learning network
# interface"; see constant DOC_NETWORK_INTERFACES below for details.

# # HELPERS

"""
    machines_given_model(node::AbstractNode)

**Private method.**

Return a dictionary of machines, keyed on model, for the all machines in the completed
learning network for which `node` is the greatest lower bound. Only machines bound to
symbolic models are included. Values are always vectors, even if they contain only a
single machine.

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
    tuple_keyed_on_model(f, machines_given_model; scalarize=true, drop_nothings=true)

**Private method.**

Given a dictionary of machine vectors, keyed on model names (symbols), broadcast `f` over
each vector, and make the result, in the returned named tuple, the value associated with
the corresponding model name as key.

Singleton vector values are scalarized, unless `scalarize = false`.

If a value in the computed named tuple is `nothing`, or a vector of `nothing`s, then the
entry is dropped from the tuple, unless `drop_nothings=false`.

"""
function tuple_keyed_on_model(f, machines_given_model; scalarize=true, drop_nothings=true)
    models = keys(machines_given_model) |> collect
    named_tuple_values = map(models) do model
        value = [f(m) for m in machines_given_model[model]]
        scalarize && return  attempt_scalarize(value)
        return value
    end
    if drop_nothings
        mask = map(named_tuple_values) do v
            !(isnothing(v) || (v isa AbstractVector && eltype(v) === Nothing))
        end |> collect
        models = models[mask]
        named_tuple_values = named_tuple_values[mask]
    end
    return NamedTuple{tuple(models...)}(tuple(named_tuple_values...))
end


# # DOC STRING

const DOC_NETWORK_INTERFACES =
    """

    A *learning network interface* is a named tuple declaring certain interface points in
    a learning network, to be used when "exporting" the network as a new stand-alone model
    type. Examples are

        (predict=yhat,)
        (transform=Xsmall,)
        (predict=yhat, transform=W, report=(loss=loss_node,))

    Here `yhat`, `Xsmall`, `W` and `loss_node` are nodes in the network.

    The keys of the signature are always one of the following:

    - The name of an operation, such as `:predict`, `:predict_mode`, `:transform`,
      `:inverse_transform`.

    - `:report`, for exposing results of calling a node *with no argumentes*.

    - `:acceleration`, for articulating acceleration mode for training the network, e.g.,
      `CPUThreads()`. Corresponding value must be an `AbstractResource`. If not included,
      `CPU1()` is used.

    ### Operation keys

    If the key is an operation, then the value must be a node `n` in the network with a
    unique origin (`length(origins(n)) === 1`). The intention of a declaration such as
    `predict=yhat` is that the exported model type implements `predict`, which, when
    applied to new data `Xnew`, should return `yhat(Xnew)`.

    #### Including report nodes in an interface

    If the key is `:report`, then the corresponding value must be a named tuple

        (k1=n1, k2=n2, ...)

    whose keys are arbitrary, and whose values are nodes of the network. The intention of
    a declaration such as `report = (loss=loss_node,)` is to include, in reports associated
    with a new exported model instance, the output of the call `loss_node()`.

   """

# # SIGNATURES

"""
    Signature(interface::NamedTuple)

**Private type.**

Return a thinly wrapped version of a learning network interface (defined below). Unwrap
with `MLJBase.unwrap`:

```julia
interface = (predict=source(), report=(loss=source(),))
signature = MLJBase.Signature(interface)
@assert MLJBase.unwrap(signature) === interface
```
$DOC_NETWORK_INTERFACES

"""
struct Signature{S<:NamedTuple}
    interface::S
end

unwrap(signature::Signature) = signature.interface


# # METHODS

"""
    operation_nodes(signature)

**Private method.**

Return the operation nodes of `signature`, as a named tuple keyed on operation names.

See also [`MLJBase.Signature`](@ref).

"""
function operation_nodes(signature::Signature)
    interface = unwrap(signature)
    ops = filter(in(OPERATIONS), keys(interface))
    return NamedTuple{ops}(map(op->getproperty(interface, op), ops))
end

"""
    report_nodes(signature)

**Private method.**

Return the report nodes of `signature`, as a named tuple keyed on operation names.

See also [`MLJBase.Signature`](@ref).

"""
function report_nodes(signature::Signature)
    interface = unwrap(signature)
    :report in keys(interface) || return NamedTuple()
    return interface.report
end

"""
    acceleration(signature)

**Private method.**

Return the acceleration mode of `signature`.

See also [`MLJBase.Signature`](@ref).

"""
function acceleration(signature::Signature)
    interface = unwrap(signature)
    :acceleration in keys(interface) || return CPU1()
    return interface.acceleration
end

"""
    operations(signature)

**Private method.**

Return the names of all operations in `signature`.

See also [`MLJBase.Signature`](@ref).

"""
operations(signature::Signature) = keys(operation_nodes(signature))

glb(signature::Signature) = glb(
    values(operation_nodes(signature))...,
    values(report_nodes(signature))...,
)


"""
    age(signature::Signature)

**Private method.**

Return the sum of the ages of all machines in the underlying network of `signature`.

See also [`MLJBase.Signature`](@ref).

"""
age(signature::Signature) = sum(age, machines(glb(signature)))

"""
    report_supplement(signature)

**Private method.**

Generate a deep copy of the supplementary report defined by the signature, i.e., by
calling the nodes appearing as values of `signature.report` with zero arguments. Returns a
named with the keys of `signature.report`.

See also [`MLJBase.Signature`](@ref).

"""
function report_supplement(signature::Signature)
    report_nodes = MLJBase.report_nodes(signature)
    _call(node) = deepcopy(node())
    _keys = keys(report_nodes)
    _values = values(report_nodes)
    return NamedTuple{_keys}(_call.(_values))
end

"""
    report(signature; supplement=true)

**Private method.**

Generate a report for the learning network associated with `signature`, including the
supplementary report.

Suppress calling of the report nodes of `signature`, and excluded their contribution to
the output, by specifying `supplement=false`.

See also [`MLJBase.report_supplement`](@ref).

See also [`MLJBase.Signature`](@ref).

"""
function report(signature::Signature; supplement=true)
    greatest_lower_bound = glb(signature)
    supplement_report = supplement ? MLJBase.report_supplement(signature) : NamedTuple()
    d = MLJBase.machines_given_model(greatest_lower_bound)
    internal_report = MLJBase.tuple_keyed_on_model(report, d)
    merge(internal_report, supplement_report)
end

"""
    output_and_report(signature, operation, Xnew)

**Private method.**

Duplicate `signature` and return appropriate output for the specified `operation` (a key
of `signature`) applied to the duplicate, together with the operational report. Report
nodes of `signature` are not called, and they make no contribution to the report.

See also [`MLJBase.Signature`](@ref).

"""
function output_and_report(signature, operation, Xnew)
    signature_clone = replace(signature, copy_unspecified_deeply=false)
    output =  getproperty(MLJBase.unwrap(signature_clone), operation)(Xnew)
    report = MLJBase.report(signature_clone; supplement=false)
    return output, report
end
