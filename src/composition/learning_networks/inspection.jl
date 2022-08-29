## INSPECTING LEARNING NETWORKS

"""
    tree(N)

Return a named-tuple respresentation of the ancestor tree `N`
(including training edges)

"""
function tree(W::Node)
    mach = W.machine
    if mach === nothing
        value2 = nothing
        endkeys = []
        endvalues = []
    else
        value2 = mach.model
        endkeys = (Symbol("train_arg", i) for i in eachindex(mach.args))
        endvalues = (tree(arg) for arg in mach.args)
    end
    keys = tuple(:operation,  :model,
                 (Symbol("arg", i) for i in eachindex(W.args))...,
                 endkeys...)
    values = tuple(W.operation, value2,
                   (tree(arg) for arg in W.args)...,
                   endvalues...)
    return NamedTuple{keys}(values)
end
tree(s::Source) = (source = s,)

# """
#    args(tree; train=false)

# Return a vector of the top level args of the tree associated with a node.
# If `train=true`, return the `train_args`.
# """
# function args(tree; train=false)
#     keys_ = filter(keys(tree) |> collect) do key
#         match(Regex("^$("train_"^train)arg[0-9]*"), string(key)) !== nothing
#     end
#     return [getproperty(tree, key) for key in keys_]
# end

"""
    models(N::AbstractNode)

A vector of all models referenced by a node `N`, each model appearing
exactly once.

"""
function models(W::AbstractNode)
    models_ = filter(flat_values(tree(W)) |> collect) do model
        model isa Union{Model,Symbol}
    end
    return unique(models_)
end

"""
    sources(N::AbstractNode)

A vector of all sources referenced by calls `N()` and `fit!(N)`. These
are the sources of the ancestor graph of `N` when including training
edges.

Not to be confused with `origins(N)`, in which training edges are
excluded.

See also: [`origins`](@ref), [`source`](@ref).
"""
function sources(W::AbstractNode; kind=:any)
    if kind == :any
        sources_ = filter(flat_values(tree(W)) |> collect) do value
            value isa Source
        end
    else
        sources_ = filter(flat_values(tree(W)) |> collect) do value
            value isa Source && value.kind == kind
        end
    end
    return unique(sources_)
end

"""
    machines(N::AbstractNode [, model::Model])

List all machines in the ancestor graph of node `N`, optionally
restricting to those machines whose corresponding model matches the
specifed `model`.

Here two models *match* if they have the same, possibly nested
hyperparameters, or, more precisely, if
`MLJModelInterface.is_same_except(m1, m2)` is `true`.

See also `MLJModelInterface.is_same_except`.

"""
function machines(W::Node, model=nothing)
    if W.machine === nothing
        machs = vcat((machines(arg) for arg in W.args)...) |> unique
    else
        machs = vcat(Machine[W.machine, ],
                    (machines(arg) for arg in W.args)...,
                    (machines(arg) for arg in W.machine.args)...) |> unique
    end
    model === nothing && return machs
    return filter(machs) do mach
        mach.model == model
    end
end

args(::Source) = []
args(N::Node) = N.args
train_args(::Source) = []
train_args(N::Node{<:Machine}) = N.machine.args
train_args(N::Node{Nothing}) = []

"""
    children(N::AbstractNode, y::AbstractNode)

List all (immediate) children of node `N` in the ancestor graph of `y`
(training edges included).

"""
children(N::AbstractNode, y::AbstractNode) = filter(nodes(y)) do W
    N in args(W) || N in train_args(W)
end |> unique

"""
    lower_bound(type_itr)

Return the minimum type in the collection `type_itr` if one exists
(mininum in the sense of `<:`). If `type_itr` is empty, return `Any`,
and in all other cases return the universal lower bound `Union{}`.

"""
function lower_bound(Ts)
    isempty(Ts) && return Any
    sorted = sort(collect(Ts), lt=<:)
    candidate = first(sorted)
    all(T -> candidate <: T, sorted[2:end]) && return candidate
    return Union{}
end

function _lower_bound(Ts)
    Unknown in Ts && return Unknown
    return lower_bound(Ts)
end

MLJModelInterface.input_scitype(N::Node) = Unknown
MLJModelInterface.input_scitype(N::Node{<:Machine}) =
    input_scitype(N.machine.model)

