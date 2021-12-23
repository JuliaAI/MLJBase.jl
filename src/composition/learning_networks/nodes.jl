## NODES

struct Node{T<:Union{Machine, Nothing}} <: AbstractNode

    operation   # eg, `predict` or a static operation, such as `exp`
    machine::T  # is `nothing` for static operations

    # nodes called to get args for `operation(model, ...) ` or
    # `operation(...)`:
    args::Tuple{Vararg{AbstractNode}}

    # sources of ancestor graph (training edges excluded)
    origins::Vector{Source}

    # all ancestors (training edges included) listed in
    # order consistent with extended graph, excluding self
    nodes::Vector{AbstractNode}

    function Node(operation,
                  machine::T,
                  args::AbstractNode...) where T<:Union{Machine, Nothing}

        # check the number of arguments:
        # if machine === nothing && isempty(args)
        #     error("`args` in `Node(::Function, args...)` must be non-empty. ")
        # end

        origins_ = unique(vcat([origins(arg) for arg in args]...))
        # length(origins_) == 1 ||
        #     @warn "A node referencing multiple origins when called " *
        #           "has been defined:\n$(origins_). "

        # initialize the list of upstream nodes:
        nodes_ = AbstractNode[]

        # merge the lists from arguments:
        nodes_ =
            vcat(AbstractNode[], (nodes(n) for n in args)...) |> unique

        # merge the lists from training arguments:
        if machine !== nothing
            nodes_ =
                vcat(nodes_, (nodes(n) for n in machine.args)...) |> unique
        end

        return new{T}(operation, machine, args, origins_, nodes_)
    end
end

"""

    nrows_at_source(N::node)

Return the number of rows of data wrapped at the source of `N`,
assumming this is unique.

Not to be confused with `J = nrows(N)`, which is a new node such that
`J() = nrows(N())`.

See also [`nrows`](@ref)

"""
function nrows_at_source(X::Node)
    ss = sources(X)
    length(ss) == 1 ||
        error("Node does not have a unique source. ")
    return nrows_at_source(first(ss))
end


"""
    origins(N)

Return a list of all origins of a node `N` accessed by a call `N()`.
These are the source nodes of ancestor graph of `N` if edges
corresponding to training arguments are excluded. A `Node` object
cannot be called on new data unless it has a unique origin.

Not to be confused with `sources(N)` which refers to the same graph
but without the training edge deletions.

See also: [`node`](@ref), [`source`](@ref).
"""
origins(X::Node) = X.origins

"""
    nodes(N)

Return all nodes upstream of a node `N`, including `N` itself, in an
order consistent with the extended directed acyclic graph of the
network. Here "extended" means edges corresponding to training
arguments are included.

*Warning.* Not the same as `N.nodes`, which may not include `N`
 itself.

"""
nodes(X::Node) = AbstractNode[X.nodes..., X]

color(N::Node{Nothing}) = :green
color(N::Node) = (N.machine.frozen ? :red : :green)

# constructor for static operations:
Node(operation, args::AbstractNode...) = Node(operation, nothing, args...)

_check(y::Node) = nothing
_check(y::Node{Nothing}) = length(y.origins) == 1 ? nothing :
    error("Node $y has multiple origins and cannot be called "*
              "on new data. ")

# make nodes callable:
(y::Node)(; rows=:) = _apply((y, y.machine); rows=rows)
(y::Node)(Xnew) = (_check(y); _apply((y, y.machine), Xnew))
(y::Node{Nothing})(; rows=:) = _apply((y, ); rows=rows)
(y::Node{Nothing})(Xnew)= (_check(y); _apply((y, ), Xnew))

function _apply(y_plus, input...; kwargs...)
    y = y_plus[1]
    mach = y_plus[2:end] # in static case this is ()
    raw_args = map(y.args) do arg
        arg(input...; kwargs...)
    end
    try
        (y.operation)(mach..., raw_args...)
    catch exception
        @error "Failed "*
        "to apply the operation `$(y.operation)` to the machine "*
        "$(y.machine), which receives it's data arguments from one or more "*
        "nodes in a learning network. Possibly, one of these nodes "*
        "is delivering data that is incompatible with the machine's model.\n"*
        diagnostics(y, input...; kwargs...)
        throw(exception)
    end
end

ScientificTypes.elscitype(N::Node) = Unknown
function ScientificTypes.elscitype(
    N::Node{<:Machine{<:Union{Deterministic,Unsupervised}}})
    if N.operation == MLJBase.predict
        return target_scitype(N.machine.model)
    elseif N.operation == MLJBase.transform
        return output_scitype(N.machine.model)
    elseif N.operation == MLJBase.inverse_transform
        return input_scitype(N.machine.model)
    end
    return Unknown
end

# TODO after
# https://github.com/alan-turing-institute/ScientificTypesBase.jl/issues/102 :
# Add Probabilistic case to above

ScientificTypes.scitype(N::Node) = CallableReturning{elscitype(N)}


## FITTING A NODE

# flush a (possibly remote) channel"
GenericChannel{T} = Union{Channel{T}, Distributed.RemoteChannel{<:Channel{T}}}
function flush!(c::GenericChannel{T}) where T
    ret = T[]
    while isready(c)
        push!(ret, take!(c))
    end
    return ret
end

"""
    fit!(N::Node;
         rows=nothing,
         verbosity=1,
         force=false,
         acceleration=CPU1())

Train all machines required to call the node `N`, in an appropriate
order.  These machines are those returned by
`machines(N)`.


"""
fit!(y::Node; acceleration=CPU1(), kwargs...) =
    fit!(y::Node, acceleration; kwargs...)

fit!(y::Node, ::AbstractResource; kwargs...) =
        error("Only `acceleration=CPU1()` currently supported")

function fit!(y::Node, ::CPU1; kwargs...)

    _machines = machines(y)

    # flush the fit_okay channels:
    @sync begin
        for mach in _machines
            @async flush!(mach.fit_okay)
        end
    end

    # fit the machines asynchronously;
    @sync begin
        for mach in _machines
            @async fit_only!(mach, true; kwargs...)
        end
    end

    return y
end
fit!(S::Source; args...) = S

# allow arguments of `Nodes` and `Machine`s to appear
# at REPL:
istoobig(d::Tuple{AbstractNode}) = length(d) > 10

# overload show method


_formula(stream::IO, X::AbstractNode, indent) =
    (print(stream, repeat(' ', indent));_formula(stream, X, 0, indent))
_formula(stream::IO, X::Source, depth, indent) = show(stream, X)
_formula(stream::IO, X::ErrorNode, depth, indent) = show(stream, X)
function _formula(stream, X::Node, depth, indent)
    operation_name = string(typeof(X.operation).name.mt.name)
    anti = max(length(operation_name) - INDENT)
    print(stream, operation_name, "(")
    n_args = length(X.args)
    if X.machine !== nothing
        print(stream, crind(indent + length(operation_name) - anti))
        printstyled(IOContext(stream, :color=>SHOW_COLOR),
#                        handle(X.machine),
                        X.machine,
                        bold=SHOW_COLOR)
        n_args == 0 || print(stream, ", ")
    end
    for k in 1:n_args
        print(stream, crind(indent + length(operation_name) - anti))
        _formula(stream, X.args[k],
                        depth + 1,
                        indent + length(operation_name) - anti )
        k == n_args || print(stream, ",")
    end
    print(stream, ")")
end

function Base.show(io::IO, ::MIME"text/plain", X::Node)
#    description = string(typeof(X).name.name)
    #    str = "$description $(handle(X))"
    println(io, "$X")
    println(io, "  args:")
    for i in eachindex(X.args)
        arg = X.args[i]
        println(io, "    $i:\t$arg")
    end
    print(io, "  formula:\n")
    _formula(io, X, 4)
    # print(io, " ")
    # printstyled(IOContext(io, :color=>SHOW_COLOR),
    #             handle(X),
    #             color=color(X))
end


## REPORTS AND FITRESULTS FOR NODES

# Both of these exposed but not intended for public use

# here `f` is `report` or `fitted_params`; returns a named tuple:
function item_given_machine(f, N)
    machs = machines(N) |> reverse
    items = map(machs) do m
        try
            f(m)
        catch exception
            if exception isa UndefRefError
                error("UndefRefError intercepted. Perhaps "*
                      "you forgot to `fit!` a machine or node?")
            else
                throw(exception)
            end
        end
    end
    key = f isa typeof(report) ?
        :report_given_machine :
        :fitted_params_given_machine
    dict = LittleDict(machs[j] => items[j] for j in eachindex(machs))
    return NamedTuple{(:machines, key)}((machs, dict))
end

report(N::Node) = item_given_machine(report, N)
report(::Source) = NamedTuple()

MLJModelInterface.fitted_params(N::Node) =
    item_given_machine(fitted_params, N)
MLJModelInterface.fitted_params(S::Source) = NamedTuple()


## SYNTACTIC SUGAR FOR LEARNING NETWORKS

"""
    N = node(f::Function, args...)

Defines a `Node` object `N` wrapping a static operation `f` and arguments
`args`. Each of the `n` elements of `args` must be a `Node` or `Source`
object. The node `N` has the following calling behaviour:

    N() = f(args[1](), args[2](), ..., args[n]())
    N(rows=r) = f(args[1](rows=r), args[2](rows=r), ..., args[n](rows=r))
    N(X) = f(args[1](X), args[2](X), ..., args[n](X))

    J = node(f, mach::Machine, args...)

Defines a dynamic `Node` object `J` wrapping a dynamic operation `f`
(`predict`, `predict_mean`, `transform`, etc), a nodal machine `mach` and
arguments `args`. Its calling behaviour, which depends on the outcome of
training `mach` (and, implicitly, on training outcomes affecting its
arguments) is this:

    J() = f(mach, args[1](), args[2](), ..., args[n]())
    J(rows=r) = f(mach, args[1](rows=r), args[2](rows=r), ..., args[n](rows=r))
    J(X) = f(mach, args[1](X), args[2](X), ..., args[n](X))

Generally `n=1` or `n=2` in this latter case.

    predict(mach, X::AbsractNode, y::AbstractNode)
    predict_mean(mach, X::AbstractNode, y::AbstractNode)
    predict_median(mach, X::AbstractNode, y::AbstractNode)
    predict_mode(mach, X::AbstractNode, y::AbstractNode)
    transform(mach, X::AbstractNode)
    inverse_transform(mach, X::AbstractNode)

Shortcuts for `J = node(predict, mach, X, y)`, etc.

Calling a node is a recursive operation which terminates in the call
to a source node (or nodes). Calling nodes on *new* data `X` fails unless the
number of such nodes is one.

See also: [`@node`](@ref), [`source`](@ref), [`origins`](@ref).

"""
const node = Node

"""
    @node f(...)

Construct a new node that applies the function `f` to some combination
of nodes, sources and other arguments.

*Important.* An argument not in global scope is assumed to be a node
 or source.

### Examples

```
X = source(π)
W = @node sin(X)
julia> W()
0

X = source(1:10)
Y = @node selectrows(X, 3:4)
julia> Y()
3:4

julia> Y(["one", "two", "three", "four"])
2-element Array{Symbol,1}:
 "three"
 "four"

X1 = source(4)
X2 = source(5)
add(a, b, c) = a + b + c
N = @node add(X1, 1, X2)
julia> N()
10

```

See also [`node`](@ref)

"""
macro node(ex)
    ex.head == :call || error("@node syntax error")
    exs = ex.args
    f_ex = first(exs)
    arg_exs = exs[2:end]

    # build lambda expression lambda_left -> lambda_right
    stuff =
        first.(map(arg_exs) do ex
                   pair = (:nothing, false)
                   try
                       evaluated = __module__.eval(ex)
                       if evaluated isa AbstractNode
                           pair =  gensym("node"), true
                       else
                           pair = ex, false
                       end
                   catch e
                       if e isa UndefVarError
                           pair = gensym("node"), true
                       else
                           error()
                       end
                   end
               end |> zip)
    right = first.(stuff)
    mask = last.(stuff)
    left = right[mask]
    lambda_left = Expr(:tuple, left...)
    lambda_right = Expr(:call, f_ex, right...)
    lambda_ex = Expr(:->, lambda_left, lambda_right)

    # the node-only arguments:
    node_arg_exs = arg_exs[mask]

    esc(quote
        node($lambda_ex, $(node_arg_exs...))
    end)
end

"""
    glb(N1, N2, ...)

Given nodes `N1`, `N2`, ... , construct a node `N` with the behaviour
`N() = (N1(), N2(), ...)`. That is, `glb` is `tuple` overloaded for
nodes.

Equivalent to `@tuple N1 N2 ...`

"""
glb(X::AbstractNode...) = node(tuple, X...)

"""
    @tuple N1 N2 ...

Construct a node `N` whose calling behaviour is `N() = (N1(), N2(), ...)`.

"""
macro tuple(ex...)
    esc(quote
        glb($(ex...))
        end)
end

"""
    nrows(X::AbstractNode)

Return a new node `N` such that `N() = nrows(X())` and `N(rows=rows) =
nrows(X(rows=rows))`. To obtain the number of rows of data at the
source of `X`, use `nrows_at_source(X)`.

"""
MLJModelInterface.nrows(X::AbstractNode) = node(nrows, X)

MMI.matrix(X::AbstractNode)      = node(matrix, X)
MMI.table(X::AbstractNode)       = node(table, X)
Base.vcat(args::AbstractNode...) = node(vcat, args...)
Base.hcat(args::AbstractNode...) = node(hcat, args...)
Statistics.mean(X::AbstractNode)   = node(v->mean.(v), X)
Statistics.median(X::AbstractNode) = node(v->median.(v), X)
StatsBase.mode(X::AbstractNode)    = node(v->mode.(v), X)
Base.log(X::AbstractNode) = node(v->log.(v), X)
Base.exp(X::AbstractNode) = node(v->exp.(v), X)
Base.first(X::AbstractNode) = node(first, X)
Base.last(X::AbstractNode) = node(last, X)

+(y1::AbstractNode, y2::AbstractNode) = node(+, y1, y2)
+(y1, y2::AbstractNode) = node(+, y1, y2)
+(y1::AbstractNode, y2) = node(+, y1, y2)
*(lambda::Real, y::AbstractNode) = node(y->lambda*y, y)

"""
    selectcols(X::AbstractNode, c)

Returns `Node` object `N` such that `N() = selectcols(X(), c)`.
"""
MMI.selectcols(X::AbstractNode, r) = node(XX->selectcols(XX, r), X)

"""
    selectrows(X::AbstractNode, r)

Returns a `Node` object `N` such that `N() = selectrows(X(), r)` (and
`N(rows=s) = selectrows(X(rows=s), r)`).

"""
MMI.selectrows(X::AbstractNode, r) = node(XX->selectrows(XX, r), X)

# for accessing and setting model hyperparameters at node:
getindex(n::Node{<:Machine{<:Model}}, s::Symbol) =
    getproperty(n.machine.model, s)
setindex!(n::Node{<:Machine{<:Model}}, v, s::Symbol) =
    setproperty!(n.machine.model, s, v)
