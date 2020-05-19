## LEARNING NETWORK MACHINES

surrogate(::Type{<:Deterministic})  = Deterministic()
surrogate(::Type{<:Probabilistic})  = Probabilistic()
surrogate(::Type{<:Unsupervised}) = Unsupervised()

"""
    args(signature)

Return a tuple of all source nodes of the learning network with
specified `signature`. Sources with `kind=:input` appear first,
followed by those with, repectively, `:target`,`:weights` and
`:other` kinds.

A learning network *signature* is a named tuple, such as
`(predict=yhat, transfrom=W)`, specifying what nodes of the network
are called to produce output of each operation represented by the
keys, in an exported version of the network.

"""
function args(signature::NamedTuple)

    glb_node = glb(values(signature)...) # greatest lower bound node

    inputs = sources(glb_node, kind=:input)
    targets = sources(glb_node, kind=:target)
    weights = sources(glb_node, kind=:weights)
    others = sources(glb_node, kind=:other)

    ret = vcat(inputs, targets, weights, others) |> Tuple

    return ret

end

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

    kinds = kind.(args(signature))
    kinds == (:input, ) && return Unsupervised

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

"""
    machine!(oper1=node1, oper2=node2, ...)

Construct a learning network machine, i.e., a machine directly
wrapping a learning network, without exporting the network. The
keyword arguments declare what nodes are called when operations
`predict`, `transform`, etc, are called on the machine.

In addition to the operations named in the constructor, the methods
`fit!`, `report`, and `fitted_params` can be applied as usual to the
machine constructed.

Learning network machines are constructed before exporting a learning
 network as a stand-alone model type. See the MLJ manual for details.

    machine!(X; kwargs...)
    machine!(X, y; kwargs...)
    machine!(X, y, W; kwargs...)

Same as above, but specifying explicitly the source nodes, to force
the roles (kinds) implied by the order in which they appear
(`:input`[, `:target`, `:weights`]).

    machine!(Probablistic(), args...; kwargs...)
    machine!(Deterministic(), args...; kwargs...)
    machine!(Unsupervised(), args...; kwargs...)

Same as above, but specifying explicitly the kind of model the
learning network is to meant to represent. 

For declaring scitype traits intended for an exported version of the
learning newtork, give the model constructor arguments, as in

    Probabilistic(input_scitype=Table(Continuous),
                  target_scitype=AbstractArray{<:Finite})

Learning network machines are not to be confused with a machine bound
to a stand-alone composite model (i.e., an *exported* learning
network).


### Examples

Supposing a supervised learning network's final predictions are
obtained by calling a node `yhat`, then the following code

```julia
mach = machine!(predict=yhat)
fit!(mach; rows=train)
predictions = predict(mach, Xnew) # `Xnew` concrete data
```
are equivalent to

```julia
fit!(yhat, rows=train)
predictions = yhat(Xnew)
```

In a unsupervised learning network that performs clustering, for
example, and in which the node `Xout` delivers the output of dimension
reduction, and `yhat` class labels, one can write

```julia
mach = machine!(transform=Xout, predict=yhat)
fit!(mach)
transformed = transform(mach, Xnew) # `Xnew` concrete data
predictions = predict(mach, Xnew)
```

which is equivalent to

```julia
fit!(Xout)
fit!(yhat)
transformed = Xout(Xnew)
predictions = yhat(Xnew)
```

"""
function machine!(model::Model, _sources::Source...; pair_itr...)

    # named tuple, such as `(predict=yhat, transform=W)`:
    signature = (; pair_itr...)
    if isempty(signature)
        model isa Supervised &&
            throw(ArgumentError(
                "You must specify at least `predict=<some node>`. "))
        model isa Unsupervised &&
            throw(ArgumentError(
                "You must specify at least `transform=<some node>`. "))
    end

    network_sources = args(signature)

    issubset(_sources, network_sources) ||
        error("Specified sources cannot be located in network. ")

    if isempty(_sources)
        sources = network_sources
        kinds = kind.(sources)
        kinds in [(:input, ),
                  (:input, :target),
                  (:input, :target, :weights)] ||
                      error(
                          "Encountered learning network sources "*
                          "with incompatible kinds, "*
                          "$kinds. \n"*
                          "Use one of the following to "*
                          "explicitly specify sources, whose "*
                          "kinds will be mutated appropriately:\n"*
                          "  `machine!(model, X; ...)`\n"*
                          "  `machine!(model, X, y; ...)`\n"*
                          "  `machine!(model, X, y, w; ...)`\n")
    else
        sources = _sources
        roles = (:input, :target, :weights)
        for i in eachindex(sources)
            s = sources[i]
            s.kind = roles[i]
        end
        if model isa Supervised
            length(sources) in [2, 3] ||
                error("Incorrect number of source nodes specified.\n"*
                    "Use  `machine!(model, X, y; ...)` or "*
                    "`machine!(model, X, y, w; ...)` when "*
                    "`model isa Supervised`. ")
        elseif model isa Unsupervised
            length(sources) == 1
                error("Incorrect number of source nodes specified.\n"*
            "Use `machine!(model, X; ...)` when "*
            "`model isa Unsupervised. `")
        else
            throw(DomainError)
        end
    end

    mach = Machine(model, sources...)

    mach.fitresult = signature
    mach.cache = anonymize!(mach.args)

    return mach

end

function machine!(sources::Source...; pair_itr...)

    signature = (; pair_itr...)

    isempty(signature) &&
        throw(ArgumentError(
            "You must specify at least `predict=<some node>` or "*
            "or `transform=<some node>`. "))

    T = model_supertype(signature)
    if T == nothing
        @warn "Unable to deduce surrogate model type. \n"*
        "Using Deterministic(). To override use:\n "*
        "`machine(Probabilistic(), ...)`"
        model = Deterministic()
    else
        model = surrogate(T)
    end

    return machine!(model, sources...; pair_itr...)

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

See also [`machine!`](@ref)

"""
function fit!(mach::Machine{<:Surrogate}; kwargs...)

    # de-anonymize sources:
    sources, data = mach.cache.sources, mach.cache.data
    for k in eachindex(sources)
        rebind!(sources[k], data[k])
    end

    signature = mach.fitresult
    glb_node = glb(values(signature)...) # greatest lower bound node
    fit!(glb_node; kwargs...)

    mach.cache = anonymize!(mach.args)
    mach.state += 1
    mach.report = report(glb_node)
    return mach

end

MLJModelInterface.fitted_params(mach::Machine{<:Surrogate}) =
    fitted_params(glb(values(mach.fitresult)...))
