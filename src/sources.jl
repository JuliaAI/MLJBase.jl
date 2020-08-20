# Machines store data for training as *arguments*. Eg, in the
# supervised case, the first two arguments are always `X` and `y`. An
# argument could be raw data (such as a table) or, in the case of a
# learning network, a "promise" of data (aka "dynamic" data) -
# formally a `Node` object, which is *callable*. For uniformity of
# interface, even raw data is stored in a wrapper that can be called
# to return the object wrapped. The name of the wrapper type is
# `Source`, as these constitute the source nodes of learning networks.

# Here we define the `Source` wrapper, as well as some methods they
# will share with the `Node` type for use in learning networks.


## SOURCE TYPE

abstract type AbstractNode <: MLJType end
abstract type CallableReturning{K} end # scitype for sources and nodes

mutable struct Source <: AbstractNode
    data     # training data
    scitype::DataType
end

"""
    Xs = source(X=nothing)

Define, a learning network `Source` object, wrapping some input data
`X`, which can be `nothing` for purposes of exporting the network as
stand-alone model. For training and testing the unexported
network, appropriate vectors, tables, or other data containers are
expected.

The calling behaviour of a `Source` object is this:

    Xs() = X
    Xs(rows=r) = selectrows(X, r)  # eg, X[r,:] for a DataFrame
    Xs(Xnew) = Xnew

See also: [`@from_network`](@ref], [`sources`](@ref),
[`origins`](@ref), [`node`](@ref).

"""
source(X) = Source(X, scitype(X))
source() = source(nothing)

source(Xs::Source; args...) = Xs

MLJScientificTypes.scitype(X::Source) = CallableReturning{X.scitype}
ScientificTypes.elscitype(X::Source) = X.scitype
nodes(X::Source) = [X, ]
Base.isempty(X::Source) = X.data === nothing
nrows_at_source(X::Source) = nrows(X.data)

color(::Source) = :yellow

# make source nodes callable:
function (X::Source)(; rows=:)
    rows == (:) && return X.data
    return selectrows(X.data, rows)
end
(X::Source)(Xnew) = Xnew

"""
    rebind!(s, X)

Attach new data `X` to an existing source node `s`. Not a public
method.

"""
function rebind!(s::Source, X)
    s.data = X
    s.scitype = scitype(X)
    return s
end

origins(s::Source) = [s,]


## DISPLAY FOR SOURCES AND OTHER ABSTRACT NODES

# show within other objects:
function Base.show(stream::IO, object::AbstractNode)
    repr = simple_repr(typeof(object))
    str = "$repr $(handle(object))"
    printstyled(IOContext(stream, :color=> SHOW_COLOR),
                    str, bold=false, color=:blue)
    return nothing
end

# show when alone:
function Base.show(stream::IO, ::MIME"text/plain", source::Source)
    show(stream, source)
    print(stream, " \u23CE `$(elscitype(source))`")
    return nothing
end

