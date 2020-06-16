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

const SOURCE_KINDS= [:input, :output, :target, :weights, :unknown]

abstract type AbstractNode <: MLJType end
abstract type CallableReturning{K} end # scitype for sources and nodes

mutable struct Source <: AbstractNode
    data     # training data
    kind::Symbol
    scitype::DataType
end

"""
    Xs = source(X)
    ys = source(y, kind=:target)
    ws = source(w, kind=:weight)

Define, respectively, learning network `Source` objects for wrapping
some input data `X` (`kind=:input`), some target data `y`, or some
sample weights `w`.  The values of each variable `X, y, w` can be
anything, even `nothing`, if the network is for exporting as a
stand-alone model only. For training and testing the unexported network,
appropriate vectors, tables, or other data containers are expected.

    Xs = source()
    ys = source(kind=:target)
    ws = source(kind=:weight)

Define source nodes wrapping `nothing` instead of concrete data. Such
definitions suffice when used in learning networks to be exported
without testing.

The calling behaviour of a `Source` object is this:

    Xs() = X
    Xs(rows=r) = selectrows(X, r)  # eg, X[r,:] for a DataFrame
    Xs(Xnew) = Xnew

See also: [`@from_network`](@ref], [`sources`](@ref),
[`origins`](@ref), [`node`](@ref).

"""
function source(X; kind=:input)
    kind in SOURCE_KINDS ||
        @warn "Source `kind` is not one of $SOURCE_KINDS. "
    return Source(X, kind, scitype(X))
end

source(X::Source; args...) = X
source(; args...) = source(nothing; args...)

kind(S::Source) = S.kind
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
    rebind!(s, X; kind=nothing)

Attach new data `X` to an existing source node `s` and optionally
change it's kind.

"""
function rebind!(s::Source, X; kind=nothing)
    s.data = X
    s.scitype = scitype(X)
    if kind != nothing
        kind in SOURCE_KINDS ||
            @warn "Source `kind` is not one of $SOURCE_KINDS. "
        s.kind = kind
    end
    return s
end

origins(s::Source) = [s,]


## DISPLAY FOR SOURCES AND OTHER ABSTRACT NODES

_extra(::Any) = ""
_extra(S::Source) = "$(S.kind) "
function Base.show(stream::IO, object::AbstractNode)
    print(stream, _extra(object))
    repr = simple_repr(typeof(object))
    str = "$repr $(handle(object))"
    if !isempty(fieldnames(typeof(object)))
        printstyled(IOContext(stream, :color=> SHOW_COLOR),
                    str, bold=false, color=:blue)
    else
        print(stream, str)
    end
    print(stream, " \u23CE `$(elscitype(object))`")
    return nothing
end
