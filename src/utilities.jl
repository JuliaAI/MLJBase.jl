function finaltypes(T::Type)
    s = InteractiveUtils.subtypes(T)
    if isempty(s)
        return [T, ]
    else
        return reduce(vcat, [finaltypes(S) for S in s])
    end
end

"""
    flat_values(t::NamedTuple)

View a nested named tuple `t` as a tree and return, as a tuple, the values
at the leaves, in the order they appear in the original tuple.

```julia-repl
julia> t = (X = (x = 1, y = 2), Y = 3);
julia> flat_values(t)
(1, 2, 3)
```

"""
function flat_values(params::NamedTuple)
    values = []
    for k in keys(params)
        value = getproperty(params, k)
        if value isa NamedTuple
            append!(values, flat_values(value))
        else
            push!(values, value)
        end
    end
    return Tuple(values)
end

## RECURSIVE VERSIONS OF getproperty and setproperty!

# applying the following to `:(a.b.c)` returns `(:(a.b), :c)`
function reduce_nested_field(ex)
    ex.head == :. || throw(ArgumentError)
    tail = ex.args[2]
    tail isa QuoteNode || throw(ArgumentError)
    field = tail.value
    field isa Symbol || throw(ArgumentError)
    subex = ex.args[1]
    return (subex, field)
end

"""
    MLJBase.prepend(::Symbol, ::Union{Symbol,Expr,Nothing})

For prepending symbols in expressions like `:(y.w)` and `:(x1.x2.x3)`.

```julia-repl
julia> prepend(:x, :y)
:(x.y)

julia> prepend(:x, :(y.z))
:(x.y.z)

julia> prepend(:w, ans)
:(w.x.y.z)
```

If the second argument is `nothing`, then `nothing` is returned.

"""
prepend(s::Symbol, ::Nothing) = nothing
prepend(s::Symbol, t::Symbol) = Expr(:(.), s, QuoteNode(t))
prepend(s::Symbol, ex::Expr) = Expr(:(.), prepend(s, ex.args[1]), ex.args[2])


"""
    recursive_getproperty(object, nested_name::Expr)

Call getproperty recursively on `object` to extract the value of some
nested property, as in the following example:

```julia-repl
julia> object = (X = (x = 1, y = 2), Y = 3);
julia> recursive_getproperty(object, :(X.y))
2
```
"""
recursive_getproperty(obj, property::Symbol) = getproperty(obj, property)
function recursive_getproperty(obj, ex::Expr)
    subex, field = reduce_nested_field(ex)
    return recursive_getproperty(recursive_getproperty(obj, subex), field)
end

recursive_getpropertytype(obj, property::Symbol) = typeof(getproperty(obj, property))
recursive_getpropertytype(obj::T, property::Symbol) where T <: Model = begin
    model_type = typeof(obj)
    property_names = fieldnames(model_type)
    property_types = model_type.types
    for (t, n) in zip(property_types, property_names)
        n == property && return t
    end
    error("Property $property not found")
end
function recursive_getpropertytype(obj, ex::Expr)
    subex, field = reduce_nested_field(ex)
    return recursive_getpropertytype(recursive_getproperty(obj, subex), field)
end

"""
    recursively_setproperty!(object, nested_name::Expr, value)

Set a nested property of an `object` to `value`, as in the following example:

```julia-repl
julia> mutable struct Foo
           X
           Y
       end

julia> mutable struct Bar
           x
           y
       end

julia> object = Foo(Bar(1, 2), 3)
Foo(Bar(1, 2), 3)

julia> recursively_setproperty!(object, :(X.y), 42)
42

julia> object
Foo(Bar(1, 42), 3)
```

"""
recursive_setproperty!(obj, property::Symbol, value) =
    setproperty!(obj, property, value)
function recursive_setproperty!(obj, ex::Expr, value)
    subex, field = reduce_nested_field(ex)
    last_obj = recursive_getproperty(obj, subex)
    return recursive_setproperty!(last_obj, field, value)
end

"""
    check_same_nrows(X, Y)

Internal function to check two objects, each a vector or a matrix,
have the same number of rows.

"""
@inline function check_same_nrows(X, Y)
    size(X, 1) == size(Y, 1) ||
        throw(DimensionMismatch("The two objects don't have the same " *
                                "number of rows."))
    return nothing
end

"""
    _permute_rows(obj, perm)

Internal function to return a vector or matrix with permuted rows given
the permutation `perm`.
"""
function _permute_rows(obj::AbstractVecOrMat, perm::Vector{Int})
    check_same_nrows(obj, perm)
    obj isa AbstractVector && return obj[perm]
    obj[perm, :]
end

"""
    shuffle_rows(X::AbstractVecOrMat,
                 Y::AbstractVecOrMat;
                 rng::AbstractRNG=Random.GLOBAL_RNG)

Return row-shuffled vectors or matrices using a random permutation of `X`
and `Y`. An optional random number generator can be specified using
the `rng` argument.

"""
function shuffle_rows(
    X::AbstractVecOrMat, Y::AbstractVecOrMat;
    rng::AbstractRNG=Random.GLOBAL_RNG
)
    check_same_nrows(X, Y)
    perm_length = size(X, 1)
    perm = randperm(rng, perm_length)
    return _permute_rows(X, perm), _permute_rows(Y, perm)
end

"""
    init_rng(rng)

Create an `AbstractRNG` from `rng`. If `rng` is a non-negative `Integer`, it returns a
`MersenneTwister` random number generator seeded with `rng`; If `rng` is
an `AbstractRNG` object it returns `rng`, otherwise it throws an error.
"""
function init_rng(rng)
    if (rng isa Integer && rng > 0)
        return Random.MersenneTwister(rng)
    elseif !(rng isa AbstractRNG)
        throw(
            ArgumentError(
                "`rng` must either be a non-negative `Integer`, "*
                "or an `AbstractRNG` object."
            )
        )
    end
    return rng
end


## FOR PRETTY PRINTING

# of coloumns:
function pretty(io::IO, X; showtypes=true, alignment=:l, kwargs...)
    # for getting rid of bold in table headings:
    style = PrettyTables.TextTableStyle(
        first_line_column_label = PrettyTables.crayon"black",
    )

    _names = schema(X).names |> collect
    if showtypes
        types = schema(X).types |> collect
        scitypes = schema(X).scitypes |> collect
        header = [_names, types, scitypes]
    else
        header  = [_names, ]
    end
    show_color = MLJBase.SHOW_COLOR[]
    color_off()
    try
        PrettyTables.pretty_table(io, MLJBase.matrix(X),
                                  column_labels=header;
                                  alignment,
                                  style,
                                  kwargs...)
    catch excptn
        println("Trouble displaying table.")
        rethrow(excptn)
    end
    show_color ? color_on() : color_off()
    return nothing
end

pretty(X; kwargs...) = pretty(stdout, X; kwargs...)

# of long vectors (returns a compact string version of a vector):
function short_string(v::Vector)
    L = length(v)
    if L <= 3
        middle = join(v, ", ")
    else
        middle = string(round3(v[1]), ", ", round3(v[2]),
                        ", ..., ", round3(v[end]))
    end
    return "[$middle]"
end

"""
    sequence_string(itr, n=3)

Return a "sequence" string from the first `n` elements generated
by `itr`.

```julia-repl
julia> MLJBase.sequence_string(1:10, 4)
"1, 2, 3, 4, ..."
```

**Private method.**

"""
function sequence_string(itr::Itr, n=3) where Itr
    n > 1 || throw(ArgumentError("Cutoff must be at least 2. "))
    I = Base.IteratorSize(Itr)
    I isa Base.HasLength ||
        I isa Base.HasShape ||
        I isa IsInfinite ||
        throw(Argumenterror("Unsupported iterator. "))
    vals = String[]
    i = 0
    earlystop = false
    for x in itr
        i += 1
        if i === n + 1
            earlystop = true
            break
        else
            push!(vals, string(x))
        end
    end
    ret = join(vals, ", ")
    earlystop && (ret *= ", ...")
    return ret
end


## UNWINDING ITERATORS

"""
    unwind(iterators...)

Represent all possible combinations of values generated by `iterators`
as rows of a matrix `A`. In more detail, `A` has one column for each
iterator in `iterators` and one row for each distinct possible
combination of values taken on by the iterators. Elements in the first
column cycle fastest, those in the last clolumn slowest.

### Example

```julia-repl
julia> iterators = ([1, 2], ["a","b"], ["x", "y", "z"]);
julia> MLJTuning.unwind(iterators...)
12×3 Matrix{Any}:
 1  "a"  "x"
 2  "a"  "x"
 1  "b"  "x"
 2  "b"  "x"
 1  "a"  "y"
 2  "a"  "y"
 1  "b"  "y"
 2  "b"  "y"
 1  "a"  "z"
 2  "a"  "z"
 1  "b"  "z"
 2  "b"  "z"
```

"""
function unwind(iterators...)
    n_iterators = length(iterators)
    iterator_lengths = map(length, iterators)

    # product of iterator lengths:
    L = reduce(*, iterator_lengths)
    L != 0 || error("Parameter iterator of length zero encountered.")

    A = Array{Any}(undef, L, n_iterators)  ## TODO: this can be done better
    n_iterators != 0 || return A

    inner = 1
    outer = L
    for j in 1:n_iterators
        outer = outer ÷ iterator_lengths[j]
        A[:,j] = repeat(iterators[j], inner=inner, outer=outer)
        inner *= iterator_lengths[j]
    end
    return A
end


"""
    chunks(range, n)

Split an `AbstractRange`  into `n` subranges of approximately equal length.

### Example
```julia-repl
julia> collect(chunks(1:5, 2))
2-element Vector{UnitRange{Int64}}:
 1:3
 4:5
```

**Private method**

"""
function chunks(c::AbstractRange, n::Integer)
    n < 1 && throw(ArgumentError("cannot split range into $n subranges"))
    return Chunks(c, divrem(length(c), Int(n))...)
end

struct Chunks{T <: AbstractRange}
    range::T
    div::Int
    rem::Int
end

Base.eltype(::Type{Chunks{T}}) where {T <: AbstractRange} = T

function Base.length(itr::Chunks{<:AbstractRange})
   l = length(itr.range)
   return itr.div == 0 ? l : div(l - itr.rem, itr.div)
end

function Base.iterate(itr::Chunks{<:AbstractRange}, state=(1,itr.rem))
    first(state) > length(itr.range) && return nothing
    rem = last(state)
    r = min(first(state) + itr.div - (rem > 0 ? 0 : 1),
                length(itr.range))
   return @inbounds itr.range[first(state):r], (r + 1, rem-1)
end


"""
    available_name(modl::Module, name::Symbol)

Function to replace, if necessary, a given `name` with a modified one
that ensures it is not the name of any existing object in the global
scope of `modl`. Modifications are created with numerical suffixes.

"""
function available_name(modl, name)
    new_name = name
    i = 1
    while isdefined(modl, Symbol(new_name))
        i += 1
        new_name = string(name, i) |> Symbol
    end
    return new_name
end

"""
    generate_name!(M, existing_names; only=Union{Function,Type}, substitute=:f)

Given a type `M` (e.g., `MyEvenInteger{N}`) return a symbolic,
snake-case, representation of the type name (such as
`my_even_integer`). The symbol is pushed to `existing_names`, which
must be an `AbstractVector` to which a `Symbol` can be pushed.

If the snake-case representation already exists in `existing_names` a
suitable integer is appended to the name.

If `only` is specified, then the operation is restricted to those `M`
for which `M isa only`. In all other cases the symbolic name is
generated using `substitute` as the base symbol.

```julia-repl
julia> existing_names = [];
julia> generate_name!(Vector{Int}, existing_names)
:vector

julia> generate_name!(Vector{Int}, existing_names)
:vector2

julia> generate_name!(AbstractFloat, existing_names)
:abstract_float

julia> generate_name!(Int, existing_names, only=Array, substitute=:not_array)
:not_array

julia> generate_name!(Int, existing_names, only=Array, substitute=:not_array)
:not_array2
```

"""
function generate_name!(M::DataType,
                        existing_names;
                        only=Any,
                        substitute=:f)
    if M <: only
        str = split(string(M), '{') |> first
        candidate = split(str, '.') |> last |> snakecase |> Symbol
    else
        candidate = substitute
    end

    candidate in existing_names ||
        (push!(existing_names, candidate); return candidate)
    n = 2
    new_candidate = candidate
    while true
        new_candidate = string(candidate, n) |> Symbol
        new_candidate in existing_names || break
        n += 1
    end
    push!(existing_names, new_candidate)
    return new_candidate
end

generate_name!(model, existing_names; kwargs...) =
    generate_name!(typeof(model), existing_names; kwargs...)


# # OBSERVATION VS CONTAINER HACKINGS TOOLS

# The following tools are used to bridge the gap between old paradigm of prescribing
# the scitype of containers of observations, and the LearnAPI.jl paradigm of prescribing
# only the scitype of the observations themeselves. This is needed because measures are
# now taken from StatisticalMeasures.jl which follows the LearnAPI.jl paradigm, but model
# `target_scitype` refers to containers.

"""
    observation(S)

*Private method.*

Tries to infer the per-observation scitype from the scitype of `S`, when `S` is known to
be the scitype of some container with multiple observations. Return `Unknown` if unable to
draw reliable inferrence.

The observation scitype for a table is here understood as the scitype of a row
converted to a vector.

"""
observation(::Type) = Unknown
observation(::Type{AbstractVector{S}}) where S = S
observation(::Type{AbstractArray{S,N}}) where {S,N} = AbstractArray{S,N-1}
for T in [:Continuous, :Count, :Finite, :Infinite, :Multiclass, :OrderedFactor]
    TM = "Union{Missing,$T}" |> Meta.parse
    for S in [T, TM]
        quote
            observation(::Type{AbstractVector{<:$S}}) = $S
            observation(::Type{AbstractArray{<:$S,N}}) where N = AbstractArray{<:$S,N-1}
            observation(::Type{Table{<:AbstractVector{<:$S}}}) = AbstractVector{<:$S}
        end |> eval
    end
end
# note that in Julia `f(::Type{AbstractVector{<:T}}) where T = T` has not a well-formed
# left-hand side

"""
    guess_observation_scitype(y)

*Private method.*

If `y` is an `AbstractArray`, return the scitype of `y[:, :, ..., :, 1]`. If `y`
is a table, return the scitype of the first row, converted to a vector, unless
this row has `missing` elements, in which case return `Unknown`.

In all other cases, `Unknown`.

```julia-repl
julia> guess_observation_scitype([missing, 1, 2, 3])
Union{Missing, Count}

julia> guess_observation_scitype(rand(3, 2))
AbstractVector{Continuous}

julia> guess_observation_scitype((x=rand(3), y=rand(Bool, 3)))
AbstractVector{Union{Continuous, Count}}

julia> guess_observation_scitype((x=[missing, 1, 2], y=[1, 2, 3]))
Unknown
```
"""
guess_observation_scitype(y) = guess_observation_scitype(y, Val(Tables.istable(y)))
guess_observation_scitype(y, ::Any) = Unknown
guess_observation_scitype(y::AbstractArray, ::Val{false}) = observation(scitype(y))
function guess_observation_scitype(table, ::Val{true})
    row = Tables.subset(table, 1, viewhint=false) |> collect
    E = eltype(row)
    nonmissingtype(E) == E || return Unknown
    scitype(row)
end

"""
    guess_model_targetobservation_scitype(model)

*Private method*

Try to infer a lowest upper bound on the scitype of target observations
acceptable to `model`, by inspecting `target_scitype(model)`. Return `Unknown`
if unable to draw reliable inferrence.

The observation scitype for a table is here understood as the scitype of a row
converted to a vector.

"""
guess_model_target_observation_scitype(model) =  observation(target_scitype(model))
