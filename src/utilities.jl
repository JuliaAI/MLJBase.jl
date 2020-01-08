function coretype(M)
    if isdefined(M, :name)
        return M.name
    else
        return coretype(M.body)
    end
end

function finaltypes(T::Type)
    s = InteractiveUtils.subtypes(T)
    if isempty(s)
        return [T, ]
    else
        return reduce(vcat, [finaltypes(S) for S in s])
    end
end


# NOTE: deprecated, see @mlj_model
"""

    @set_defaults ModelType(args...)
    @set_defaults ModelType args

Create a keyword constructor for any type `ModelType<:MLJBase.Model`,
using as default values those listed in `args`. These must include a
value for every field, and in the order appearing in
`fieldnames(ModelType)`.

The constructor created calls `MLJBase.clean!(model)` on the
instantiated object `model` and calls `@warn messsage` if `messsage =
MLJBase.clean!(model)` is non-empty. Note that `MLJBase.clean!` has a
trivial fallback defined for all subtypes of `MLJBase.Model`.

### Example

   mutable struct Foo
      x::Int
      y
   end

   @set_defaults Foo(1,2)

   julia> Foo()
   Foo(1, 2)

   julia> Foo(x=1, y="house")
   Foo(1, "house")

   @set_defaults Foo [4, 5]

   julia> Foo()
   Foo(4, 5)

"""
macro set_defaults(ex)
    T_ex = ex.args[1]
    value_exs = ex.args[2:end]
    values = [__module__.eval(ex) for ex in value_exs]
    set_defaults_(__module__, T_ex, values)
    return nothing
end

macro set_defaults(T_ex, values_ex)
    values =__module__.eval(values_ex)
    set_defaults_(__module__, T_ex, values)
    return nothing
end

function set_defaults_(mod, T_ex, values)
    T = mod.eval(T_ex)
    fields = fieldnames(T)
    length(fields) == length(values) ||
        error("Provide the same number of default values as fields. ")

    equality_pair_exs = [Expr(:kw, fields[i], values[i]) for i in
                         eachindex(values)]

    program = quote
        $T_ex(; $(equality_pair_exs...)) =
            $T_ex($(fields...))
    end
    mod.eval(program)

end

"""
    flat_values(t::NamedTuple)

View a nested named tuple `t` as a tree and return, as a tuple, the values
at the leaves, in the order they appear in the original tuple.

```julia-repl
julia> t = (X = (x = 1, y = 2), Y = 3)
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
    field isa Symbol || throw(ArgmentError)
    subex = ex.args[1]
    return (subex, field)
end

"""
    recursive_getproperty(object, nested_name::Expr)

Call getproperty recursively on `object` to extract the value of some
nested property, as in the following example:

    julia> object = (X = (x = 1, y = 2), Y = 3)
    julia> recursive_getproperty(object, :(X.y))
    2

"""
recursive_getproperty(obj, property::Symbol) = getproperty(obj, property)
function recursive_getproperty(obj, ex::Expr)
    subex, field = reduce_nested_field(ex)
    return recursive_getproperty(recursive_getproperty(obj, subex), field)
end

"""
    recursively_setproperty!(object, nested_name::Expr, value)

Set a nested property of an `object` to `value`, as in the following example:

```
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
    check_dimension(X, Y)

Check that two vectors or matrices have matching dimensions
"""
function check_dimensions(X::AbstractVecOrMat, Y::AbstractVecOrMat)
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
    check_dimensions(obj, perm)
    obj isa AbstractVector && return obj[perm]
    obj[perm, :]
end

"""
shuffle_rows(X, Y, ...; rng=)

Return a shuffled view of a vector or  matrix `X` (or set of such) using a
random permutation (which can be seeded specifying `rng`).
"""
function shuffle_rows(X::AbstractVecOrMat, Y::AbstractVecOrMat; rng=nothing)
    check_dimensions(X, Y)
    rng === nothing || Random.seed!(rng)
    perm = randperm(size(X, 1))
    return _permute_rows(X, perm), _permute_rows(Y, perm)
end


## FOR PRETTY PRINTING COLUMN TABLES

function pretty(io::IO, X; showtypes=true, alignment=:l, kwargs...)
    names = schema(X).names |> collect
    if showtypes
        types = schema(X).types |> collect
        scitypes = schema(X).scitypes |> collect
        header = hcat(names, types, scitypes) |> permutedims
    else
        header  = names
    end
    show_color = MLJBase.SHOW_COLOR
    color_off()
    try
        PrettyTables.pretty_table(io, MLJBase.matrix(X),
                                  header; alignment=alignment, kwargs...)
    catch
        println("Trouble displaying table.")
    end
    show_color ? color_on() : color_off()
    return nothing
end

pretty(X; kwargs...) = pretty(stdout, X; kwargs...)
