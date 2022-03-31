## REGISTERING LABELS OF OBJECTS DURING ASSIGNMENT

"""
    color_on()

Enable color and bold output at the REPL, for enhanced display of MLJ objects.

"""
color_on() = (SHOW_COLOR[] = true;)
"""
    color_off()

Suppress color and bold output at the REPL for displaying MLJ objects.

"""
color_off() = (SHOW_COLOR[] = false;)


macro colon(p)
    Expr(:quote, p)
end

"""
    @constant x = value

Private method (used in testing).

Equivalent to `const x = value` but registers the binding thus:

    MLJBase.HANDLE_GIVEN_ID[objectid(value)] = :x

Registered objects get displayed using the variable name to which it
was bound in calls to `show(x)`, etc.

WARNING: As with any `const` declaration, binding `x` to new value of
the same type is not prevented and the registration will not be updated.

"""
macro constant(ex)
    ex.head == :(=) || throw(error("Expression must be an assignment."))
    handle = ex.args[1]
    value = ex.args[2]
    quote
        const $(esc(handle)) = $(esc(value))
        id = objectid($(esc(handle)))
        HANDLE_GIVEN_ID[id] = @colon $handle
        $(esc(handle))
    end
end

"""to display abbreviated versions of integers"""
function abbreviated(n)
    as_string = string(n)
    return "@"*as_string[end-2:end]
end

"""return abbreviated object id (as string) or it's registered handle
(as string) if this exists"""
function handle(X)
    id = objectid(X)
    if id in keys(HANDLE_GIVEN_ID)
        return string(HANDLE_GIVEN_ID[id])
    else
        return abbreviated(id)
    end
end


## SHOW METHOD FOR NAMED TUPLES

# long version of showing a named tuple:
Base.show(stream::IO, ::MIME"text/plain", t::NamedTuple) = fancy_nt(stream, t)
fancy_nt(t) = fancy_nt(stdout, t)
fancy_nt(stream::IO, t) = fancy_nt(stream, t, 0)
fancy_nt(stream, t, n) = show(stream, t)
function fancy_nt(stream, t::NamedTuple, n)
    print(stream, "(")
    first_item = true
    for k in keys(t)
        value =  getproperty(t, k)
        if !first_item
            print(stream, crind(n + 1))
        else
            first_item = false
        end
        print(stream, "$k = ")
        fancy_nt(stream, value, n + length("$k = ") + 1)
        print(stream, ",")
    end
    print(stream, ")")
end


## OTHER EXPOSED SHOW METHODS

# string consisting of carriage return followed by indentation of length n:
crind(n) = "\n"*repeat(' ', max(n, 0))

# trait to tag those objects to be displayed as constructed:
show_as_constructed(::Type) = false
show_as_constructed(::Type{<:Model}) = true
show_compact(::Type) = false

show_as_constructed(object) = show_as_constructed(typeof(object))
show_compact(object) = show_compact(typeof(object))
show_handle(object) = false

# simplified string rep of an Type:
function simple_repr(T)
    repr = string(T.name.name)
    parameters = T.parameters
    p_string = ""
    if length(parameters) > 0
        p = parameters[1]
        if p isa DataType
            p_string = simple_repr(p)
        elseif p isa Symbol
            p_string = string(":", p)
        end
        if length(parameters) > 1
            p_string *= ",…"
        end
    end
    isempty(p_string) || (repr *= "{"*p_string*"}")
    return repr
end

# short version of showing a `MLJType` object:
function Base.show(stream::IO, object::MLJType)
    str = simple_repr(typeof(object))
    show_handle(object) && (str *= " $(handle(object))")
    if false # !isempty(propertynames(object))
        printstyled(IOContext(stream, :color=> SHOW_COLOR[]),
                    str, bold=false, color=:blue)
    else
        print(stream, str)
    end
    return nothing
end

# longer versions of showing objects
function Base.show(stream::IO, T::MIME"text/plain", object::MLJType)
    show(stream, T, object, Val(show_as_constructed(typeof(object))))
end

# fallback:
function Base.show(stream::IO, ::MIME"text/plain", object, ::Val{false})
    show(stream, MIME("text/plain"), object)
end

# fallback for MLJType:
function Base.show(stream::IO, ::MIME"text/plain",
                   object::MLJType, ::Val{false})
    _recursive_show(stream, object, 1, DEFAULT_SHOW_DEPTH)
end

function Base.show(stream::IO, ::MIME"text/plain", object, ::Val{true})
    fancy(stream, object)
end
fancy(stream::IO, object) = fancy(stream, object, 0,
                                  DEFAULT_AS_CONSTRUCTED_SHOW_DEPTH, 0)
fancy(stream, object, current_depth, depth, n) = show(stream, object)
function fancy(stream, object::MLJType, current_depth, depth, n)
    if current_depth == depth
        show(stream, object)
    else
        prefix = MLJModelInterface.name(object)
        anti = max(length(prefix) - INDENT)
        print(stream, prefix, "(")
        names = propertynames(object)
        n_names = length(names)
        for k in eachindex(names)
            value =  getproperty(object, names[k])
            show_compact(object) ||
                print(stream, crind(n + length(prefix) - anti))
            print(stream, "$(names[k]) = ")
            fancy(stream, value, current_depth + 1, depth, n + length(prefix)
                  - anti + length("$k = "))
            k == n_names || print(stream, ",")
        end
        print(stream, ")")
        if current_depth == 0 && show_handle(object)
            description = " $(handle(object))"
            printstyled(IOContext(stream, :color=> SHOW_COLOR[]),
                        description, bold=false, color=:blue)
        end
    end
    return nothing
end


# version showing a `MLJType` object to arbitrary depth:
Base.show(stream::IO, object::M, depth::Int) where M<:MLJType =
    show(stream, object, depth, Val(show_as_constructed(M)))
Base.show(stream::IO, object::MLJType, depth::Int, ::Val{false}) =
    _recursive_show(stream, object, 1, depth)
Base.show(stream::IO, object::MLJType, depth::Int, ::Val{true}) =
    fancy(stream, object, 0, 100, 0)

# for convenience:
Base.show(object::MLJType, depth::Int) = show(stdout, object, depth)


"""
    @more

Entered at the REPL, equivalent to `show(ans, 100)`. Use to get a
recursive description of all properties of the last REPL value.

"""
macro more()
    esc(quote
        show(Main.ans, 100)
    end)
end


## METHODS TO SUPRESS THE DISPLAY OF LARGE NON-BASETYPE OBJECTS

istoobig(::Any) = true
istoobig(::DataType) = false
istoobig(::UnionAll) = false
istoobig(::Union) = false
istoobig(::Number) = false
istoobig(::Char) = false
istoobig(::Function) = false
istoobig(::Symbol) = false
istoobig(::Distributions.Distribution) = false
istoobig(str::AbstractString) = length(str) > 50

## THE `_show` METHOD

# Note: The `_show` method controls how properties are displayed in
# the table generated by `_recursive_show`. See top of file.

# _show fallback:
function _show(stream::IO, object)
    if !istoobig(object)
        show(stream, MIME("text/plain"), object)
        println(stream)
    else
        println(stream, "(omitted ", typeof(object), ")")
    end
end

_show(stream::IO, object::MLJType) = println(stream, object)

# _show for other types:

istoobig(t::Tuple{Vararg{T}}) where T<:Union{Number,Symbol,Char,MLJType} =
    length(t) > 5
function _show(stream::IO, t::Tuple)
    if !istoobig(t)
        show(stream, MIME("text/plain"), t)
        println(stream)
    else
        println(stream, "(omitted $(typeof(t)) of length $(length(t)))")
    end
end

istoobig(A::AbstractArray{T}) where T<:Union{Number,Symbol,Char,MLJType} =
    maximum(size(A)) > 5
function _show(stream::IO, A::AbstractArray)
    if !istoobig(A)
        show(stream, MIME("text/plain"), A)
        println(stream)
    else
        println(stream, "(omitted $(typeof(A)) of size $(size(A)))")
    end
end

istoobig(d::Dict{T,Any}) where T <: Union{Number,Symbol,Char,MLJType} =
    length(keys(d)) > 5
function _show(stream::IO, d::Dict{T, Any}) where T <: Union{Number,Symbol}
    if isempty(d)
        println(stream, "empty $(typeof(d))")
    elseif !istoobig(d)
        println(stream, "omitted $(typeof(d)) with keys: ")
        show(stream, MIME("text/plain"), collect(keys(d)))
        println(stream)
    else
        println(stream, "(omitted $(typeof(d)))")
    end
end

function _show(stream::IO, v::Array{T, 1}) where T
    if !istoobig(v)
        show(stream, MIME("text/plain"), v)
        println(stream)
    else
        println(stream, "(omitted Vector{$T} of length $(length(v)))")
    end
end

_show(stream::IO, T::DataType) = println(stream, T)

_show(stream::IO, ::Nothing) = println(stream, "nothing")


## THE RECURSIVE SHOW METHOD

"""
    _recursive_show(stream, object, current_depth, depth)

Generate a table of the properties of the `MLJType` object, dislaying
each property value by calling the method `_show` on it. The behaviour
of `_show(stream, f)` is as follows:

1. If `f` is itself a `MLJType` object, then its short form is shown
and `_recursive_show` generates as separate table for each of its
properties (and so on, up to a depth of argument `depth`).

2. Otherwise `f` is displayed as "(omitted T)" where `T = typeof(f)`,
unless `istoobig(f)` is false (the `istoobig` fall-back for arbitrary
types being `true`). In the latter case, the long (ie,
MIME"plain/text") form of `f` is shown. To override this behaviour,
overload the `_show` method for the type in question.

"""
function _recursive_show(stream::IO, object::MLJType, current_depth, depth)
    if depth == 0 || isempty(propertynames(object))
        println(stream, object)
    elseif current_depth <= depth
        fields = propertynames(object)
        print(stream, "#"^current_depth, " ")
        show(stream, object)
        println(stream, ": ")
#        println(stream)
        if isempty(fields)
            println(stream)
            return
        end
        for fld in fields
            fld_string = string(fld)*
                " "^(max(0,COLUMN_WIDTH - length(string(fld))))*"=>   "
            print(stream, fld_string)
            if isdefined(object, fld)
                _show(stream, getproperty(object, fld))
                #           println(stream)
            else
                println(stream, "(undefined)")
                #           println(stream)
            end
        end
        println(stream)
        for fld in fields
            if isdefined(object, fld)
                subobject = getproperty(object, fld)
                if isa(subobject, MLJType) &&
                    !isempty(propertynames(subobject))
                    _recursive_show(stream, getproperty(object, fld),
                                    current_depth + 1, depth)
                end
            end
        end
    end
    return nothing
end
