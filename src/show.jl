## REGISTERING LABELS OF OBJECTS DURING ASSIGNMENT

const HANDLE_GIVEN_ID = Dict{UInt64,Symbol}()
SHOW_COLOR = true
"""
    color_on()

Enable color and bold output at the REPL, for enhanced display of MLJ objects.

"""
color_on() = (global SHOW_COLOR=true;)
"""
    color_off()

Suppress color and bold output at the REPL for displaying MLJ objects. 

"""
color_off() = (global SHOW_COLOR=false;)


macro colon(p)
    Expr(:quote, p)
end

"""
    @constant x = value

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
    return as_string[1]*"…"*as_string[end-1:end]
end

"""return abbreviated object id (as string)  or it's registered handle (as string) if this exists"""
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
Base.show(stream::IO, ::MIME"text/plain", t::NamedTuple) = pretty_nt(stream, t)
pretty_nt(t) = pretty_nt(stdout, t)
pretty_nt(stream::IO, t) = pretty_nt(stream, t, 0)
pretty_nt(stream, t, n) = print(stream, t)
function pretty_nt(stream, t::NamedTuple, n)
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
        pretty_nt(stream, value, n + length("$k = ") + 1)
        print(stream, ",")
    end
    print(stream, ")")
end


## OTHER EXPOSED SHOW METHODS

# string consisting of carriage return followed by indentation of length n:
crind(n) = "\n"*repeat(' ', n)

# trait to tag those objects to be displayed as constructed:
show_as_constructed(::Any) = false
show_as_constructed(::Type{<:Model}) = true

# short version of showing a `MLJType` object:
function Base.show(stream::IO, object::MLJType)
    id = objectid(object) 
    description = string(typeof(object).name.name)
    parameters = typeof(object).parameters
    p_string = ""
    if length(parameters) == 1
        p = parameters[1]
        p isa DataType && (p_string = string(p.name.name))
    end
    description *= "{"*p_string*"}"
    str = "$description @ $(handle(object))"
    if !isempty(fieldnames(typeof(object)))
        printstyled(IOContext(stream, :color=> SHOW_COLOR), str, bold=false, color=:blue)
    else
        print(stream, str)
    end
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
function Base.show(stream::IO, ::MIME"text/plain", object::MLJType, ::Val{false})
    _recursive_show(stream, object, 1, DEFAULT_SHOW_DEPTH)
end

function Base.show(stream::IO, ::MIME"text/plain", object, ::Val{true})
    pretty(stream, object)
end
pretty(stream::IO, object) = pretty(stream, object, 0, DEFAULT_SHOW_DEPTH + 1, 0)
pretty(stream, object, current_depth, depth, n) = show(stream, object)
function pretty(stream, object::M, current_depth, depth, n) where M<:MLJType
    if current_depth == depth
        show(stream, object)
    else
        prefix = string(coretype(typeof(object)))
        print(stream, prefix, "(")
        first_item = true
        for k in fieldnames(M)
            value =  getproperty(object, k)
            if !first_item
                print(stream, crind(n + length(prefix) + 1))
            else
                first_item = false
            end
            print(stream, "$k = ")
            pretty(stream, value, current_depth + 1, depth, n + length(prefix) + 1 + length("$k = "))
            print(stream, ",")
        end
        print(stream, ")")
        if current_depth == 0
            description = " @ $(handle(object))"
            printstyled(IOContext(stream, :color=> SHOW_COLOR), description, bold=false, color=:blue)
        end
    end
end


# version showing a `MLJType` object to arbitrary depth:
Base.show(stream::IO, object::M, depth::Int) where M<:MLJType =
    show(stream, object, depth, Val(show_as_constructed(M)))
Base.show(stream::IO, object::MLJType, depth::Int, ::Val{false}) =
    _recursive_show(stream, object, 1, depth)
Base.show(stream::IO, object::MLJType, depth::Int, ::Val{true}) = 
    pretty(stream, object, 0, 100, 0)

# for convenience:
Base.show(object::MLJType, depth::Int) = show(stdout, object, depth)


""" 
    @more

Entered at the REPL, equivalent to `show(ans, 100)`. Use to get a
recursive description of all fields of the last REPL value.

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
istoobig(s::Symbol) = false
istoobig(str::AbstractString) = length(str) > 50

## THE `_show` METHOD

# Note: The `_show` method controls how field values are displayed in
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

Generate a table of the field values of the `MLJType` object,
dislaying each value by calling the method `_show` on it. The
behaviour of `_show(stream, f)` is as follows:

1. If `f` is itself a `MLJType` object, then its short form is shown
and `_recursive_show` generates as separate table for each of its
field values (and so on, up to a depth of argument `depth`).

2. Otherwise `f` is displayed as "(omitted T)" where `T = typeof(f)`,
unless `istoobig(f)` is false (the `istoobig` fall-back for arbitrary
types being `true`). In the latter case, the long (ie,
MIME"plain/text") form of `f` is shown. To override this behaviour,
overload the `_show` method for the type in question. 

"""
function _recursive_show(stream::IO, object::MLJType, current_depth, depth)
    if depth == 0 || isempty(fieldnames(typeof(object)))
        println(stream, object)
    elseif current_depth <= depth 
        fields = fieldnames(typeof(object))
        print(stream, "#"^current_depth, " ")
        show(stream, object)
        println(stream, ": ")
#        println(stream)
        if isempty(fields)
            println(stream)
            return
        end
        for fld in fields
            fld_string = string(fld)*" "^(max(0,COLUMN_WIDTH - length(string(fld))))*"=>   "
            print(stream, fld_string)
            if isdefined(object, fld)
                _show(stream, getfield(object, fld))
                #           println(stream)
            else
                println(stream, "(undefined)")
                #           println(stream)
            end
        end
        println(stream)
        for fld in fields
            if isdefined(object, fld)
                subobject = getfield(object, fld)
                if isa(subobject, MLJType) && !isempty(fieldnames(typeof(subobject)))
                    _recursive_show(stream, getfield(object, fld),
                                    current_depth + 1, depth)
                end
            end
        end
    end
end






    
    
    
    


    


