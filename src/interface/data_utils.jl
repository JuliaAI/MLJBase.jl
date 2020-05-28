# ------------------------------------------------------------------------
# categorical

MMI.categorical(::FI, a...; kw...) = categorical(a...; kw...)

# ------------------------------------------------------------------------
# matrix

MMI.matrix(::FI, ::Val{:table}, X; kw...) = Tables.matrix(X; kw...)

# ------------------------------------------------------------------------
# int

MMI.int(::FI, x) = throw(
    DomainError(x, "Can only convert categorical elements to integers. "))

MMI.int(::FI, x::Missing)       = missing
MMI.int(::FI, x::AbstractArray) = int.(x)

# first line is no good because it promotes type to higher ineger type:
# MMI.int(::FI, x::CategoricalValue) = CategoricalArrays.levelcode(x)
MMI.int(::FI, x::CategoricalValue) = CategoricalArrays.level(x)

# ------------------------------------------------------------------------
# classes

MMI.classes(::FI, p::CategoricalPool) = [p[i] for i in 1:length(p)]
MMI.classes(::FI, x::CategoricalValue) = classes(CategoricalArrays.pool(x))
MMI.classes(::FI, v::CategoricalArray) = classes(CategoricalArrays.pool(v))

# ------------------------------------------------------------------------
# schema

MMI.schema(::FI, ::Val{:table}, X; kw...) = schema(X; kw...)

# ------------------------------------------------------------------------
# decoder

struct CategoricalDecoder{T,R}
    classes::CategoricalVector{T,R}
end

MMI.decoder(::FI, x::CategoricalValue) =
    CategoricalDecoder(classes(x))
MMI.decoder(::FI, v::CategoricalArray) = decoder(first(v))

(d::CategoricalDecoder{T,R})(i::Integer) where {T,R} =
    CategoricalValue{T,R}(d.classes[i])
(d::CategoricalDecoder)(a::AbstractArray{<:Integer}) = d.(a)

# ------------------------------------------------------------------------
# table

function MMI.table(::FI, cols::NamedTuple; prototype=NamedTuple())
    Tables.istable(prototype) || error("`prototype` is not a table. ")
    if !Tables.istable(cols)
        tuple_of_vectors = tuple((collect(v) for v in values(cols))...)
        names = keys(cols)
        cols = NamedTuple{names}(tuple_of_vectors)
    end
    return Tables.materializer(prototype)(cols)
end

function MMI.table(::FI, A::AbstractMatrix; names=nothing, prototype=nothing)
    if names === nothing
        _names = [Symbol(:x, j) for j in 1:size(A, 2)]
    else
        _names = collect(names)
    end
    matrix_table = Tables.table(A, header=_names)
    prototype === nothing && return matrix_table
    return Tables.materializer(prototype)(matrix_table)
end

# ------------------------------------------------------------------------
# nrows, selectrows, selectcols

function MMI.nrows(::FI, ::Val{:table}, X)
    Tables.columnaccess(X) || return length(collect(X))
    # if has columnaccess
    cols = Tables.columntable(X)
    !isempty(cols) || return 0
    return length(cols[1])
end

MMI.selectrows(::FI, ::Val{:table}, X, ::Colon) = X
MMI.selectcols(::FI, ::Val{:table}, X, ::Colon) = X

function MMI.selectrows(::FI, ::Val{:table}, X, r)
    r = r isa Integer ? (r:r) : r
    # next uncommented line is a hack; see
    # https://github.com/alan-turing-institute/MLJBase.jl/issues/151
    isdataframe(X) && return X[r, :]
    cols = Tables.columntable(X)
    new_cols = NamedTuple{keys(cols)}(tuple((c[r] for c in values(cols))...))
    return Tables.materializer(X)(new_cols)
end

function MMI.selectcols(::FI, ::Val{:table}, X, c::Union{Symbol,Integer})
    cols = Tables.columntable(X) # named tuple of vectors
    return cols[c]
end

function MMI.selectcols(::FI, ::Val{:table}, X, c::AbstractArray)
    cols = Tables.columntable(X) # named tuple of vectors
    newcols = project(cols, c)
    return Tables.materializer(X)(newcols)
end

# -------------------------------
# utils for select*

# project named tuple onto a tuple with only specified `labels` or indices:
project(t::NamedTuple, labels::AbstractArray{Symbol}) =
    NamedTuple{tuple(labels...)}(t)
project(t::NamedTuple, label::Colon) = t
project(t::NamedTuple, label::Symbol) = project(t, [label,])
project(t::NamedTuple, indices::AbstractArray{<:Integer}) =
    NamedTuple{tuple(keys(t)[indices]...)}(tuple([t[i] for i in indices]...))
project(t::NamedTuple, i::Integer) = project(t, [i,])

# utils for selectrows
typename(X)    = split(string(supertype(typeof(X)).name), '.')[end]
isdataframe(X) = typename(X) == "AbstractDataFrame"

