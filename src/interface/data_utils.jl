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

MMI.int(::FI, x::CategoricalElement) = CategoricalArrays.order(x.pool)[x.level]

# ------------------------------------------------------------------------
# classes

MMI.classes(::FI, p::CategoricalPool) =
    [p[i] for i in invperm(CategoricalArrays.order(p))]

# older method that avoids inverting a permutation but has dict lookup:
# classes(p::CategoricalPool) = [p.valindex[p.invindex[v]] for v in p.levels]
MMI.classes(::FI, x::CategoricalElement) = classes(x.pool)

# ------------------------------------------------------------------------
# decoder

struct CategoricalDecoder{T,R}
    pool::CategoricalPool{T,R}
    invorder::Vector{Int}
end

MMI.decoder(::FI, x::CategoricalElement) =
    CategoricalDecoder(x.pool, sortperm(x.pool.order))

(d::CategoricalDecoder{T,R})(i::Integer) where {T,R} =
    CategoricalValue{T,R}(d.invorder[i], d.pool)
(d::CategoricalDecoder{String,R})(i::Integer) where R =
    CategoricalString{R}(d.invorder[i], d.pool)
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

MMI.nrows(::FI, ::Val{:table}, X) = Tables.rowcount(X)

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
