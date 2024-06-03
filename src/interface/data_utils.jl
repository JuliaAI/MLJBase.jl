# `vtrait` (internal method, not to be re-exported)
MMI.vtrait(::FI, X, s) = ScientificTypes.vtrait(X)
# ------------------------------------------------------------------------
# `categorical`
MMI.categorical(::FI, a...; kw...) = categorical(a...; kw...)

# ------------------------------------------------------------------------
# `matrix`
MMI.matrix(::FI, ::Val{:table}, X; kw...) = Tables.matrix(X; kw...)

# ------------------------------------------------------------------------
# int

MMI.int(::FI, x; args...) = CategoricalDistributions.int(x; args...)

# ------------------------------------------------------------------------
# classes

MMI.classes(::FI, x) = CategoricalDistributions.classes(x)

# ------------------------------------------------------------------------
# `scitype`
function MMI.scitype(::FI, ::Union{Val{:other}, Val{:table}}, X)
    return ScientificTypes.scitype(X)
end

# ------------------------------------------------------------------------
# `schema`
function MMI.schema(::FI, ::Union{Val{:other}, Val{:table}}, X)
    return ScientificTypes.schema(X)
end

# ------------------------------------------------------------------------
# decoder

MMI.decoder(::FI, x) = CategoricalDistributions.decoder(x)

# ------------------------------------------------------------------------
# `table`
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
# `nrows`, `selectrows`, `selectcols`
function MMI.nrows(::FI, ::Val{:table}, X)
    if Tables.rowaccess(X)
        rows = Tables.rows(X)
        return _nrows_rat(Base.IteratorSize(typeof(rows)), rows)
    else
        cols = Tables.columns(X)
        return _nrows_cat(cols)
    end
end

# number of rows for columnaccessed table
function _nrows_cat(cols)
    names = Tables.columnnames(cols)
    !isempty(names) || return 0
    return length(Tables.getcolumn(cols, names[1]))
end

# number of rows for rowaccessed table
_nrows_rat(::Base.HasShape, rows) = size(rows, 1)
_nrows_rat(::Base.HasLength, rows) = length(rows)
_nrows_rat(iter_size, rows) = length(collect(rows))

MMI.selectrows(::FI, ::Val{:table}, X, ::Colon) = X
MMI.selectcols(::FI, ::Val{:table}, X, ::Colon) = X

function MMI.selectrows(::FI, ::Val{:table}, X, r)
    r = r isa Integer ? (r:r) : r
    # next uncommented line is a hack; see
    # https://github.com/JuliaAI/MLJBase.jl/issues/151
    isdataframe(X) && return X[r, :]
    cols = Tables.columntable(X)
    new_cols = NamedTuple{keys(cols)}(tuple((c[r] for c in values(cols))...))
    return Tables.materializer(X)(new_cols)
end

function MMI.selectcols(::FI, ::Val{:table}, X, c::Union{Symbol, Integer})
    cols = Tables.columns(X)
    return Tables.getcolumn(cols, c)
end

function MMI.selectcols(::FI, ::Val{:table}, X, c::Union{Colon, AbstractArray})
    if isdataframe(X)
        return X[!, c]
    else
        cols = Tables.columntable(X) # named tuple of vectors
        newcols = project(cols, c)
        return Tables.materializer(X)(newcols)
    end
end

# -------------------------------
# utils for `select`*

# project named tuple onto a tuple with only specified `labels` or indices:
function project(t::NamedTuple, labels::AbstractArray{Symbol})
    return NamedTuple{tuple(labels...)}(t)
end

project(t::NamedTuple, label::Colon) = t
project(t::NamedTuple, label::Symbol) = project(t, [label,])
project(t::NamedTuple, i::Integer) = project(t, [i,])

function project(t::NamedTuple, indices::AbstractArray{<:Integer})
    return NamedTuple{tuple(keys(t)[indices]...)}(tuple([t[i] for i in indices]...))
end

# utils for selectrows
typename(X) = split(string(supertype(typeof(X))), '.')[end]
isdataframe(X) = typename(X) == "AbstractDataFrame"

# ----------------------------------------------------------------
# univariate finite

MMI.UnivariateFinite(::FI, b...; kwargs...) =
    CategoricalDistributions.UnivariateFinite(b...; kwargs...)
