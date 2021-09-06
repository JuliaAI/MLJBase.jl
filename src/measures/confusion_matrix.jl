## CONFUSION MATRIX OBJECT

"""
    ConfusionMatrixObject{C}

Confusion matrix with `C ≥ 2` classes. Rows correspond to predicted values
and columns to the ground truth.
"""
struct ConfusionMatrixObject{C}
    mat::Matrix
    labels::Vector{String}
end

"""
    ConfusionMatrixObject(m, labels)

Instantiates a confusion matrix out of a square integer matrix `m`.
Rows are the predicted class, columns the ground truth. See also the
[wikipedia article](https://en.wikipedia.org/wiki/Confusion_matrix).

"""
function ConfusionMatrixObject(m::Matrix{Int}, labels::Vector{String})
    s = size(m)
    s[1] == s[2] || throw(ArgumentError("Expected a square matrix."))
    s[1] > 1 || throw(ArgumentError("Expected a matrix of size ≥ 2x2."))
    length(labels) == s[1] ||
        throw(ArgumentError("As many labels as classes must be provided."))
    ConfusionMatrixObject{s[1]}(m, labels)
end

# allow to access cm[i,j] but not set (it's immutable)
Base.getindex(cm::ConfusionMatrixObject, inds...) = getindex(cm.mat, inds...)

_levels(y1, y2) = vcat(levels(y1), levels(y2)) |> unique

# simultaneous coercion of two vectors into categorical vectors having
# the same pool:
function _categorical(y1, y2)
    L = _levels(y1, y2)
    return categorical(y1, levels=L), categorical(y2, levels=L)
end
_categorical(y1::CategoricalArray{V1,N},
             y2::CategoricalArray{V2,N}) where
    {V, V1<:Union{Missing,V}, V2<:Union{Missing,V}, N} =
    y1, y2

"""
    _confmat(ŷ, y; rev=false)

A private method. General users should use `confmat` or other instances
of the measure type [`ConfusionMatrix`](@ref).

Computes the confusion matrix given a predicted `ŷ` with categorical elements
and the actual `y`. Rows are the predicted class, columns the ground truth.
The ordering follows that of `levels(y)`.

## Keywords

* `rev=false`: in the binary case, this keyword allows to swap the ordering of
               classes.
* `perm=[]`:   in the general case, this keyword allows to specify a permutation
               re-ordering the classes.
* `warn=true`: whether to show a warning in case `y` does not have scientific
               type `OrderedFactor{2}` (see note below).

## Note

To decrease the risk of unexpected errors, if `y` does not have
scientific type `OrderedFactor{2}` (and so does not have a "natural
ordering" negative-positive), a warning is shown indicating the
current order unless the user explicitly specifies either `rev` or
`perm` in which case it's assumed the user is aware of the class
ordering.

The `confusion_matrix` is a measure (although neither a score nor a
loss) and so may be specified as such in calls to `evaluate`,
`evaluate!`, although not in `TunedModel`s.  In this case, however,
there no way to specify an ordering different from `levels(y)`, where
`y` is the target.

"""
function _confmat(ŷraw::Union{Arr{V1,N}, CategoricalArray{V1,N}},
                  yraw::Union{Arr{V2,N}, CategoricalArray{V2,N}};
                  rev::Union{Nothing,Bool}=nothing,
                  perm::Union{Nothing,Vector{<:Integer}}=nothing,
                  warn::Bool=true) where
    {V,V1<:Union{Missing,V}, V2<:Union{Missing,V},N}

    # no-op if vectors already categorical arrays:
    ŷ, y = _categorical(ŷraw, yraw)

    levels_ = levels(y)
    nc = length(levels_)
    if rev !== nothing && rev && nc > 2
        throw(ArgumentError("Keyword `rev` can only be used in binary case."))
    end
    if perm !== nothing && !isempty(perm)
        length(perm) == nc ||
            throw(ArgumentError("`perm` must be of length matching the "*
                                "number of classes."))
        Set(perm) == Set(collect(1:nc)) ||
            throw(ArgumentError("`perm` must specify a valid permutation of "*
                                "`[1, 2, ..., c]`, where `c` is "*
                                "number of classes."))
    end

    # warning
    if rev === nothing && perm === nothing
        S = nonmissingtype(elscitype(y))
        if warn &&
            if nc==2 &&  !(S <: OrderedFactor)
                @warn "The classes are un-ordered,\n" *
                    "using: negative='$(levels_[1])' "*
                    "and positive='$(levels_[2])'.\n" *
                    "To suppress this warning, consider coercing "*
                    "to OrderedFactor."
            elseif !(S <: OrderedFactor)
                @warn "The classes are un-ordered,\n" *
                      "using order: $([l for l in levels_]).\n" *
                      "To suppress this warning, consider "*
                      "coercing to OrderedFactor."
            end
        end
        rev  = false
        perm = Int[]
    elseif rev !== nothing && nc == 2
        # rev takes precedence in binary case
        if rev
            perm = [2, 1]
        else
            perm = Int[]
        end
    end

    # No permutation
    if isempty(perm)
        cmat = zeros(Int, nc, nc)
        @inbounds for i in eachindex(y)
            (isinvalid(y[i]) || isinvalid(ŷ[i])) && continue
            cmat[int(ŷ[i]), int(y[i])] += 1
        end
        return ConfusionMatrixObject(cmat, string.(levels_))
    end

    # With permutation
    cmat = zeros(Int, nc, nc)
    iperm = invperm(perm)
    @inbounds for i in eachindex(y)
        (isinvalid(y[i]) || isinvalid(ŷ[i])) && continue
        cmat[iperm[int(ŷ[i])], iperm[int(y[i])]] += 1
    end
    return ConfusionMatrixObject(cmat, string.(levels_[perm]))
end


# Machinery to display the confusion matrix in a non-confusing way
# (provided the REPL is wide enough)

splitw(w::Int) = (sp1 = div(w, 2); sp2 = w - sp1; (sp1, sp2))

function Base.show(stream::IO, m::MIME"text/plain", cm::ConfusionMatrixObject{C}
                   ) where C
    width    = displaysize(stream)[2]
    cw       = 13
    textlim  = 9
    totalwidth = cw * (C+1) + C + 2
    width < totalwidth && (show(stream, m, cm.mat); return)

    iob     = IOBuffer()
    wline   = s -> write(iob, s * "\n")
    splitcw = s -> (w = cw - length(s); splitw(w))
    cropw   = s -> length(s) > textlim ? s[1:prevind(s, textlim)] * "…" : s

    # 1.a top box
    " "^(cw+1) * "┌" * "─"^((cw + 1) * C - 1) * "┐" |> wline
    gt = "Ground Truth"
    w  = (cw + 1) * C - 1 - length(gt)
    sp1, sp2 = splitw(w)
    " "^(cw+1) * "│" * " "^sp1 * gt * " "^sp2 * "│" |> wline
    # 1.b separator
    "┌" * "─"^cw * "┼" * ("─"^cw * "┬")^(C-1) * "─"^cw * "┤" |> wline
    # 2.a description line
    pr = "Predicted"
    sp1, sp2 = splitcw(pr)
    partial = "│" * " "^sp1 * pr * " "^sp2 * "│"
    for c in 1:C
        # max = 10
        s = cm.labels[c] |> cropw
        sp1, sp2 = splitcw(s)
        partial *= " "^sp1 * s * " "^sp2 * "│"
    end
    partial |> wline
    # 2.b separating line
    "├" * "─"^cw * "┼" * ("─"^cw * "┼")^(C-1) * ("─"^cw * "┤") |> wline
    # 2.c line by line
    for c in 1:C
        # line
        s  = cm.labels[c] |> cropw
        sp1, sp2 = splitcw(s)
        partial = "│" * " "^sp1 * s * " "^sp2 * "│"
        for r in 1:C
            e = string(cm[c, r])
            sp1, sp2 = splitcw(e)
            partial *= " "^sp1 * e * " "^sp2 * "│"
        end
        partial |> wline
        # separator
        if c < C
            "├" * "─"^cw * "┼" * ("─"^cw * "┼")^(C-1) * ("─"^cw * "┤") |> wline
        end
    end
    # 2.d final line
    "└" * "─"^cw * "┴" * ("─"^cw * "┴")^(C-1) * ("─"^cw * "┘") |> wline
    write(stream, take!(iob))
end


## CONFUSION MATRIX AS MEASURE

struct ConfusionMatrix <: Aggregated
    perm::Union{Nothing,Vector{<:Integer}}
end

ConfusionMatrix(; perm=nothing) = ConfusionMatrix(perm)

is_measure(::ConfusionMatrix) = true
is_measure_type(::Type{ConfusionMatrix}) = true
human_name(::Type{<:ConfusionMatrix}) = "confusion matrix"
target_scitype(::Type{ConfusionMatrix}) =
    Union{AbstractVector{<:Union{Missing,OrderedFactor}},
          AbstractVector{<:Union{Missing,OrderedFactor}}}
supports_weights(::Type{ConfusionMatrix}) = false
prediction_type(::Type{ConfusionMatrix}) = :deterministic
instances(::Type{<:ConfusionMatrix}) = ["confusion_matrix", "confmat"]
orientation(::Type{ConfusionMatrix}) = :other
reports_each_observation(::Type{ConfusionMatrix}) = false
is_feature_dependent(::Type{ConfusionMatrix}) = false
aggregation(::Type{ConfusionMatrix}) = Sum()

@create_aliases ConfusionMatrix

@create_docs(ConfusionMatrix,
body=
"""
If `r` is the return value, then the raw confusion matrix is `r.mat`,
whose rows correspond to predictions, and columns to ground truth.
The ordering follows that of `levels(y)`.

Use `ConfusionMatrix(perm=[2, 1])` to reverse the class order for binary
data. For more than two classes, specify an appropriate permutation, as in
`ConfusionMatrix(perm=[2, 3, 1])`.

""",
scitype=DOC_ORDERED_FACTOR_BINARY)

# calling behaviour:
call(m::ConfusionMatrix, ŷ, y) = _confmat(ŷ, y, perm=m.perm)

# overloading addition to make aggregation work:
Base.round(m::MLJBase.ConfusionMatrixObject; kws...) = m
function Base.:+(m1::ConfusionMatrixObject, m2::ConfusionMatrixObject)
    if m1.labels != m2.labels
        throw(ArgumentError("Confusion matrix labels must agree"))
    end
    ConfusionMatrixObject(m1.mat + m2.mat, m1.labels)
end
