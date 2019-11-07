## CLASSIFICATION METRICS (FOR DETERMINISTIC PREDICTIONS)

struct MisclassificationRate <: Measure end

"""
    misclassification_rate(ŷ, y)
    misclassification_rate(ŷ, y, w)

Returns the rate of misclassification of the (point) predictions `ŷ`,
given true observations `y`, optionally weighted by the weights
`w`. All three arguments must be abstract vectors of the same length.

For more information, run `info(misclassification_rate)`.

"""
misclassification_rate = MisclassificationRate()
name(::Type{<:MisclassificationRate}) = "misclassification_rate"
target_scitype(::Type{<:MisclassificationRate}) = AbstractVector{<:Finite}
prediction_type(::Type{<:MisclassificationRate}) = :deterministic
orientation(::Type{<:MisclassificationRate}) = :loss
reports_each_observation(::Type{<:MisclassificationRate}) = false
is_feature_dependent(::Type{<:MisclassificationRate}) = false
supports_weights(::Type{<:MisclassificationRate}) = true

(::MisclassificationRate)(ŷ::AbstractVector{<:CategoricalElement},
                          y::AbstractVector{<:CategoricalElement}) =
                              mean(y .!= ŷ)
(::MisclassificationRate)(ŷ::AbstractVector{<:CategoricalElement},
                          y::AbstractVector{<:CategoricalElement},
                          w::AbstractVector{<:Real}) =
                              sum((y .!= ŷ) .* w) / sum(w)


## CLASSIFICATION METRICS (FOR PROBABILISTIC PREDICTIONS)

struct CrossEntropy <: Measure end

"""
    cross_entropy(ŷ, y::AbstractVector{<:Finite})

Given an abstract vector of `UnivariateFinite` distributions `ŷ` (ie,
probabilistic predictions) and an abstract vector of true observations
`y`, return the negative log-probability that each observation would
occur, according to the corresponding probabilistic prediction.

For more information, run `info(cross_entropy)`.

"""
cross_entropy = CrossEntropy()
name(::Type{<:CrossEntropy}) = "cross_entropy"
target_scitype(::Type{<:CrossEntropy}) = AbstractVector{<:Finite}
prediction_type(::Type{<:CrossEntropy}) = :probabilistic
orientation(::Type{<:CrossEntropy}) = :loss
reports_each_observation(::Type{<:CrossEntropy}) = true
is_feature_dependent(::Type{<:CrossEntropy}) = false
supports_weights(::Type{<:CrossEntropy}) = false

# for single observation:
_cross_entropy(d, y) = -log(pdf(d, y))

function (::CrossEntropy)(ŷ::AbstractVector{<:UnivariateFinite},
                          y::AbstractVector{<:CategoricalElement})
    check_dimensions(ŷ, y)
    check_pools(ŷ, y)
    return broadcast(_cross_entropy, Any[ŷ...], y)
end

# TODO: support many distributions/samplers D below:
struct BrierScore{D} <: Measure end

"""
    brier = BrierScore(; distribution=UnivariateFinite)
    brier(ŷ, y)

Given an abstract vector of distributions `ŷ` and an abstract vector
of true observations `y`, return the corresponding Brier (aka
quadratic) scores.

Currently only `distribution=UnivariateFinite` is supported, which is
applicable to superivised models with `Finite` target scitype. In this
case, if `p(y)` is the predicted probability for a *single*
observation `y`, and `C` all possible classes, then the corresponding
Brier score for that observation is given by

``2p(y) - \\left(\\sum_{η ∈ C} p(η)^2\\right) - 1``

For more information, run `info(brier_score)`.

"""
function BrierScore(; distribution=UnivariateFinite)
    distribution == UnivariateFinite ||
        error("Only `UnivariateFinite` Brier scores currently supported. ")
    return BrierScore{distribution}()
end

name(::Type{<:BrierScore{D}}) where D = "BrierScore{$(string(D))}"
target_scitype(::Type{<:BrierScore{D}}) where D = AbstractVector{<:Finite}
prediction_type(::Type{<:BrierScore}) = :probabilistic
orientation(::Type{<:BrierScore}) = :score
reports_each_observation(::Type{<:BrierScore}) = true
is_feature_dependent(::Type{<:BrierScore}) = false
supports_weights(::Type{<:BrierScore}) = true

# For single observations (no checks):

# UnivariateFinite:
function brier_score(d::UnivariateFinite, y)
    levels = classes(d)
    pvec = broadcast(pdf, d, levels)
    offset = 1 + sum(pvec.^2)
    return 2*pdf(d, y) - offset
end

# For multiple observations:

# UnivariateFinite:
function (::BrierScore{<:UnivariateFinite})(
    ŷ::AbstractVector{<:UnivariateFinite},
    y::AbstractVector{<:CategoricalElement})
    check_dimensions(ŷ, y)
    check_pools(ŷ, y)
    return broadcast(brier_score, Any[ŷ...], y)
end

function (score::BrierScore{<:UnivariateFinite})(
    ŷ, y, w::AbstractVector{<:Real})
    check_dimensions(y, w)
    return w .* score(ŷ, y) ./ (sum(w)/length(y))
end


"""
ConfusionMatrix{C}

Confusion matrix with `C ≥ 2` classes.
"""
struct ConfusionMatrix{C}
    mat::Matrix
end

Base.show(s::IO, m::MIME"text/plain", cm::ConfusionMatrix) = show(s, m, cm.mat)

Base.getindex(cm::ConfusionMatrix, inds...) = getindex(cm.mat, inds...)

"""
ConfusionMatrix(m)

Instantiates a confusion matrix out of a square integer matrix `m`.
Rows are the predicted class, columns the 'true' class. See also
the [wikipedia article](https://en.wikipedia.org/wiki/Confusion_matrix).
"""
function ConfusionMatrix(m::Matrix{Int})
    s = size(m)
    s[1] == s[2] || throw(ArgumentError("Expected a square matrix."))
    s[1] > 1 || throw(ArgumentError("Expected a matrix of size ≥ 2x2."))
    ConfusionMatrix{s[1]}(m)
end

"""
confusion_matrix(ŷ, y; rev=false)

Computes the confusion matrix given a predicted `ŷ` with categorical elements and the actual `y`.
Rows are the predicted class, columns the 'true' class.
The ordering follows that of `levels(y)`.
In the binary case, the first class is by default the "negative" and the second the "positive";
this can be reversed using `rev=true`.
"""
function confusion_matrix(ŷ::VC, y::VC;
                          rev::Bool=false) where VC <: AbstractVector{<:CategoricalElement}
    check_dimensions(ŷ, y)
    nc   = length(levels(y))
    if rev && nc > 2
        throw(ArgumentError("Keyword `rev` can only be used in binary case."))
    end
    cmat = zeros(Int, nc, nc)
    @inbounds for i in eachindex(y)
        cmat[int(ŷ[i]), int(y[i])] += 1
    end
    if rev
        cm2 = zeros(Int, 2, 2)
        cm2[1, 1] = cmat[2, 2]
        cm2[1, 2] = cmat[2, 1]
        cm2[2, 1] = cmat[1, 2]
        cm2[2, 2] = cmat[1, 1]
        cmat = cm2
    end
    return ConfusionMatrix(cmat)
end

confmat = confusion_matrix

# ============================
struct Accuracy <: Measure end

const accuracy = Accuracy()

(::Accuracy)(args...) = 1.0 - misclassification_rate(args...)
(::Accuracy)(m::ConfusionMatrix) = sum(diag(m.mat)) / sum(m.mat)

metadata_measure(Accuracy;
    name="accuracy",
    target=AbstractVector{<:Finite},
    pred=:deterministic,
    orientation=:score,
    reports_each=false,
    feat_dep=false,
    weights=true)

# === Binary specific conf-mat related metrics
# NOTE: rev keyword allows to swap considered ordering positive <> negative

@with_kw_noshow struct Recall <: Measure;      rev::Bool=false; end
@with_kw_noshow struct Precision <: Measure;   rev::Bool=false; end
@with_kw_noshow struct Specificity <: Measure; rev::Bool=false; end
@with_kw_noshow struct FScore{β} <: Measure;   rev::Bool=false; end
@with_kw_noshow struct AUC <: Measure;         rev::Bool=false; end

const recall = Recall()
const sensitivity = recall

const specificity = Specificity()
const selectivity = specificity

# nothing for precision as there is Base.precision

const f1score = FScore{1}()

const auc = AUC()

metadata_measure.((Recall, Precision, Specificity, FScore);
    target=AbstractVector{<:Finite},
    pred=:deterministic,
    orientation=:score,
    reports_each=false,
    feat_dep=false,
    weights=false)

# adjustments
name(::Type{<:Recall}) = "recall"
name(::Type{<:Precision}) = "precision"
name(::Type{<:Specificity}) = "specificity"
name(::Type{<:FScore{β}}) where β = "F$β-score"

metadata_measure(AUC;
    target=AbstractVector{<:Finite},
    pred=:probabilistic,
    orientation=:score,
    reports_each=false,
    feat_dep=false,
    weights=false)

name(::Type{<:AUC}) = "auc"

###

function positive_label(y::VC; rev::Bool=false) where VC <: AbstractVector{<:CategoricalElement}
    lvls = levels(y)
    length(lvls) == 2 || throw(ArgumentError("Expected a 2-class vector."))
    rev && return lvls[1]
    return lvls[2]
end

negative_label(y; rev::Bool=false) = positive_label(y; rev=!rev)

truepositive(m::ConfusionMatrix{2},  r=false; rev::Bool=r) = ifelse(rev, m[1,1], m[2,2])
truenegative(m::ConfusionMatrix{2},  r=false; rev::Bool=r) = ifelse(rev, m[2,2], m[1,1])
falsepositive(m::ConfusionMatrix{2}, r=false; rev::Bool=r) = ifelse(rev, m[1,2], m[2,1])
falsenegative(m::ConfusionMatrix{2}, r=false; rev::Bool=r) = ifelse(rev, m[2,1], m[1,2])

function truepositive_rate(m::ConfusionMatrix{2}, r=false; rev=r)
    tp = truepositive(m, rev)
    fn = falsenegative(m, rev)
    return tp / (tp + fn)
end

function truenegative_rate(m::ConfusionMatrix{2}, r=false; rev=r)
    tn = truenegative(m, rev)
    fp = falsepositive(m, rev)
    return tn / (tn + fp)
end

function falsediscovery_rate(m::ConfusionMatrix{2}, r=false; rev=r)
    fp = falsepositive(m, rev)
    tp = truepositive(m, rev)
    return fp / (tp + fp)
end

(r::Recall)(m::ConfusionMatrix{2})      = truepositive_rate(m, r.rev)
(s::Specificity)(m::ConfusionMatrix{2}) = truenegative_rate(m, s.rev)
(p::Precision)(m::ConfusionMatrix{2})   = 1.0 - falsediscovery_rate(m, p.rev)

Base.precision(m::ConfusionMatrix{2}) = m |> Precision()

function (f::FScore{β})(m::ConfusionMatrix{2}) where β
    β2 = β^2
    if f.rev
        prec = m |> Precision(rev=true)
        rec  = m |> Recall(rev=true)
    else
        prec = precision(m)
        rec  = recall(m)
    end
    return (1 + β2) * (prec * rec) / (β2 * prec + rec)
end

truepositive(ŷ, y; rev=false)  = truepositive(confmat(ŷ, y),  rev)
truenegative(ŷ, y; rev=false)  = truenegative(confmat(ŷ, y),  rev)
falsepositive(ŷ, y; rev=false) = falsepositive(confmat(ŷ, y), rev)
falsenegative(ŷ, y; rev=false) = falsenegative(confmat(ŷ, y), rev)

truepositive_rate(ŷ, y; rev=false)   = truepositive_rate(confmat(ŷ, y),   rev)
truenegative_rate(ŷ, y; rev=false)   = truenegative_rate(confmat(ŷ, y),   rev)
falsediscovery_rate(ŷ, y; rev=false) = falsediscovery_rate(confmat(ŷ, y), rev)

(m::Union{Recall,Specificity,Precision,FScore})(ŷ, y) = confmat(ŷ, y) |> m

Base.precision(ŷ, y) = confmat(ŷ, y) |> Precision()

function (a::AUC)(ŷ, y)
    # implementation drawn from the ranked comparison in
    # https://blog.revolutionanalytics.com/2016/11/calculating-auc.html
    label_1 = levels(y)[1]
    scores  = pdf.(ŷ, label_1)
    ranking = sortperm(scores, rev=true)

    # sorted scores
    scores = scores[ranking]
    mask_1 = (y[ranking] .== label_1)

    scores_1 = scores[mask_1]
    scores_2 = scores[.!mask_1]

    n_1 = length(scores_1)
    n_2 = length(scores_2)

    M = 0.0
    for i in n_1:-1:1, j in 1:n_2
        M += (1 + sign(scores_1[i] - scores_2[j]))/2
    end
    auc = M / (n_1 * n_2)
    a.rev || return  1.0 - auc
    return auc
end
