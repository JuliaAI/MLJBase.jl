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
confusion_matrix(ŷ, y)

Computes the confusion matrix given a predicted `ŷ` with categorical elements and the actual `y`.
Rows are the predicted class, columns the 'true' class.
"""
function confusion_matrix(ŷ::VC, y::VC) where VC <: AbstractVector{<:CategoricalElement}
    check_dimensions(ŷ, y)
    nc   = length(levels(y))
    cmat = zeros(Int, nc, nc)
    @inbounds for i in eachindex(y)
        cmat[int(ŷ[i]), int(y[i])] += 1
    end
    return ConfusionMatrix(cmat)
end

confmat = confusion_matrix

# ============================
struct Accuracy <: Measure end
const TAccuracy = Type{<:Accuracy}

const accuracy = Accuracy()

name(::TAccuracy) = "accuracy"
target_scitype(::TAccuracy) = AbstractVector{<:Finite}
prediction_type(::TAccuracy) = :deterministic
orientation(::TAccuracy) = :score
reports_each_observation(::TAccuracy) = true
is_feature_dependent(::TAccuracy) = false
supports_weights(::TAccuracy) = true

(::Accuracy)(args...) = 1.0 - misclassification_rate(args...)
(::Accuracy)(m::ConfusionMatrix) = sum(diag(m.mat)) / sum(m.mat)

# ==========================
struct Recall <: Measure end
struct Precision <: Measure end
struct Specificity <: Measure end
struct FScore{β} <: Measure end
struct AUC <: Measure end

const TRecall = Type{<:Recall}
const TPrecision = Type{<:Precision}
const TSpecificity = Type{<:Specificity}
const TFScore = Type{<:FScore}
const TAUC = Type{<:AUC}
const TScoreCM = Union{TRecall,TPrecision,TSpecificity,TFScore}

const recall = Recall()
const sensitivity = recall

const specificity = Specificity()
const selectivity = specificity

# nothing for precision as there is a Base.precision

const f1score = FScore{1}()

const auc = AUC()

name(::TRecall) = "recall"
name(::TPrecision) = "precision"
name(::TSpecificity) = "specificity"
name(::Type{<:FScore{β}}) where β = "F$β-score"
name(::Type{<:AUC}) where β = "AUC"

target_scitype(::TScoreCM) = AbstractVector{<:Finite}
prediction_type(::TScoreCM) = :deterministic
orientation(::TScoreCM) = :score
reports_each_observation(::TScoreCM) = true
is_feature_dependent(::TScoreCM) = false
supports_weights(::TScoreCM) = false

target_scitype(::TAUC) = AbstractVector{<:Finite}
prediction_type(::TAUC) = :probabilistic
orientation(::TAUC) = :score
reports_each_observation(::TAUC) = true
is_feature_dependent(::TAUC) = false
supports_weights(::TAUC) = false

truepositive(m::ConfusionMatrix{2}, fcp=true; first_class_positive::Bool=fcp) =
    ifelse(first_class_positive, m.mat[1,1], m.mat[2,2])
truenegative(m::ConfusionMatrix{2}, fcp=true; first_class_positive::Bool=fcp) =
    ifelse(first_class_positive, m.mat[2,2], m.mat[1,1])
falsepositive(m::ConfusionMatrix{2}, fcp=true; first_class_positive::Bool=fcp) =
    ifelse(first_class_positive, m.mat[1,2], m.mat[2,1])
falsenegative(m::ConfusionMatrix{2}, fcp=true; first_class_positive::Bool=fcp) =
    ifelse(first_class_positive, m.mat[2,1], m.mat[1,2])

function truepositive_rate(m::ConfusionMatrix{2}, fcp=true; first_class_positive=fcp)
    tp = truepositive(m, first_class_positive)
    fn = falsenegative(m, first_class_positive)
    return tp / (tp + fn)
end

function truenegative_rate(m::ConfusionMatrix{2}, fcp=true; first_class_positive=fcp)
    tn = truenegative(m, first_class_positive)
    fp = falsepositive(m, first_class_positive)
    return tn / (tn + fp)
end

function falsediscovery_rate(m::ConfusionMatrix{2}, fcp=true; first_class_positive=fcp)
    fp = falsepositive(m, first_class_positive)
    tp = truepositive(m, first_class_positive)
    return fp / (tp + fp)
end

(::Recall)(m::ConfusionMatrix{2}, fcp=true; first_class_positive=fcp) =
    truepositive_rate(m, first_class_positive)

(::Specificity)(m::ConfusionMatrix{2}, fcp=true; first_class_positive=fcp) =
    truenegative_rate(m, first_class_positive)

Base.precision(m::ConfusionMatrix{2}, fcp=true; first_class_positive=fcp) =
    1.0 - falsediscovery_rate(m, first_class_positive)

function (::FScore{β})(m::ConfusionMatrix{2}, fcp=true; first_class_positive=fcp) where β
    β2 = β^2
    prec = precision(m, first_class_positive)
    rec  = recall(m, first_class_positive)
    return (1 + β2) * (prec * rec) / (β2 * prec + rec)
end

truepositive(ŷ, y; args...)  = truepositive(confmat(ŷ, y); args...)
truenegative(ŷ, y; args...)  = truenegative(confmat(ŷ, y); args...)
falsepositive(ŷ, y; args...) = falsepositive(confmat(ŷ, y); args...)
falsenegative(ŷ, y; args...) = falsenegative(confmat(ŷ, y); args...)

truepositive_rate(ŷ, y; args...)   = truepositive_rate(confmat(ŷ, y); args...)
truenegative_rate(ŷ, y; args...)   = truenegative_rate(confmat(ŷ, y); args...)
falsediscovery_rate(ŷ, y; args...) = falsediscovery_rate(confmat(ŷ, y); args...)

(r::Recall)(ŷ, y; args...) = r(confmat(ŷ, y); args...)
(s::Specificity)(ŷ, y; args...) = s(confmat(ŷ, y); args...)
Base.precision(ŷ, y; args...) = precision(confmat(ŷ, y); args...)
(f::FScore)(ŷ, y; args...) = f(confmat(ŷ, y); args...)

# implementation drawn from the ranked comparison in
# https://blog.revolutionanalytics.com/2016/11/calculating-auc.html
function (::AUC)(ŷ, y; first_class_positive=true)
    pos_label = levels(y)[1]
    scores    = pdf.(ŷ, pos_label)
    ranking   = sortperm(scores, rev=true)
    # sorted scores
    scores   = scores[ranking]
    pos_mask = (y[ranking] .== pos_label)

    pos_scores = scores[pos_mask]
    neg_scores = scores[.!pos_mask]

    n_pos = length(pos_scores)
    n_neg = length(neg_scores)

    M = 0.0
    for i in n_pos:-1:1, j in 1:n_neg
        M += (1 + sign(pos_scores[i] - neg_scores[j]))/2
    end

    auc = M / (n_pos * n_neg)

    first_class_positive && return auc
    return 1.0 - auc
end
