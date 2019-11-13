# ---------------------------------------------------
## CLASSIFICATION METRICS (PROBABILISTIC PREDICTIONS)
# >> CrossEntropy
# >> BriersScore
# ---------------------------------------------------

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


# ---------------------------------------------------
## CLASSIFICATION METRICS (DETERMINISTIC PREDICTIONS)
# ==> MisclassificationRate / MCR
# ==> Accuracy
# Binary only
# = order independent
# ==> AUC
# = order dependent
# ==> Recall
# ==> Precision
# ---------------------------------------------------

struct MisclassificationRate <: Measure end

"""
misclassification_rate(ŷ, y)
misclassification_rate(ŷ, y, w)
misclassification_rate(conf_mat)

Returns the rate of misclassification of the (point) predictions `ŷ`,
given true observations `y`, optionally weighted by the weights
`w`. All three arguments must be abstract vectors of the same length.
A confusion matrix can also be passed as argument.

For more information, run `info(misclassification_rate)`.
You can also equivalently use `mcr`.
"""
const misclassification_rate = MisclassificationRate()
const mcr = misclassification_rate

name(::Type{<:MisclassificationRate}) = "misclassification_rate"
target_scitype(::Type{<:MisclassificationRate}) = AbstractVector{<:Finite}
prediction_type(::Type{<:MisclassificationRate}) = :deterministic
orientation(::Type{<:MisclassificationRate}) = :loss
reports_each_observation(::Type{<:MisclassificationRate}) = false
is_feature_dependent(::Type{<:MisclassificationRate}) = false
supports_weights(::Type{<:MisclassificationRate}) = true

const MCR = MisclassificationRate

(::MCR)(ŷ::AbstractVector{<:CategoricalElement},
        y::AbstractVector{<:CategoricalElement}) = mean(y .!= ŷ)

(::MCR)(ŷ::AbstractVector{<:CategoricalElement},
        y::AbstractVector{<:CategoricalElement},
        w::AbstractVector{<:Real}) = sum((y .!= ŷ) .* w) / sum(w)

(::MCR)(cm::ConfusionMatrix) = 1.0 - sum(diag(cm.mat)) / sum(cm.mat)


struct Accuracy <: Measure end

"""
accuracy(ŷ, y)
accuracy(ŷ, y, w)
accuracy(conf_mat)

Returns the accuracy of the (point) predictions `ŷ`,
given true observations `y`, optionally weighted by the weights
`w`. All three arguments must be abstract vectors of the same length.

For more information, run `info(accuracy)`.
"""
const accuracy = Accuracy()

(::Accuracy)(args...) = 1.0 - misclassification_rate(args...)
(::Accuracy)(m::ConfusionMatrix) = sum(diag(m.mat)) / sum(m.mat)

metadata_measure(Accuracy;
    name="accuracy",
    target_scitype=AbstractVector{<:Finite},
    prediction_type=:deterministic,
    orientation=:score,
    reports_each_observation=false,
    is_feature_dependent=false,
    supports_weights=true)


struct BalancedAccuracy <: Measure end

const BACC = BalancedAccuracy

const balanced_accuracy = BACC()
const bacc = balanced_accuracy
const bac = bacc

function (::BACC)(ŷ::AbstractVector{<:CategoricalElement},
                  y::AbstractVector{<:CategoricalElement})
    class_count = Dist.countmap(y)
    ŵ = 1.0 ./ [class_count[yi] for yi in y]
    return sum( (ŷ .== y) .* ŵ ) / sum(ŵ)
end

function (::BACC)(ŷ::AbstractVector{<:CategoricalElement},
                  y::AbstractVector{<:CategoricalElement},
                  w::AbstractVector{<:Real})
    levels_ = levels(y)
    ŵ = similar(w)
    @inbounds for i in eachindex(w)
        ŵ[i] = w[i] / sum(w .* (y .== y[i]))
    end
    return sum( (ŷ .== y) .* ŵ ) / sum(ŵ)
end

metadata_measure(BACC;
    name="balanced accuracy",
    target_scitype=AbstractVector{<:Finite},
    prediction_type=:deterministic,
    orientation=:score,
    reports_each_observation=false,
    is_feature_dependent=false,
    supports_weights=true)


struct MatthewsCorrelation <: Measure end

const MCC = MatthewsCorrelation

const matthews_correlation = MCC()
const mcc = matthews_correlation

# http://rk.kvl.dk/introduction/index.html
function (::MCC)(cm::ConfusionMatrix{C}) where C
    # NOTE: this is O(C^3), there may be a clever way to adjust this
    num = 0
    @inbounds for k in 1:C, l in 1:C, m in 1:C
        num += cm[k,k] * cm[l,m] - cm[k,l] * cm[m,k]
    end
    den1 = 0
    den2 = 0
    @inbounds for k in 1:C
        a = sum(cm[k, :])
        b = sum(cm[setdiff(1:C, k), :])
        den1 += a * b
        a = sum(cm[:, k])
        b = sum(cm[:, setdiff(1:C, k)])
        den2 += a * b
    end
    mcc = num / sqrt(den1 * den2)

    isnan(mcc) && return 0
    return mcc
end

(m::MCC)(ŷ::AbstractVector{<:CategoricalElement},
         y::AbstractVector{<:CategoricalElement}) = confmat(ŷ, y, warn=false) |> m

## Binary but order independent

struct AUC <: Measure end
const auc = AUC()

function (::AUC)(ŷ::AbstractVector{<:UnivariateFinite},
                 y::AbstractVector{<:CategoricalElement})
    # implementation drawn from the ranked comparison in
    # ht_tps://blog.revolutionanalytics.com/2016/11/calculating-auc.html
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
    auc = 1 - M / (n_1 * n_2)
    return auc
end

metadata_measure(AUC;
    name="auc",
    target_scitype=AbstractVector{<:Finite},
    prediction_type=:probabilistic,
    orientation=:score,
    reports_each_observation=false,
    is_feature_dependent=false,
    supports_weights=false)

## Binary and order dependent

const CM2 = ConfusionMatrix{2}

for M in (:TruePositive, :TrueNegative, :FalsePositive, :FalseNegative,
          :TruePositiveRate, :TrueNegativeRate, :FalsePositiveRate, :FalseNegativeRate, :FalseDiscoveryRate, :Precision, :NPV)
    ex = quote
        struct $M <: Measure rev::Union{Nothing,Bool} end
        $M(; rev=nothing) = $M(rev)
    end
    eval(ex)
end

# synonyms
const TPR = TruePositiveRate
const TNR = TrueNegativeRate
const FPR = FalsePositiveRate
const FNR = FalseNegativeRate

const FDR = FalseDiscoveryRate
const PPV = Precision

const Recall      = TPR
const Specificity = TNR

struct FScore{β} <: Measure rev::Union{Nothing,Bool} end

FScore{β}(; rev=nothing) where β = FScore{β}(rev)

## Names and synonyms
# NOTE: nothing for precision as there is Base.precision

const truepositive  = TruePositive()
const truenegative  = TrueNegative()
const falsepositive = FalsePositive()
const falsenegative = FalseNegative()

const tp = truepositive
const tn = truenegative
const fp = falsepositive
const fn = falsenegative

const truepositive_rate  = TPR()
const truenegative_rate  = TNR()
const falsepositive_rate = FPR()
const falsenegative_rate = FNR()

const tpr = truepositive_rate
const tnr = truenegative_rate
const fpr = falsepositive_rate
const fnr = falsenegative_rate

const falsediscovery_rate = FDR()

const fdr = falsediscovery_rate
const npv = NPV()
const ppv = precision

const recall       = truepositive_rate
const sensitivity  = recall
const hit_rate     = recall
const miss_rate    = falsenegative_rate
const fallout      = falsepositive_rate
const specificity  = truenegative_rate
const selectivity  = specificity
const f1score      = FScore{1}()
const f1           = f1score

const balanced_accuracy = BACC()
const bacc = balanced_accuracy

metadata_measure.((FalsePositive, FalseNegative, FPR, FNR, FDR);
    target_scitype=AbstractVector{<:Finite},
    prediction_type=:deterministic,
    orientation=:loss,
    reports_each_observation=false,
    is_feature_dependent=false,
    supports_weights=false)

metadata_measure.((TruePositive, TrueNegative, TPR, TNR, Precision, FScore, NPV);
    target_scitype=AbstractVector{<:Finite},
    prediction_type=:deterministic,
    orientation=:score,
    reports_each_observation=false,
    is_feature_dependent=false,
    supports_weights=false)

# adjustments
name(::Type{<:TruePositive})  = "true positive"
name(::Type{<:TrueNegative})  = "true negative"
name(::Type{<:FalsePositive}) = "false positive"
name(::Type{<:FalseNegative}) = "false negative"

name(::Type{<:TPR}) = "true positive rate (sensitivity, recall, hit rate)"
name(::Type{<:TNR}) = "true negative rate (specificity, selectivity)"
name(::Type{<:FPR}) = "false positive rate (fallout)"
name(::Type{<:FNR}) = "false negative rate (miss rate)"

name(::Type{<:FDR}) = "false discovery rate"
name(::Type{<:NPV}) = "negative predictive value"

name(::Type{<:Precision}) = "precision (positive predictive value)"

name(::Type{<:FScore{β}}) where β = "F$β-score"

aggregation(::Type{<:TruePositive}) = Sum()
aggregation(::Type{<:TrueNegative})  = Sum()
aggregation(::Type{<:FalsePositive}) = Sum()
aggregation(::Type{<:FalseNegative}) = Sum()

## Internal functions on Confusion Matrix

_tp(m::CM2) = m[2,2]
_tn(m::CM2) = m[1,1]
_fp(m::CM2) = m[2,1]
_fn(m::CM2) = m[1,2]

_tpr(m::CM2) = tp(m) / (tp(m) + fn(m))
_tnr(m::CM2) = tn(m) / (tn(m) + fp(m))
_fpr(m::CM2) = 1 - _tnr(m)
_fnr(m::CM2) = 1 - _tpr(m)

_fdr(m::CM2) = fp(m) / (tp(m) + fp(m))
_npv(m::CM2) = tn(m) / (tn(m) + fn(m))

## Callables on CM2
# NOTE: here we assume the CM was constructed a priori with the
# proper ordering so the field `rev` in the measure is ignored

(::TruePositive)(m::CM2)  = _tp(m)
(::TrueNegative)(m::CM2)  = _tn(m)
(::FalsePositive)(m::CM2) = _fp(m)
(::FalseNegative)(m::CM2) = _fn(m)

(::TPR)(m::CM2) = _tpr(m)
(::TNR)(m::CM2) = _tnr(m)
(::FPR)(m::CM2) = _fpr(m)
(::FNR)(m::CM2) = _fnr(m)

(::FDR)(m::CM2) = _fdr(m)
(::NPV)(m::CM2) = _npv(m)

(::Precision)(m::CM2) = 1.0 - _fdr(m)

function (::FScore{β})(m::CM2) where β
    β2   = β^2
    prec = precision(m)
    rec  = recall(m)
    return (1 + β2) * (prec * rec) / (β2 * prec + rec)
end

## Callables on vectors

for M in (TruePositive, TrueNegative, FalsePositive, FalseNegative,
          TPR, TNR, FPR, FNR,
          FDR, Precision, NPV, FScore)
    (m::M)(ŷ, y) = confmat(ŷ, y; rev=m.rev) |> m
end

# specify this as `precision` is in Base and so is ambiguous
Base.precision(m::CM2) = m |> Precision()
Base.precision(ŷ, y)   = confmat(ŷ, y) |> Precision()
