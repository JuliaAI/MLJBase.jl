# ---------------------------------------------------
## CLASSIFICATION METRICS (PROBABILISTIC PREDICTIONS)
# >> CrossEntropy
# >> BriersScore

# -----------------------------------------------------
# cross entropy

struct CrossEntropy{R} <: Measure where R <: AbstractFloat
    eps::R
end
CrossEntropy(;eps=eps()) = CrossEntropy(eps)

metadata_measure(CrossEntropy;
    name                     = "cross_entropy",
    target_scitype           = Vec{<:Finite},
    prediction_type          = :probabilistic,
    orientation              = :loss,
    reports_each_observation = true,
    is_feature_dependent     = false,
    supports_weights         = false,
    docstring                = "Cross entropy loss with probabilities " *
                 "clamped between `eps()` and `1-eps()`; "*
                 "aliases: `cross_entropy`.",
    distribution_type        = UnivariateFinite)

"""
    cross_entropy

$(docstring(CrossEntropy()))

    ce = CrossEntropy(; eps=eps())
    ce(ŷ, y)

Given an abstract vector of distributions `ŷ` and an abstract vector
of true observations `y`, return the corresponding cross-entropy
loss (aka log loss) scores.

Since the score is undefined in the case of the true observation has
predicted probability zero, probablities are clipped between `eps` and
`1-eps` where `eps` can be specified.

If `sᵢ` is the predicted probability for the true class `yᵢ` then
the score for that example is given by

    -log(clamp(sᵢ, eps, 1-eps))

For more information, run `info(cross_entropy)`.
"""
cross_entropy = CrossEntropy()

# for single observation:
_cross_entropy(d::UnivariateFinite{S,V,R,P}, y, eps) where {S,V,R,P} =
    -log(clamp(pdf(d, y), P(eps), P(1) - P(eps)))

# multiple observations:
function (c::CrossEntropy)(ŷ::Vec{<:UnivariateFinite},
                           y::Vec)
    check_dimensions(ŷ, y)
    check_pools(ŷ, y)
    return broadcast(_cross_entropy, ŷ, y, c.eps)
end
# performant in case of UnivariateFiniteArray:
function (c::CrossEntropy)(ŷ::UnivariateFiniteVector{S,V,R,P},
                           y::Vec) where {S,V,R,P}
    check_dimensions(ŷ, y)
    check_pools(ŷ, y)
    return -log.(clamp.(broadcast(pdf, ŷ, y), P(c.eps), P(1) - P(c.eps)))
end

# -----------------------------------------------------
# brier score

# TODO: support many distributions/samplers D below:

struct BrierScore{D} <: Measure end

# As this measure is parametric, the use of `metadata_measure` is not
# appropriate.

MLJModelInterface.name(::Type{<:BrierScore{D}}) where D =
    "BrierScore{$(string(D))}"
MLJModelInterface.docstring(::Type{<:BrierScore{D}}) where D =
    "Brier proper scoring rule for distributions of type $D; "*
    "aliases: `BrierScore{$D}`"
MLJModelInterface.docstring(::Type{<:BrierScore{<:UnivariateFinite}}) =
    "Brier proper scoring rule for `MultiClass` or `OrderedFactor` data; "*
    "aliases: `BrierScore()`, `BrierScore{UnivariateFinite}`"
MLJModelInterface.docstring(::Type{BrierScore}) =
    "Brier proper scoring rule for various distribution types; " *
    "use `brier_score` for `BrierScore{UnivariateFinite}` "*
"(`Multiclass` or `OrderedFactor` targets)."
MLJModelInterface.target_scitype(::Type{<:BrierScore{D}}) where D =
    AbstractVector{<:Finite}
MLJModelInterface.prediction_type(::Type{<:BrierScore}) = :probabilistic
orientation(::Type{<:BrierScore}) = :score
reports_each_observation(::Type{<:BrierScore}) = true
is_feature_dependent(::Type{<:BrierScore}) = false
MLJModelInterface.supports_weights(::Type{<:BrierScore}) =  true
distribution_type(::Type{<:BrierScore{D}}) where D =
    UnivariateFinite

"""
    BrierScore(; distribution=UnivariateFinite)(ŷ, y [, w])

Given an abstract vector of distributions `ŷ` of type `distribution`,
and an abstract vector of true observations `y`, return the
corresponding Brier (aka quadratic) scores. Weight the scores using
`w` if provided.

Currently only `distribution=UnivariateFinite` is supported, which is
applicable to superivised models with `Finite` target scitype. In this
case, if `p(y)` is the predicted probability for a *single*
observation `y`, and `C` all possible classes, then the corresponding
Brier score for that observation is given by

``2p(y) - \\left(\\sum_{η ∈ C} p(η)^2\\right) - 1``

Note that `BrierScore()=BrierScore{UnivariateFinite}` has the alias
`brier_score`.

*Warning.* Here `BrierScore` is a "score" in the sense that bigger is
better (with `0` optimal, and all other values negative). In Brier's
original 1950 paper, and many other places, it has the opposite sign,
despite the name. Moreover, the present implementation does not treat
the binary case as special, so that the score may differ, in that
case, by a factor of two from usage elsewhere.

For more information, run `info(BrierScore)`.

"""
function BrierScore(; distribution=UnivariateFinite)
    distribution == UnivariateFinite ||
        error("Only `UnivariateFinite` Brier scores currently supported. ")
    return BrierScore{distribution}()
end

# For single observations (no checks):

# UnivariateFinite:
function _brier_score(d::UnivariateFinite{S,V,R,P}, y) where {S,V,R,P}
    levels = classes(d)
    pvec = broadcast(pdf, d, levels)
    offset = P(1) + sum(pvec.^2)
    return P(2) * pdf(d, y) - offset
end

# For multiple observations:

# UnivariateFinite:
function (::BrierScore{<:UnivariateFinite})(
    ŷ::Vec{<:UnivariateFinite},
    y::Vec,
    w::Union{Nothing,Vec{<:Real}}=nothing)

    check_dimensions(ŷ, y)
    w == nothing || check_dimensions(w, y)

    check_pools(ŷ, y)
    unweighted = broadcast(_brier_score, ŷ, y)

    if w == nothing
        return unweighted
    end
    return w.*unweighted
end
# performant version in case of UnivariateFiniteArray:
function (::BrierScore{<:UnivariateFinite})(
    ŷ::UnivariateFiniteVector{S,V,R,P},
    y::Vec,
    w::Union{Nothing,Vec{<:Real}}=nothing) where {S,V,R,P<:Real}

    check_dimensions(ŷ, y)
    w == nothing || check_dimensions(w, y)

    isempty(y) && return P(0)

    check_pools(ŷ, y)

    probs = pdf(ŷ, classes(first(ŷ)))
    offset = P(1) .+ vec(sum(probs.^2, dims=2))

    unweighted = P(2) .* broadcast(pdf, ŷ, y) .- offset

    if w == nothing
        return unweighted
    end
    return w.*unweighted
end

const brier_score = BrierScore()


# ============================================================
## CLASSIFICATION METRICS (DETERMINISTIC PREDICTIONS)

const INVARIANT_LABEL_MULTICLASS = "This metric is invariant to class labelling and can be used for multiclass classification."
const INVARIANT_LABEL_BINARY = "This metric is invariant to class labelling and can be used only for binary classification."
const VARIANT_LABEL_BINARY = "This metric is labelling-dependent and can only be used for binary classification."

# ==============================================================
## MULTICLASS

# ---------------------------------------------------
# misclassification rate

struct MisclassificationRate <: Measure end

metadata_measure(MisclassificationRate;
    name                     = "misclassification_rate",
    target_scitype           = Vec{<:Finite},
    prediction_type          = :deterministic,
    orientation              = :loss,
    reports_each_observation = false,
    is_feature_dependent     = false,
    supports_weights         = true,
    docstring                = "misclassification rate; aliases: " *
                               "`misclassification_rate`, `mcr`.")

"""
    misclassification_rate

$(docstring(MisclassificationRate()))

    misclassification_rate(ŷ, y)
    misclassification_rate(ŷ, y, w)
    misclassification_rate(conf_mat)

Returns the rate of misclassification of the (point) predictions `ŷ`,
given true observations `y`, optionally weighted by the weights
`w`. All three arguments must be abstract vectors of the same length.
A confusion matrix can also be passed as argument.
$INVARIANT_LABEL_MULTICLASS

For more information, run `info(misclassification_rate)`.

"""
const misclassification_rate = MisclassificationRate()
const mcr = misclassification_rate
const MCR = MisclassificationRate

(::MCR)(ŷ::Vec{<:CategoricalValue},
        y::Vec{<:CategoricalValue}) = mean(y .!= ŷ)

(::MCR)(ŷ::Vec{<:CategoricalValue},
        y::Vec{<:CategoricalValue},
        w::Vec{<:Real}) = sum((y .!= ŷ) .* w) / length(y)

(::MCR)(cm::ConfusionMatrix) = 1.0 - sum(diag(cm.mat)) / sum(cm.mat)

# -------------------------------------------------------------
# accuracy

struct Accuracy <: Measure end

metadata_measure(Accuracy;
    name                     = "accuracy",
    target_scitype           = Vec{<:Finite},
    prediction_type          = :deterministic,
    orientation              = :score,
    reports_each_observation = false,
    is_feature_dependent     = false,
    supports_weights         = true,
    docstring                = "Classification accuracy; aliases: `accuracy`.")

"""
    accuracy

$(docstring(Accuracy()))

    accuracy(ŷ, y)
    accuracy(ŷ, y, w)
    accuracy(conf_mat)

Returns the accuracy of the (point) predictions `ŷ`,
given true observations `y`, optionally weighted by the weights
`w`. All three arguments must be abstract vectors of the same length.
$INVARIANT_LABEL_MULTICLASS

For more information, run `info(accuracy)`.

"""
const accuracy = Accuracy()

(::Accuracy)(args...) = 1.0 - misclassification_rate(args...)
(::Accuracy)(m::ConfusionMatrix) = sum(diag(m.mat)) / sum(m.mat)


# -----------------------------------------------------------
# balanced accuracy

struct BalancedAccuracy <: Measure end

metadata_measure(BalancedAccuracy;
    name                     = "balanced_accuracy",
    target_scitype           = Vec{<:Finite},
    prediction_type          = :deterministic,
    orientation              = :score,
    reports_each_observation = false,
    is_feature_dependent     = false,
    supports_weights         = true,
    docstring                = "Balanced classification accuracy; aliases: "*
                               "`balanced_accuracy`, `bacc`, `bac`.")

"""
    balanced_accuracy

$(docstring(BalancedAccuracy()))

    balanced_accuracy(ŷ, y [, w])
    balanced_accuracy(conf_mat)

Return the balanced accuracy of the point prediction `ŷ`, given true
observations `y`, optionally weighted by `w`. The balanced accuracy takes
into consideration class imbalance.
All  three arguments must have the same length.
$INVARIANT_LABEL_MULTICLASS

For more information, run `info(balanced_accuracy)`.

"""
const balanced_accuracy = BalancedAccuracy()
const bacc = balanced_accuracy
const bac  = bacc
const BACC = BalancedAccuracy

function (::BACC)(ŷ::Vec{<:CategoricalValue},
                  y::Vec{<:CategoricalValue})
    class_count = Dist.countmap(y)
    ŵ = 1.0 ./ [class_count[yi] for yi in y]
    return sum( (ŷ .== y) .* ŵ ) / sum(ŵ)
end

function (::BACC)(ŷ::Vec{<:CategoricalValue},
                  y::Vec{<:CategoricalValue},
                  w::Vec{<:Real})
    levels_ = levels(y)
    ŵ = similar(w)
    @inbounds for i in eachindex(w)
        ŵ[i] = w[i] / sum(w .* (y .== y[i]))
    end
    return sum( (ŷ .== y) .* ŵ ) / sum(ŵ)
end

# ==================================================================
## BINARY AND ORDER-INDEPENDENT

# ------------------------------------------------------------------
# Matthew's correlation

struct MatthewsCorrelation <: Measure end

metadata_measure(MatthewsCorrelation;
    name                     = "matthews_correlation",
    target_scitype           = Vec{<:Finite{2}},
    prediction_type          = :deterministic,
    orientation              = :score,
    reports_each_observation = false,
    is_feature_dependent     = false,
    supports_weights         = false,
    docstring                = "Matthew's correlation; aliases: " *
                               "`matthews_correlation`, `mcc`")

"""
    matthews_correlation

$(docstring(MatthewsCorrelation()))

    matthews_correlation(ŷ, y)
    matthews_correlation(conf_mat)

Return Matthews' correlation coefficient corresponding to the point
prediction `ŷ`, given true observations `y`.
$INVARIANT_LABEL_MULTICLASS

For more information, run `info(matthews_correlation)`.

"""
const matthews_correlation = MatthewsCorrelation()
const mcc = matthews_correlation
const MCC = MatthewsCorrelation

function (::MCC)(cm::ConfusionMatrix{C}) where C
    # http://rk.kvl.dk/introduction/index.html
    # NOTE: this is O(C^3), there may be a clever way to
    # speed this up though in general this is only used for low  C
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

(m::MCC)(ŷ::Vec{<:CategoricalValue},
         y::Vec{<:CategoricalValue}) =
             confmat(ŷ, y, warn=false) |> m

# ---------------------------------------------------------
# area under the ROC curve

struct AUC <: Measure end

metadata_measure(AUC;
    name                     = "area_under_curve",
    target_scitype           = Vec{<:Finite{2}},
    prediction_type          = :probabilistic,
    orientation              = :score,
    reports_each_observation = false,
    is_feature_dependent     = false,
    supports_weights         = false,
    docstring                = "Area under the ROC curve; "*
       "aliases: `area_under_curve`, `auc`")

"""
    area_under_curve

$(docstring(AUC()))

    area_under_curve(ŷ, y)

Return the area under the receiver operator characteristic (curve),
for probabilistic predictions `ŷ`, given ground truth `y`.
$INVARIANT_LABEL_BINARY

For more information, run `info(area_under_curve)`.

"""
const area_under_curve = AUC()
const auc = AUC()

# implementation drawn from
# https://www.ibm.com/developerworks/community/blogs/jfp/entry/Fast_Computation_of_AUC_ROC_score?lang=en
function _auc(::Type{P}, ŷ::Vec{<:UnivariateFinite},
              y::Vec) where P<:Real # type of probabilities
    lab_pos = classes(first(ŷ))[2] # 'positive' label
    scores  = pdf.(ŷ, lab_pos)     # associated scores
    y_sort  = y[sortperm(scores)]  # sort by scores
    n       = length(y)
    n_neg   = 0  # to keep of the number of negative preds
    auc     = P(0)
    @inbounds for i in 1:n
        # y[i] == lab_p --> it's a positive label in the ground truth
        # in that case increase the auc by the cumulative sum
        # otherwise increase the number of negatives by 1
        δ_auc, δ_neg = ifelse(y_sort[i] == lab_pos, (n_neg, 0), (0, 1))
        auc   += δ_auc
        n_neg += δ_neg
    end
    n_pos = n - n_neg
    return auc / (n_neg * n_pos)
end

# ŷ inhomogeneous type:
(::AUC)(ŷ::Vec{<:UnivariateFinite}, y::Vec) = _auc(Float64, ŷ, y)

# ŷ homogeneous type (eg UnivariateFiniteVector case):
(::AUC)(ŷ::Vec{<:UnivariateFinite{S,V,R,P}}, y::Vec) where {S,V,R,P} =
    _auc(P, ŷ, y)


# ==========================================================================
## BINARY AND ORDER DEPENDENT

const CM2 = ConfusionMatrix{2}

# --------------------------------------------------------------------------
# F_β-Score

"""
    FScore{β}(rev=nothing)

One-parameter generalization, ``F_β``, of the F-measure or balanced F-score.

[Wikipedia entry](https://en.wikipedia.org/wiki/F1_score)

    FScore{β}(ŷ, y)

Evaluate ``F_β`` score on observations ,`ŷ`, given ground truth values, `y`.

By default, the second element of `levels(y)` is designated as
`true`. To reverse roles, use `FScore{β}(rev=true)` instead of
`FScore{β}`.

For more information, run `info(FScore)`.

"""
struct FScore{β} <: Measure rev::Union{Nothing,Bool} end

FScore{β}(; rev=nothing) where β = FScore{β}(rev)

metadata_measure(FScore;
    target_scitype           = Vec{<:Finite{2}},
    prediction_type          = :deterministic,
    orientation              = :score,
    reports_each_observation = false,
    is_feature_dependent     = false,
    supports_weights         = false)

MLJModelInterface.name(::Type{<:FScore{β}}) where β = "FScore{$β}"
MLJModelInterface.name(::Type{FScore})            = "FScore" # for registry
MLJModelInterface.docstring(::Type{<:FScore})       = "F_β score; aliases: " *
                                        "`FScore{β}`, `f1score=FScore{1}`"
const f1score      = FScore{1}()


# -------------------------------------------------------------------------
# truepositive, true_negative, etc

for M in (:TruePositive, :TrueNegative, :FalsePositive, :FalseNegative,
          :TruePositiveRate, :TrueNegativeRate, :FalsePositiveRate,
          :FalseNegativeRate, :FalseDiscoveryRate, :Precision, :NPV)
    ex = quote
        struct $M <: Measure rev::Union{Nothing,Bool} end
        $M(; rev=nothing) = $M(rev)
    end
    eval(ex)
end

metadata_measure.((FalsePositive, FalseNegative);
    target_scitype           = Vec{<:Finite{2}},
    prediction_type          = :deterministic,
    orientation              = :loss,
    reports_each_observation = false,
    aggregation              = Sum(),
    is_feature_dependent     = false,
    supports_weights         = false)

metadata_measure.((FalsePositiveRate, FalseNegativeRate, FalseDiscoveryRate);
    target_scitype           = Vec{<:Finite{2}},
    prediction_type          = :deterministic,
    orientation              = :loss,
    reports_each_observation = false,
    is_feature_dependent     = false,
    supports_weights         = false)

metadata_measure.((TruePositive, TrueNegative);
    target_scitype           = Vec{<:Finite{2}},
    prediction_type          = :deterministic,
    orientation              = :score,
    reports_each_observation = false,
    aggregation              = Sum(),
    is_feature_dependent     = false,
    supports_weights         = false)

metadata_measure.((TruePositiveRate, TrueNegativeRate, Precision, NPV);
    target_scitype           = Vec{<:Finite{2}},
    prediction_type          = :deterministic,
    orientation              = :score,
    reports_each_observation = false,
    is_feature_dependent     = false,
    supports_weights         = false)

# adjustments
MMI.name(::Type{<:TruePositive})       = "true_positive"
MMI.docstring(::Type{<:TruePositive})  = "Number of true positives; " *
                                         "aliases: `true_positive`, `truepositive`."
MMI.name(::Type{<:TrueNegative})       = "true_negative"
MMI.docstring(::Type{<:TrueNegative})  = "Number of true negatives; " *
                                         "aliases: `true_negative`, `truenegative`."
MMI.name(::Type{<:FalsePositive})      = "false_positive"
MMI.docstring(::Type{<:FalsePositive}) = "Number of false positives; " *
                                         "aliases: `false_positive`, `falsepositive`."
MMI.name(::Type{<:FalseNegative})      = "false_negative"
MMI.docstring(::Type{<:FalseNegative}) = "Number of false negatives; " *
                                         "aliases: `false_negative`, `falsenegative`."

MMI.name(::Type{<:TruePositiveRate})      = "true_positive_rate"
MMI.docstring(::Type{<:TruePositiveRate}) = "True positive rate; aliases: " *
                               "`true_positive_rate`, `truepositive_rate`, `tpr`, `sensitivity`, " *
                               "`recall`, `hit_rate`."
MMI.name(::Type{<:TrueNegativeRate})      = "true_negative_rate"
MMI.docstring(::Type{<:TrueNegativeRate}) = "true negative rate; aliases: " *
                               "`true_negative_rate`, `truenegative_rate`, `tnr`, `specificity`, " *
                               "`selectivity`."
MMI.name(::Type{<:FalsePositiveRate})      = "false_positive_rate"
MMI.docstring(::Type{<:FalsePositiveRate}) = "false positive rate; aliases: " *
                               "`false_positive_rate`, `falsepositive_rate`, `fpr`, `fallout`."
MMI.name(::Type{<:FalseNegativeRate})      = "false_negative_rate"
MMI.docstring(::Type{<:FalseNegativeRate}) = "false negative rate; aliases: " *
                               "`false_negative_rate`, `falsenegative_rate`, `fnr`, `miss_rate`."
MMI.name(::Type{<:FalseDiscoveryRate})      = "false_discovery_rate"
MMI.docstring(::Type{<:FalseDiscoveryRate}) = "false discovery rate; "*
                               "aliases: `false_discovery_rate`, `falsediscovery_rate`, `fdr`."
MMI.name(::Type{<:NPV})      = "negative_predictive_value"
MMI.docstring(::Type{<:NPV}) = "negative predictive value; aliases: " *
                               "`negative_predictive_value`, `negativepredictive_value`, `npv`."

MMI.name(::Type{<:Precision})         = "positive_predictive_value"
MMI.docstring(::Type{<:Precision})    = "positive predictive value "*
  "(aka precision); aliases: `positive_predictive_value`, `ppv`, `Precision()`, `positivepredictive_value`. "

"""
    true_positive

$(docstring(TruePositive()))

    true_positive(ŷ, y)

Number of true positives for observations `ŷ` and ground truth
`y`. Assigns `false` to first element of `levels(y)`. To reverse roles,
use `TruePositive(rev=true)` instead of `true_positive`.

For more information, run `info(true_positive)`.

"""
const true_positive = TruePositive()
const tp = TruePositive()
const truepositive  = TruePositive()

"""
    true_negative

$(docstring(TrueNegative()))

    true_negative(ŷ, y)

Number of true negatives for observations `ŷ` and ground truth
`y`. Assigns `false` to first element of `levels(y)`. To reverse roles,
use `TrueNegative(rev=true)` instead of `true_negative`.


For more information, run `info(true_negative)`.

"""
const true_negative = TrueNegative()
const tn = TrueNegative()
const truenegative  = TrueNegative()

"""
    false_positive

$(docstring(FalsePositive()))

    false_positive(ŷ, y)

Number of false positives for observations `ŷ` and ground truth
`y`. Assigns `false` to first element of `levels(y)`. To reverse roles,
use `FalsePositive(rev=true)` instead of `false_positive`.


For more information, run `info(false_positive)`.

"""
const false_positive = FalsePositive()
const fp = FalsePositive()
const falsepositive = FalsePositive()

"""
    false_negative

$(docstring(FalseNegative()))

    false_negative(ŷ, y)

Number of false positives for observations `ŷ` and ground truth
`y`. Assigns `false` to first element of `levels(y)`. To reverse roles,
use `FalseNegative(rev=true)` instead of `false_negative`.

For more information, run `info(false_negative)`.

"""
const false_negative = FalseNegative()
const fn = FalseNegative()
const falsenegative = FalseNegative()

"""
    true_positive_rate

$(docstring(TruePositiveRate()))

    true_positive_rate(ŷ, y)

True positive rate for observations `ŷ` and ground truth `y`. Assigns
`false` to first element of `levels(y)`. To reverse roles, use
`TruePositiveRate(rev=true)` instead of `true_positive_rate`.

For more information, run `info(true_positive_rate)`.

"""
const true_positive_rate = TruePositiveRate()
const tpr = TruePositiveRate()
const TPR = TruePositiveRate
const truepositive_rate  = TPR()
const recall = TPR()
const Recall = TPR
const sensitivity  = recall
const hit_rate     = recall

"""
    true_negative_rate

$(docstring(TrueNegativeRate()))

    true_negative_rate(ŷ, y)

True negative rate for observations `ŷ` and ground truth `y`. Assigns
`false` to first element of `levels(y)`. To reverse roles, use
`TrueNegativeRate(rev=true)` instead of `true_negative_rate`.

For more information, run `info(true_negative_rate)`.

"""
const true_negative_rate = TrueNegativeRate()
const tnr = TrueNegativeRate()
const TNR = TrueNegativeRate
const truenegative_rate  = TNR()
const Specificity = TNR
const specificity  = truenegative_rate
const selectivity  = specificity

"""
    false_positive_rate

$(docstring(FalsePositiveRate()))

    false_positive_rate(ŷ, y)

False positive rate for observations `ŷ` and ground truth `y`. Assigns
`false` to first element of `levels(y)`. To reverse roles, use
`FalsePositiveRate(rev=true)` instead of `false_positive_rate`.

For more information, run `info(false_positive_rate)`.

"""
const false_positive_rate = FalsePositiveRate()
const fpr = FalsePositiveRate()
const FPR = FalsePositiveRate
const falsepositive_rate = FPR()
const fallout      = falsepositive_rate

"""
    false_negative_rate

$(docstring(FalseNegativeRate()))

    false_negative_rate(ŷ, y)

False negative rate for observations `ŷ` and ground truth `y`. Assigns
`false` to first element of `levels(y)`. To reverse roles, use
`FalseNegativeRate(rev=true)` instead of `false_negative_rate`.

For more information, run `info(false_negative_rate)`.

"""
const false_negative_rate = FalseNegativeRate()
const fnr = FalseNegativeRate()
const FNR = FalseNegativeRate
const falsenegative_rate = FNR()
const miss_rate    = falsenegative_rate

"""
    false_discovery_rate

$(docstring(FalseDiscoveryRate()))

    false_discovery_rate(ŷ, y)

False discovery rate for observations `ŷ` and ground truth `y`. Assigns
`false` to first element of `levels(y)`. To reverse roles, use
`FalseDiscoveryRate(rev=true)` instead of `false_discovery_rate`.

For more information, run `info(false_discovery_rate)`.

"""
const false_discovery_rate = FalseDiscoveryRate()
const fdr = FalseDiscoveryRate()
const FDR = FalseDiscoveryRate
const falsediscovery_rate = FDR()

"""
    negative_predictive_value

$(docstring(NPV()))

    negative_predictive_value(ŷ, y)

Negative predictive value for observations `ŷ` and ground truth
`y`. Assigns `false` to first element of `levels(y)`. To reverse roles,
use `NPV(rev=true)` instead of `negative_predictive_value`.

For more information, run `info(negative_predictive_value)`.

"""
const negative_predictive_value = NPV()
const npv = NPV()
const negativepredictive_value = NPV()

"""
    positive_predictive_value

$(docstring(Precision()))

    positive_predictive_value(ŷ, y)

Positive predictive value for observations `ŷ` and ground truth
`y`. Assigns `false` to first element of `levels(y)`. To reverse roles,
use `Precision(rev=true)` instead of `positive_predictive_value`.

For more information, run `info(positive_predictive_value)`.

"""
const positive_predictive_value = Precision()
const ppv = Precision()
const positivepredictive_value = Precision()
const PPV = Precision


## INTERNAL FUNCTIONS ON CONFUSION MATRIX

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

## MULTICLASS AND ORDER INDEPENDENT

const CM = ConfusionMatrix{N} where N

struct MulticlassFScore{β} <:Measure average::Symbol end

MulticlassFScore{β}(; average=:macro) where β = MulticlassFScore{β}(average)

metadata_measure(MulticlassFScore;
    target_scitype           = Vec{<:Finite{N}} where N,
    prediction_type          = :deterministic,
    orientation              = :score,
    reports_each_observation = false,
    is_feature_dependent     = false,
    supports_weights         = false,
    supports_class_weights   = true)

MLJModelInterface.name(::Type{<:MulticlassFScore{β}}) where β = "MulticlassFScore{$β}"
MLJModelInterface.name(::Type{MulticlassFScore})        = "MulticlassFScore" # for registry
MLJModelInterface.docstring(::Type{<:MulticlassFScore}) = "Multiclass F_β score; aliases: " *
            "`macro_f1score=MulticlassFScore{1}`, `multiclass_f1score=MulticlassFScore{1}`" *
            "`MulticlassFScore{β}`, `micro_f1score=MulticlassFScore{1}(; average=:micro)`."

const micro_f1score      = MulticlassFScore{1}(; average=:micro)
const macro_f1score      = MulticlassFScore{1}()
const multiclass_f1score = macro_f1score

for M in (:MulticlassTruePositive, :MulticlassTrueNegative,
          :MulticlassFalsePositive, :MulticlassFalseNegative,
          :MulticlassTruePositiveRate, :MulticlassTrueNegativeRate,
          :MulticlassFalsePositiveRate, :MulticlassFalseNegativeRate,
          :MulticlassFalseDiscoveryRate, :MulticlassPrecision, :MulticlassNPV)
    ex = quote
        struct $M{T<:Symbol} <: Measure average::T end
        $M(; average=:macro) = $M(average)
    end
    eval(ex)
end

metadata_measure.((MulticlassFalsePositive, MulticlassFalseNegative);
    target_scitype           = Vec{<:Finite{N}} where N,
    prediction_type          = :deterministic,
    orientation              = :loss,
    reports_each_observation = false,
    aggregation               = Sum(),
    is_feature_dependent     = false,
    supports_weights         = false,
    supports_class_weights   = false)

metadata_measure.((MulticlassFalsePositiveRate, MulticlassFalseNegativeRate,
                   MulticlassFalseDiscoveryRate);
    target_scitype           = Vec{<:Finite{N}} where N,
    prediction_type          = :deterministic,
    orientation              = :loss,
    reports_each_observation = false,
    is_feature_dependent     = false,
    supports_weights         = false,
    supports_class_weights   = true)


metadata_measure.((MulticlassTruePositive, MulticlassTrueNegative);
    target_scitype           = Vec{<:Finite{N}} where N,
    prediction_type          = :deterministic,
    orientation              = :score,
    reports_each_observation = false,
    aggregation              = Sum(),
    is_feature_dependent     = false,
    supports_weights         = false,
    supports_class_weights   = false)

metadata_measure.((MulticlassTrueNegativeRate, MulticlassNPV);
    target_scitype           = Vec{<:Finite{N}} where N,
    prediction_type          = :deterministic,
    orientation              = :score,
    reports_each_observation = false,
    is_feature_dependent     = false,
    supports_weights         = false,
    supports_class_weights   = true)

metadata_measure.((MulticlassTruePositiveRate, MulticlassPrecision);
    target_scitype           = Vec{<:Finite{N}} where N,
    prediction_type          = :deterministic,
    orientation              = :score,
    reports_each_observation = false,
    is_feature_dependent     = false,
    supports_weights         = false,
    supports_class_weights   = true)

MMI.name(::Type{<:MulticlassTruePositive})       = "multiclass_true_positive"
MMI.docstring(::Type{<:MulticlassTruePositive})  = "Number of true positives; " *
                "aliases: `multiclass_true_positive`, `multiclass_truepositive`."
MMI.name(::Type{<:MulticlassTrueNegative})       = "multiclass_true_negative"
MMI.docstring(::Type{<:MulticlassTrueNegative})  = "Number of true negatives; " *
                "aliases: `multiclass_true_negative`, `multiclass_truenegative`."
MMI.name(::Type{<:MulticlassFalsePositive})      = "multiclass_false_positive"
MMI.docstring(::Type{<:MulticlassFalsePositive}) = "Number of false positives; " *
                "aliases: `multiclass_false_positive`, `multiclass_falsepositive`."
MMI.name(::Type{<:MulticlassFalseNegative})      = "multiclass_false_negative"
MMI.docstring(::Type{<:MulticlassFalseNegative}) = "Number of false negatives; " *
                "aliases: `multiclass_false_negative`, `multiclass_falsenegative`."

MMI.name(::Type{<:MulticlassTruePositiveRate})      = "multiclass_true_positive_rate"
MMI.docstring(::Type{<:MulticlassTruePositiveRate}) = "True positive rate; aliases: " *
                       "`multiclass_true_positive_rate`, `multiclass_tpr`, " *
                       "`multiclass_sensitivity`, `multiclass_recall`, " *
                       "`multiclass_hit_rate`, `multiclass_truepositive_rate`."
MMI.name(::Type{<:MulticlassTrueNegativeRate})      = "multiclass_true_negative_rate"
MMI.docstring(::Type{<:MulticlassTrueNegativeRate}) = "true negative rate; aliases: " *
                       "`multiclass_true_negative_rate`, `multiclass_tnr` " *
                       " `multiclass_specificity`, `multiclass_selectivity`, " *
                       "`multiclass_truenegative_rate`."
MMI.name(::Type{<:MulticlassFalsePositiveRate})      = "multiclass_false_positive_rate"
MMI.docstring(::Type{<:MulticlassFalsePositiveRate}) = "false positive rate; aliases: " *
                       "`multiclass_false_positive_rate`, `multiclass_fpr` " *
                       "`multiclass_fallout`, `multiclass_falsepositive_rate`."
MMI.name(::Type{<:MulticlassFalseNegativeRate})      = "multiclass_false_negative_rate"
MMI.docstring(::Type{<:MulticlassFalseNegativeRate}) = "false negative rate; aliases: " *
                       "`multiclass_false_negative_rate`, `multiclass_fnr`, " *
                       "`multiclass_miss_rate`, `multiclass_falsenegative_rate`."
MMI.name(::Type{<:MulticlassFalseDiscoveryRate})      = "multiclass_false_discovery_rate"
MMI.docstring(::Type{<:MulticlassFalseDiscoveryRate}) = "false discovery rate; "*
                       "aliases: `multiclass_false_discovery_rate`, " *
                       "`multiclass_falsediscovery_rate`, `multiclass_fdr`."
MMI.name(::Type{<:MulticlassNPV})      = "multiclass_negative_predictive_value"
MMI.docstring(::Type{<:MulticlassNPV}) = "negative predictive value; aliases: " *
                       "`multiclass_negative_predictive_value`, " *
                       "`multiclass_negativepredictive_value`, `multiclass_npv`."
MMI.name(::Type{<:MulticlassPrecision})      = "multiclass_positive_predictive_value"
MMI.docstring(::Type{<:MulticlassPrecision}) = "positive predictive value "*
  "(aka precision); aliases: `multiclass_positive_predictive_value`, `multiclass_ppv`, " *
  "`MulticlassPrecision()`, `multiclass_positivepredictive_value`. "

const DS_MICRO_MACRO = "The `average` can be specified as `:micro` or " *
                       "`:macro`.`:macro` is taken as default mode, and it " *
                       "returns a value for each class. For `:micro`, pass "
const CLASS_W        = "`AbstractDict` of weights for each class can be " *
                       "specified using `class_w`."
const W_KEY_MISMATCH = "Encountered target with levels different from the " *
                       "keys of user-specified dictionary of class weights."
const W_PROMOTE_WARN = "As `class_w` specifies the weights for each class, " *
                       "promoting average to `:macro`."


"""
  multiclass_true_positive

$(docstring(MulticlassTruePositive()))

  multiclass_true_positive(ŷ, y)

Number of true positives for multiclass observations `ŷ` and ground truth
`y`. $DS_MICRO_MACRO `MulticlassTruePositive(; average=:micro)`.

For more information, run `info(multiclass_true_positive)`.

"""
const multiclass_true_positive  = MulticlassTruePositive()
const multiclass_truepositive   = MulticlassTruePositive()
const mtp = MulticlassTruePositive()

"""
  multiclass_true_negative

$(docstring(MulticlassTrueNegative()))

  multiclass_true_negative(ŷ, y)

Number of true negatives for multiclass observations `ŷ` and ground truth
`y`. $DS_MICRO_MACRO `MulticlassTrueNegative(; average=:micro)`.

For more information, run `info(multiclass_true_negative)`.

"""
const multiclass_true_negative  = MulticlassTrueNegative()
const multiclass_truenegative   = MulticlassTrueNegative()
const mtn = MulticlassTrueNegative()

"""
  multiclass_false_positive

$(docstring(MulticlassFalsePositive()))

  multiclass_false_positive(ŷ, y)

Number of false positives for multiclass observations `ŷ` and ground truth
`y`. $DS_MICRO_MACRO `MulticlassFalsePositive(; average=:micro)`.

For more information, run `info(multiclass_false_positive)`.

"""
const multiclass_false_positive = MulticlassFalsePositive()
const multiclass_falsepositive  = MulticlassFalsePositive()
const mfp = MulticlassFalsePositive()

"""
  multiclass_false_negative

$(docstring(MulticlassFalseNegative()))

  multiclass_false_negative(ŷ, y)

Number of false positives for multiclass observations `ŷ` and ground truth
`y`. $DS_MICRO_MACRO `MulticlassFalseNegative(; average=:micro)`.

For more information, run `info(multiclass_false_negative)`.

"""
const multiclass_false_negative = MulticlassFalseNegative()
const multiclass_falsenegative  = MulticlassFalseNegative()
const mfn = MulticlassFalseNegative()

"""
    multiclass_true_positive_rate

$(docstring(MulticlassTruePositiveRate()))

    multiclass_true_positive_rate(ŷ, y)
    multiclass_true_positive_rate(ŷ, y, class_w)

True positive rate for multiclass observations `ŷ` and ground truth `y`.
$DS_MICRO_MACRO `MulticlassTruePositiveRate(; average=:micro)`. $CLASS_W

For more information, run `info(multiclass_true_positive_rate)`.

"""
const multiclass_true_positive_rate = MulticlassTruePositiveRate()
const multiclass_truepositive_rate  = MulticlassTruePositiveRate()
const multiclass_tpr                = MulticlassTruePositiveRate()
const multiclass_recall             = MulticlassTruePositiveRate()
const multiclass_sensitivity        = MulticlassTruePositiveRate()
const multiclass_hit_rate           = MulticlassTruePositiveRate()
const MTPR                          = MulticlassTruePositiveRate
const MulticlassRecall              = MulticlassTruePositiveRate

"""
    multiclass_true_negative_rate

$(docstring(MulticlassTrueNegativeRate()))

    multiclass_true_negative_rate(ŷ, y)
    multiclass_true_negative_rate(ŷ, y, class_w)

True negative rate for multiclass observations `ŷ` and ground truth `y`.
$DS_MICRO_MACRO `MulticlassTrueNegativeRate(; average=:micro)`. $CLASS_W

For more information, run `info(multiclass_true_negative_rate)`.

"""
const multiclass_true_negative_rate = MulticlassTrueNegativeRate()
const multiclass_truenegative_rate  = MulticlassTrueNegativeRate()
const multiclass_tnr                = MulticlassTrueNegativeRate()
const multiclass_specificity        = MulticlassTrueNegativeRate()
const multiclass_selectivity        = MulticlassTrueNegativeRate()
const MulticlassSpecificity         = MulticlassTrueNegativeRate
const MTNR                          = MulticlassTrueNegativeRate

## define all consts #####

"""
    multiclass_false_positive_rate

$(docstring(MulticlassFalsePositiveRate()))

    multiclass_false_positive_rate(ŷ, y)
    multiclass_false_positive_rate(ŷ, y, class_w)

False positive rate for multiclass observations `ŷ` and ground truth `y`.
$DS_MICRO_MACRO `MulticlassFalsePositiveRate(; average=:micro)`. $CLASS_W

For more information, run `info(multiclass_false_positive_rate)`.

"""
const multiclass_false_positive_rate = MulticlassFalsePositiveRate()
const multiclass_falsepositive_rate  = MulticlassFalsePositiveRate()
const multiclass_fpr                 = MulticlassFalsePositiveRate()
const MFPR                           = MulticlassFalsePositiveRate
const multiclass_fallout             = MFPR()

"""
    multiclass_false_negative_rate

$(docstring(MulticlassFalseNegativeRate()))

    multiclass_false_negative_rate(ŷ, y)
    multiclass_false_negative_rate(ŷ, y, class_w)

False negative rate for multiclass observations `ŷ` and ground truth `y`.
$DS_MICRO_MACRO `MulticlassFalseNegativeRate(; average=:micro)`. $CLASS_W

For more information, run `info(multiclass_false_negative_rate)`.

"""
const multiclass_false_negative_rate = MulticlassFalseNegativeRate()
const multiclass_falsenegative_rate  = MulticlassFalseNegativeRate()
const multiclass_fnr                 = MulticlassFalseNegativeRate()
const MFNR                           = MulticlassFalseNegativeRate
const multiclass_miss_rate           = MFNR()

"""
    multiclass_false_discovery_rate

$(docstring(MulticlassTruePositiveRate()))

    multiclass_false_discovery_rate(ŷ, y)
    multiclass_false_discovery_rate(ŷ, y, class_w)

False discovery rate for multiclass observations `ŷ` and ground truth `y`.
$DS_MICRO_MACRO `MulticlassFalseDiscoveryRate(; average=:micro)`. $CLASS_W

For more information, run `info(multiclass_false_discovery_rate)`.

"""
const multiclass_false_discovery_rate = MulticlassFalseDiscoveryRate()
const multiclass_falsediscovery_rate  = MulticlassFalseDiscoveryRate()
const multiclass_fdr                  = MulticlassFalseDiscoveryRate()
const MFDR                            = MulticlassFalseDiscoveryRate

"""
    multiclass_positive_predictive_value

$(docstring(MulticlassPrecision()))

    multiclass_positive_predictive_value(ŷ, y)
    multiclass_positive_predictive_value(ŷ, y, class_w)

Positive predictive value for multiclass observations `ŷ` and ground truth
`y`. $DS_MICRO_MACRO `MulticlassPrecision(; average=:micro)`. $CLASS_W

For more information, run `info(multiclass_positive_predictive_value)`.

"""
const multiclass_precision                 = MulticlassPrecision()
const multiclass_ppv                       = MulticlassPrecision()
const multiclass_positive_predictive_value = MulticlassPrecision()
const multiclass_positivepredictive_value  = MulticlassPrecision()
const MPPV                                 = MulticlassPrecision

"""
    multiclass_negative_predictive_value

$(docstring(MulticlassNPV()))

    multiclass_negative_predictive_value(ŷ, y)
    multiclass_negative_predictive_value(ŷ, y, class_w)

Negative predictive value for multiclass observations `ŷ` and ground truth
`y`.$DS_MICRO_MACRO `MulticlassNPV(; average=:micro)`. $CLASS_W

For more information, run `info(multiclass_negative_predictive_value)`.

"""
const multiclass_npv                       = MulticlassNPV()
const multiclass_negative_predictive_value = MulticlassNPV()
const multiclass_negativepredictive_value  = MulticlassNPV()
const MNPV                                 = MulticlassNPV

## INTERNAL FUCNTIONS ON MULTICLASS CONFUSION MATRIX

_mtp(m::CM) = diag(m.mat)
_mtn(m::CM)= sum(m.mat) .- (sum(m.mat, dims=2) .+ sum(m.mat, dims=1)' .+ diag(m.mat))
_mfp(m::CM{N}) where N = N < 31 ? sum(m.mat, dims=2) .-= diag(m.mat) :
                                  sum(m.mat, dims=2) .- diag(m.mat)
_mfn(m::CM{N}) where N = N < 31 ? sum(m.mat, dims=1)' .-= diag(m.mat) :
                                  sum(m.mat, dims=1)' .- diag(m.mat)

@inline function _class_w(level_m::Vec{<:String}, class_w::AbstractDict{<:Any, <:Real})
    class_w_labels = levels(keys(class_w))
    string.(class_w_labels) == level_m || throw(ArgumentError(W_KEY_MISMATCH))
    return [class_w[l] for l in class_w_labels]
end

function _mtpr(m::CM{N}; average=:macro) where N
    n_classes = N
    mtp_val, mfn_val = mtp(m), mfn(m)
    average == :micro && return sum(mtp_val) / (sum(mtp_val) + sum(mfn_val))
    return mtp_val ./ (mtp_val + mfn_val)
end

function _mtpr(m::CM, class_w::AbstractDict{<:Any, <:Real}; average=:macro)
    average == :micro && @warn W_PROMOTE_WARN
    level_w = _class_w(m.labels, class_w)
    return _mtpr(m, average=:macro) .* level_w
end

function _mtnr(m::CM{N}; average=:macro) where N
    n_classes = N
    mtn_val, mfp_val = mtn(m), mfp(m)
    average == :micro && return sum(mtn_val) / (sum(mtn_val) + sum(mfp_val))
    return mtn_val ./ (mtn_val + mfp_val)
end

function _mtnr(m::CM, class_w::AbstractDict{<:Any, <:Real}; average=:macro)
    average == :micro && @warn W_PROMOTE_WARN
    level_w = _class_w(m.labels, class_w)
    return _mtnr(m, average=average) .* level_w
end

_mfpr(m::CM{N}; average=:macro) where N = 1 .- _mtnr(m, average=average)
function _mfpr(m::CM, class_w::AbstractDict{<:Any, <:Real}; average=:macro)
    level_w = _class_w(m.labels, class_w)
    return (1 .- _mtnr(m, average=average)) .* level_w
end

_mfnr(m::CM{N}; average=:macro) where N = 1 .- _mtpr(m, average=average)
function _mfnr(m::CM, class_w::AbstractDict{<:Any, <:Real}; average=:macro)
    level_w = _class_w(m.labels, class_w)
    return (1 .- _mtpr(m, average=average)) .* level_w
end

function _mfdr(m::CM{N}; average=:macro) where N
    n_classes = N
    mtp_val, mfp_val = mtp(m), mfp(m)
    average == :micro && return sum(mfp_val) / (sum(mtp_val) + sum(mfp_val))
    return mfp_val ./ (mtp_val + mfp_val)
end

function _mfdr(m::CM, class_w::AbstractDict{<:Any, <:Real}; average=:macro)
    average == :micro && @warn W_PROMOTE_WARN
    level_w = _class_w(m.labels, class_w)
    return _mfdr(m, average=average) .* level_w
end

function _mnpv(m::CM{N}; average=:micro) where N
    n_classes = N
    mtn_val, mfn_val = mtn(m), mfn(m)
    average == :micro && return sum(mtn_val) / (sum(mtn_val) + sum(mfn_val))
    return mtn_val ./ (mtn_val + mfn_val)
end


function _mnpv(m::CM, class_w::AbstractDict{<:Any, <:Real}; average=:macro)
    average == :micro && @warn W_PROMOTE_WARN
    level_w = _class_w(m.labels, class_w)
    return _mnpv(m, average=average) .* level_w
end

## CALLABLES ON MULTICLASS CONFUSION MATRIX

(::MulticlassTruePositive)(m::CM{N}) where N  = _mtp(m)
(::MulticlassTrueNegative)(m::CM{N}) where N  = _mtn(m)
(::MulticlassFalsePositive)(m::CM{N}) where N = _mfp(m)
(::MulticlassFalseNegative)(m::CM{N}) where N = _mfn(m)

(r::MTPR)(m::CM) = _mtpr(m, average=r.average)
(r::MTPR)(m::CM, w::AbstractDict{<:Any, <:Real}) = _mtpr(m, w, average=r.average)

(r::MTNR)(m::CM) = _mtnr(m, average=r.average)
(r::MTNR)(m::CM, w::AbstractDict{<:Any, <:Real}) = _mtnr(m, w, average=r.average)

(r::MFPR)(m::CM) = _mfpr(m, average=r.average)
(r::MFPR)(m::CM, w::AbstractDict{<:Any, <:Real}) = _mfpr(m, w, average=r.average)

(r::MFNR)(m::CM) = _mfnr(m, average=r.average)
(r::MFNR)(m::CM, w::AbstractDict{<:Any, <:Real}) = _mfnr(m, w, average=r.average)

(r::MFDR)(m::CM) = _mfdr(m, average=r.average)
(r::MFDR)(m::CM, w::AbstractDict{<:Any, <:Real}) = _mfdr(m, w, average=r.average)

(v::MNPV)(m::CM) = _mnpv(m, average=v.average)
(v::MNPV)(m::CM, w::AbstractDict{<:Any, <:Real}) = _mnpv(m, w, average=v.average)

(p::MulticlassPrecision)(m::CM) = 1.0 .- _mfdr(m, average=p.average)
function (p::MulticlassPrecision)(m::CM, class_w::AbstractDict{<:Any, <:Real})
    average == :micro && @warn W_PROMOTE_WARN
    level_w = _class_w(m.labels, class_w)
    return (1.0 .- _mfdr(m, average=p.average)) .* level_w
end

function (f::MulticlassFScore{β})(m::CM) where β
    β2   = β^2
    rec  = MulticlassRecall(; average=f.average)(m)
    prec = MulticlassPrecision(; average=f.average)(m)
    f.average == :micro && return (1 + β2) * (prec * rec) / (β* prec + rec)
    return (1 + β2) .* (prec .* rec) ./ (β2 .* prec .+ rec)
end

function (f::MulticlassFScore{β})(m::CM, class_w::AbstractDict{<:Any, <:Real}) where β
    f.average == :micro && @warn W_PROMOTE_WARN
    level_w = _class_w(m.labels, class_w)
    return MulticlassFScore{β}()(m) .* level_w
end

## Callables on vectors

for M in (MulticlassTruePositive, MulticlassTrueNegative,
          MulticlassFalsePositive, MulticlassFalseNegative)
    (m::M)(ŷ, y) = confmat(ŷ, y, warn=false) |> m
end

for M in (MTPR, MTNR, MFPR, MFNR, MFDR, MulticlassPrecision, MNPV)
    @eval (m::$M)(ŷ, y) = m(confmat(ŷ, y, warn=false))
    @eval (m::$M)(ŷ, y, class_w::AbstractDict{<:Any, <:Real}) =
                          m(confmat(ŷ, y, warn=false), class_w)
end

(m::MulticlassFScore{β})(ŷ, y) where β =
    MulticlassFScore{β}(; average=m.average)(confmat(ŷ, y, warn=false))
(m::MulticlassFScore{β})(ŷ, y, class_w::AbstractDict{<:Any, <:Real}) where β =
    MulticlassFScore{β}(; average=m.average)(confmat(ŷ, y, warn=false), class_w)

## ROC COMPUTATION

"""
    _idx_unique_sorted(v)

Internal function to return the index of unique elements in `v` under the
assumption that the vector `v` is sorted in decreasing order.
"""
function _idx_unique_sorted(v::Vec{<:Real})
    n    = length(v)
    idx  = ones(Int, n)
    p, h = 1, 1
    cur  = v[1]
    @inbounds while h < n
        h     += 1                  # head position
        cand   = v[h]               # candidate value
        cand   < cur || continue    # is it new? otherwise skip
        p     += 1                  # if new store it
        idx[p] = h
        cur    = cand               # and update the last seen value
    end
    p < n && deleteat!(idx, p+1:n)
    return idx
end

"""
    fprs, tprs, ts = roc_curve(ŷ, y) = roc(ŷ, y)

Return the ROC curve for a two-class probabilistic prediction `ŷ` given the
ground  truth `y`. The true positive rates, false positive rates over a range
of thresholds `ts` are returned. Note that if there are `k` unique scores,
there are correspondingly  `k` thresholds and `k+1` "bins" over which the FPR
and TPR are constant:

* `[0.0 - thresh[1]]`
* `[thresh[1] - thresh[2]]`
* ...
* `[thresh[k] - 1]`

consequently, `tprs` and `fprs` are of length `k+1` if `ts` is of length `k`.

To draw the curve using your favorite plotting backend, do `plot(fprs, tprs)`.
"""
function roc_curve(ŷ::Vec{<:UnivariateFinite},
                   y::Vec{<:CategoricalValue})

    n       = length(y)
    lab_pos = levels(y)[2]
    scores  = pdf.(ŷ, lab_pos)
    ranking = sortperm(scores, rev=true)

    scores_sort = scores[ranking]
    y_sort_bin  = (y[ranking] .== lab_pos)

    idx_unique = _idx_unique_sorted(scores_sort)
    thresholds = scores_sort[idx_unique]

    # detailed computations with example:
    # y = [  1   0   0   1   0   0   1]
    # s = [0.5 0.5 0.2 0.2 0.1 0.1 0.1] thresh are 0.5 0.2, 0.1 // idx [1, 3, 5]
    # ŷ = [  0   0   0   0   0   0   0] (0.5 - 1.0] # no pos pred
    # ŷ = [  1   1   0   0   0   0   0] (0.2 - 0.5] # 2 pos pred
    # ŷ = [  1   1   1   1   0   0   0] (0.1 - 0.2] # 4 pos pred
    # ŷ = [  1   1   1   1   1   1   1] [0.0 - 0.1] # all pos pre

    idx_unique_2 = idx_unique[2:end]   # [3, 5]
    n_ŷ_pos      = idx_unique_2 .- 1   # [2, 4] implicit [0, 2, 4, 7]

    cs   = cumsum(y_sort_bin)          # [1, 1, 1, 2, 2, 2, 3]
    n_tp = cs[n_ŷ_pos]                 # [1, 2] implicit [0, 1, 2, 3]
    n_fp = n_ŷ_pos .- n_tp             # [1, 2] implicit [0, 1, 2, 4]

    # add end points
    P = sum(y_sort_bin) # total number of true positives
    N = n - P           # total number of true negatives

    n_tp = [0, n_tp..., P] # [0, 1, 2, 3]
    n_fp = [0, n_fp..., N] # [0, 1, 2, 4]

    tprs = n_tp ./ P  # [0/3, 1/3, 2/3, 1]
    fprs = n_fp ./ N  # [0/4, 1/4, 2/4, 1]

    return fprs, tprs, thresholds
end

const roc = roc_curve
