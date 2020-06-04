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

# for single observation NO LONGER USED:
_cross_entropy(d, y, eps) = -log(clamp(pdf(d, y), eps, 1 - eps))

function (c::CrossEntropy)(ŷ::Vec{<:UnivariateFinite},
                           y::Vec)
    check_dimensions(ŷ, y)
    check_pools(ŷ, y)
    return -log.(clamp.(broadcast(pdf, ŷ, y), c.eps, 1 - c.eps))
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
MLJModelInterface.supports_weights(::Type{<:BrierScore}) = false
distribution_type(::Type{<:BrierScore{D}}) where D =
    UnivariateFinite

"""
    BrierScore(; distribution=UnivariateFinite)(ŷ, y)

Given an abstract vector of distributions `ŷ` of type `distribution`,
and an abstract vector of true observations `y`, return the
corresponding Brier (aka quadratic) scores.

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

# UnivariateFinite: NO LONGER USED
function _brier_score(d::UnivariateFinite, y)
    levels = classes(d)
    pvec = broadcast(pdf, d, levels)
    offset = 1 + sum(pvec.^2)
    return 2 * pdf(d, y) - offset
end

# For multiple observations:

# UnivariateFinite:
function (::BrierScore{<:UnivariateFinite})(
    ŷ::Vec{UnivariateFinite{S,V,R,P}},
    y::Vec) where {S,V,R,P}

    check_dimensions(ŷ, y)
    isempty(y) && return P(0)

    check_pools(ŷ, y)

    probs = pdf(ŷ, classes(first(ŷ)))
    offset = P(1) .+ sum(probs.^2, dims=2)
    return P(2) .* broadcast(pdf, ŷ, y) .- offset
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
    target_scitype           = Vec{<:Finite},
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

function (::AUC)(ŷ::Vec{<:UnivariateFinite},
                 y::Vec)
    # implementation drawn from https://www.ibm.com/developerworks/community/blogs/jfp/entry/Fast_Computation_of_AUC_ROC_score?lang=en
    lab_pos = classes(first(ŷ))[2] # 'positive' label
    scores  = pdf.(ŷ, lab_pos)     # associated scores
    y_sort  = y[sortperm(scores)]  # sort by scores
    n       = length(y)
    n_neg   = 0  # to keep of the number of negative preds
    auc     = 0
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
    target_scitype           = Vec{<:Finite},
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
    target_scitype           = Vec{<:Finite},
    prediction_type          = :deterministic,
    orientation              = :loss,
    reports_each_observation = false,
    aggregation              = Sum(),
    is_feature_dependent     = false,
    supports_weights         = false)

metadata_measure.((FalsePositiveRate, FalseNegativeRate, FalseDiscoveryRate);
    target_scitype           = Vec{<:Finite},
    prediction_type          = :deterministic,
    orientation              = :loss,
    reports_each_observation = false,
    is_feature_dependent     = false,
    supports_weights         = false)

metadata_measure.((TruePositive, TrueNegative);
    target_scitype           = Vec{<:Finite},
    prediction_type          = :deterministic,
    orientation              = :score,
    reports_each_observation = false,
    aggregation              = Sum(),
    is_feature_dependent     = false,
    supports_weights         = false)

metadata_measure.((TruePositiveRate, TrueNegativeRate, Precision, NPV);
    target_scitype           = Vec{<:Finite},
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
const miss_rate    = falsenegative_rate

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
    tprs, fprs, ts = roc_curve(ŷ, y) = roc(ŷ, y)

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
