# ---------------------------------------------------
## CLASSIFICATION METRICS (PROBABILISTIC PREDICTIONS)
# >> CrossEntropy
# >> BriersScore
# ---------------------------------------------------

"""
    ce = CrossEntropy(; eps=eps())
    ce(ŷ, y)

Given an abstract vector of distributions `ŷ` and an abstract vector
of true observations `y`, return the corresponding Cross-Entropy
loss (aka log loss) scores.

Since the score is undefined in the case of the true observation has
predicted probability zero, probablities are clipped between `eps` and `1-eps`
where `eps` can be specified.

If `sᵢ` is the predicted probability for the true class `yᵢ` then
the score for that example is given by

``-log(clamp(sᵢ, eps, 1-eps))``

For more information, run `info(cross_entropy)`.
"""

struct CrossEntropy{R} <: Measure where R <: AbstractFloat
    eps::R
end
CrossEntropy(;eps=eps()) = CrossEntropy(eps)

"""
    cross_entropy(ŷ, y::Vec{<:Finite})

Given an abstract vector of `UnivariateFinite` distributions `ŷ` (ie,
probabilistic predictions) and an abstract vector of true observations
`y`, return the negative log-probability that each observation would
occur, according to the corresponding probabilistic prediction.

For more information, run `info(cross_entropy)`.
"""
cross_entropy = CrossEntropy()

metadata_measure(CrossEntropy;
    name                     = "cross_entropy",
    target_scitype           = Vec{<:Finite},
    prediction_type          = :probabilistic,
    orientation              = :loss,
    reports_each_observation = true,
    is_feature_dependent     = false,
    supports_weights         = false,
    docstring                = "Cross entropy loss with probabilities " *
                               "clamped between eps and 1-eps; aliases: " *
                               "`cross_entropy`.",
    distribution_type        = UnivariateFinite)

# for single observation:
_cross_entropy(d, y, eps) = -log(clamp(pdf(d, y), eps, 1 - eps))

function (c::CrossEntropy)(ŷ::Vec{<:UnivariateFinite},
                           y::Vec{<:CategoricalElement})
    check_dimensions(ŷ, y)
    check_pools(ŷ, y)
    return broadcast(_cross_entropy, ŷ, y, c.eps)
end

# TODO: support many distributions/samplers D below:
struct BrierScore{D} <: Measure end

"""
    brier = BrierScore(; distribution=UnivariateFinite)
    brier(ŷ, y)

Given an abstract vector of distributions `ŷ` and an abstract vector
of true observations `y`, return the corresponding Brier (aka
quadratic) scores.

*Warning.* `BrierScore` defines a true *score* (bigger is better). In
Brier's original 1950 paper, and some other places, it is defined as a
loss, despite the name. The binary case is *not* treated as special in
MLJ, so that the score may differ by a factor of two in the binary
case from usage elsewhere. The precise formula used is given below.

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

metadata_measure(BrierScore;
    target_scitype           = Vec{<:Finite},
    prediction_type          = :probabilistic,
    orientation              = :score,
    reports_each_observation = true,
    is_feature_dependent     = false,
    supports_weights         = true,
    distribution_type        = UnivariateFinite)

# adjustments
MMI.name(::Type{<:BrierScore{D}}) where D      = "BrierScore{$(string(D))}"
MMI.docstring(::Type{<:BrierScore{D}}) where D =
    "Brier proper scoring rule for distributions of type $D; " *
    "aliases: `BrierScore($D)`."

# For single observations (no checks):

# UnivariateFinite:
function brier_score(d::UnivariateFinite, y)
    levels = classes(d)
    pvec = broadcast(pdf, d, levels)
    offset = 1 + sum(pvec.^2)
    return 2 * pdf(d, y) - offset
end

# For multiple observations:

# UnivariateFinite:
function (::BrierScore{<:UnivariateFinite})(ŷ::Vec{<:UnivariateFinite},
                                            y::Vec{<:CategoricalElement})
    check_dimensions(ŷ, y)
    check_pools(ŷ, y)
    return broadcast(brier_score, ŷ, y)
end

function (score::BrierScore{<:UnivariateFinite})(ŷ, y, w::Vec{<:Real})
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

const INVARIANT_LABEL_MULTICLASS = "This metric is invariant to class labelling and can be used for multiclass classification."
const INVARIANT_LABEL_BINARY = "This metric is invariant to class labelling and can be used only for binary classification."
const VARIANT_LABEL_BINARY = "This metric is labelling-dependent and can only be used for binary classification."

struct MisclassificationRate <: Measure end

"""
    misclassification_rate(ŷ, y)
    misclassification_rate(ŷ, y, w)
    misclassification_rate(conf_mat)

Returns the rate of misclassification of the (point) predictions `ŷ`,
given true observations `y`, optionally weighted by the weights
`w`. All three arguments must be abstract vectors of the same length.
A confusion matrix can also be passed as argument.
$INVARIANT_LABEL_MULTICLASS

For more information, run `info(misclassification_rate)`.
You can also equivalently use `mcr`.
"""
const misclassification_rate = MisclassificationRate()
const mcr = misclassification_rate

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

const MCR = MisclassificationRate

(::MCR)(ŷ::Vec{<:CategoricalElement},
        y::Vec{<:CategoricalElement}) = mean(y .!= ŷ)

(::MCR)(ŷ::Vec{<:CategoricalElement},
        y::Vec{<:CategoricalElement},
        w::Vec{<:Real}) = sum((y .!= ŷ) .* w) / sum(w)

(::MCR)(cm::ConfusionMatrix) = 1.0 - sum(diag(cm.mat)) / sum(cm.mat)


struct Accuracy <: Measure end

"""
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

metadata_measure(Accuracy;
    name                     = "accuracy",
    target_scitype           = Vec{<:Finite},
    prediction_type          = :deterministic,
    orientation              = :score,
    reports_each_observation = false,
    is_feature_dependent     = false,
    supports_weights         = true,
    docstring                = "Classification accuracy; aliases: `accuracy`.")

struct BalancedAccuracy <: Measure end

const BACC = BalancedAccuracy

"""
    balanced_accuracy(ŷ, y [, w])
    bacc(ŷ, y [, w])
    bac(ŷ, y [, w])
    balanced_accuracy(conf_mat)

Return the balanced accuracy of the point prediction `ŷ`, given true
observations `y`, optionally weighted by `w`. The balanced accuracy takes
into consideration class imbalance.
All  three arguments must have the same length.
$INVARIANT_LABEL_MULTICLASS

For more information, run `info(balanced_accuracy)`.
"""
const balanced_accuracy = BACC()
const bacc = balanced_accuracy
const bac  = bacc

function (::BACC)(ŷ::Vec{<:CategoricalElement},
                  y::Vec{<:CategoricalElement})
    class_count = Dist.countmap(y)
    ŵ = 1.0 ./ [class_count[yi] for yi in y]
    return sum( (ŷ .== y) .* ŵ ) / sum(ŵ)
end

function (::BACC)(ŷ::Vec{<:CategoricalElement},
                  y::Vec{<:CategoricalElement},
                  w::Vec{<:Real})
    levels_ = levels(y)
    ŵ = similar(w)
    @inbounds for i in eachindex(w)
        ŵ[i] = w[i] / sum(w .* (y .== y[i]))
    end
    return sum( (ŷ .== y) .* ŵ ) / sum(ŵ)
end

metadata_measure(BACC;
    name                     = "balanced_accuracy",
    target_scitype           = Vec{<:Finite},
    prediction_type          = :deterministic,
    orientation              = :score,
    reports_each_observation = false,
    is_feature_dependent     = false,
    supports_weights         = true,
    docstring                = "Balanced classification accuracy; aliases: "*
                               "`balanced_accuracy`, `bacc`, `bac`.")


## BINARY BUT ORDER-DEPENDENT

struct MatthewsCorrelation <: Measure end

const MCC = MatthewsCorrelation

"""
    matthews_correlation(ŷ, y)
    mcc(ŷ, y)
    matthews_correlation(conf_mat)

Return Matthews' correlation coefficient corresponding to the point
prediction `ŷ`, given true observations `y`.
$INVARIANT_LABEL_MULTICLASS

For more information, run `info(matthews_correlation)`.
"""
const matthews_correlation = MCC()
const mcc = matthews_correlation

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

(m::MCC)(ŷ::Vec{<:CategoricalElement},
         y::Vec{<:CategoricalElement}) =
             confmat(ŷ, y, warn=false) |> m

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

struct AUC <: Measure end

"""
    auc(ŷ, y)

Return the Area Under the (ROC) Curve for probabilistic prediction `ŷ` given
true observations `y`.
$INVARIANT_LABEL_BINARY

For more information, run `info(auc)`.
"""
const auc = AUC()

function (::AUC)(ŷ::Vec{<:UnivariateFinite},
                 y::Vec{<:CategoricalElement})
    # implementation drawn from https://www.ibm.com/developerworks/community/blogs/jfp/entry/Fast_Computation_of_AUC_ROC_score?lang=en
    lab_pos = levels(y)[2]         # 'positive' label
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


metadata_measure(AUC;
    name                     = "auc",
    target_scitype           = Vec{<:Finite},
    prediction_type          = :probabilistic,
    orientation              = :score,
    reports_each_observation = false,
    is_feature_dependent     = false,
    supports_weights         = false,
    docstring                = "Area under the ROC curve; aliases: `auc`.")

## Binary and order dependent

const CM2 = ConfusionMatrix{2}

"""
    fb = FScore{β}(rev=nothing)
 
One-parameter generalization, `F_β`, of the F-measure or balanced F-score.

[Wikipedia entry]([https://en.wikipedia.org/wiki/F1_score])

    fb(ŷ, y)

Evaluate `F_β` on observations ,`ŷ`, given ground truth values, `y`.

By default, the second element of `levels(y)` is designated as `true`,
unless `rev=true`.

For more information, run `info(FScore)`.

"""
struct FScore{β} <: Measure rev::Union{Nothing,Bool} end

FScore{β}(; rev=nothing) where β = FScore{β}(rev)
const f1score      = FScore{1}()

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

metadata_measure.((TruePositiveRate, TrueNegativeRate, Precision, FScore, NPV);
    target_scitype           = Vec{<:Finite},
    prediction_type          = :deterministic,
    orientation              = :score,
    reports_each_observation = false,
    is_feature_dependent     = false,
    supports_weights         = false)

# adjustments
MMI.name(::Type{<:TruePositive})       = "tp"
MMI.docstring(::Type{<:TruePositive})  = "Number of true positives; " *
                                         "aliases: `truepositive`, `tp`."
MMI.name(::Type{<:TrueNegative})       = "tn"
MMI.docstring(::Type{<:TrueNegative})  = "Number of true negatives; " *
                                         "aliases: `truenegative`, `tn`."
MMI.name(::Type{<:FalsePositive})      = "fp"
MMI.docstring(::Type{<:FalsePositive}) = "Number of false positives; " *
                                         "aliases: `falsepositive`, `fp`."
MMI.name(::Type{<:FalseNegative})      = "fn"
MMI.docstring(::Type{<:FalseNegative}) = "Number of false negatives; " *
                                         "aliases: `falsenegative`, `fn`."

MMI.name(::Type{<:TruePositiveRate})      = "tpr"
MMI.docstring(::Type{<:TruePositiveRate}) = "True postive rate; aliases: " *
                               "`truepositive_rate`, `tpr`, `sensitivity`, " *
                               "`recall`, `hit_rate`."
MMI.name(::Type{<:TrueNegativeRate})      = "tnr"
MMI.docstring(::Type{<:TrueNegativeRate}) = "true negative rate; aliases: " *
                               "`truenegative_rate`, `tnr`, `specificity`, " *
                               "`selectivity`."
MMI.name(::Type{<:FalsePositiveRate})      = "fpr"
MMI.docstring(::Type{<:FalsePositiveRate}) = "false positive rate; aliases: " *
                               "`falsepositive_rate`, `fpr`, `fallout`."
MMI.name(::Type{<:FalseNegativeRate})      = "fnr"
MMI.docstring(::Type{<:FalseNegativeRate}) = "false negative rate; aliases: " *
                               "`falsenegative_rate`, `fnr`, `miss_rate`."
MMI.name(::Type{<:FalseDiscoveryRate})      = "fdr"
MMI.docstring(::Type{<:FalseDiscoveryRate}) = "false discovery rate; "*
                               "aliases: `falsediscovery_rate`, `fdr`."
MMI.name(::Type{<:NPV})      = "npv"
MMI.docstring(::Type{<:NPV}) = "negative predictive value; aliases: " *
                               "`negativepredictive_value`, `npv`."

MMI.name(::Type{<:Precision})         = "ppv"
MMI.docstring(::Type{<:Precision})    = "positive predictive value "*
  "(aka precision); aliases: `ppv`, `Precision()`, `positivepredictive_value`. "

MMI.name(::Type{<:FScore{β}}) where β = "FScore($β)"
MMI.name(::Type{<:FScore})            = "FScore(β)" # for registry
MMI.docstring(::Type{<:FScore})       = "F_β score; aliases: " *
                                        "`FScore(β)`, `f1score=FScore(1)`"

"""
    tp

$(docstring(TruePositive()))

    tp(ŷ, y)

Number of true positives for observations `ŷ` and ground truth
`y`. Assigns `true` to first element of `levels(y)`. To reverse roles,
use `TruePositive(rev=true)` instead of `tp`.

For more information, run `info(tp)`.

"""
const tp = TruePositive()
const truepositive  = TruePositive()

"""
    tn

$(docstring(TrueNegative()))

    tn(ŷ, y)

Number of true negatives for observations `ŷ` and ground truth
`y`. Assigns `true` to first element of `levels(y)`. To reverse roles,
use `TrueNegative(rev=true)` instead of `tn`.


For more information, run `info(tn)`.
"""
const tn = TrueNegative()
const truenegative  = TrueNegative()

"""
    fp

$(docstring(FalsePositive()))

    fp(ŷ, y)

Number of false positives for observations `ŷ` and ground truth
`y`. Assigns `true` to first element of `levels(y)`. To reverse roles,
use `FalsePositive(rev=true)` instead of `fp`.


For more information, run `info(fp)`.
"""
const fp = FalsePositive()
const falsepositive = FalsePositive()

"""
    fn

$(docstring(FalseNegative()))

    fn(ŷ, y)

Number of false positives for observations `ŷ` and ground truth
`y`. Assigns `true` to first element of `levels(y)`. To reverse roles,
use `FalseNegative(rev=true)` instead of `fn`.

For more information, run `info(fn)`.

"""
const fn = FalseNegative()
const falsenegative = FalseNegative()

"""
    tpr

$(docstring(TruePositiveRate()))

    tpr(ŷ, y)

True positive rate for observations `ŷ` and ground truth `y`. Assigns
`true` to first element of `levels(y)`. To reverse roles, use
`TPR(rev=true)` instead of `tpr`.

For more information, run `info(tpr)`.

"""
const tpr = TruePositiveRate()
const TPR = TruePositiveRate
const truepositive_rate  = TPR()
const recall = TPR()
const Recall = TPR
const sensitivity  = recall
const hit_rate     = recall

"""
    tnr

$(docstring(TrueNegativeRate()))

    tpr(ŷ, y)

True negative rate for observations `ŷ` and ground truth `y`. Assigns
`true` to first element of `levels(y)`. To reverse roles, use
`TNR(rev=true)` instead of `tnr`.

For more information, run `info(tnr)`.

"""
const tnr = TrueNegativeRate()
const TNR = TrueNegativeRate
const truenegative_rate  = TNR()
const Specificity = TNR
const specificity  = truenegative_rate
const selectivity  = specificity

"""
    fpr

$(docstring(FalsePositiveRate()))

    fpr(ŷ, y)

False positive rate for observations `ŷ` and ground truth `y`. Assigns
`true` to first element of `levels(y)`. To reverse roles, use
`FPR(rev=true)` instead of `fpr`.

"""
const fpr = FalsePositiveRate()
const FPR = FalsePositiveRate
const falsepositive_rate = FPR()
const fallout      = falsepositive_rate

"""
    fnr

$(docstring(FalseNegativeRate()))

    fnr(ŷ, y)

False negative rate for observations `ŷ` and ground truth `y`. Assigns
`true` to first element of `levels(y)`. To reverse roles, use
`FNR(rev=true)` instead of `fnr`.

For more information, run `info(fnr)`.

"""
const fnr = FalseNegativeRate()
const FNR = FalseNegativeRate
const falsenegative_rate = FNR()


"""
    fdr

$(docstring(FalseDiscoveryRate()))

    fdr(ŷ, y)

False discovery rate for observations `ŷ` and ground truth `y`. Assigns
`true` to first element of `levels(y)`. To reverse roles, use
`FDR(rev=true)` instead of `fdr`.

For more information, run `info(fdr)`.

"""
const fdr = FalseDiscoveryRate()
const FDR = FalseDiscoveryRate
const falsediscovery_rate = FDR()
const miss_rate    = falsenegative_rate

"""
    npv

$(docstring(NPV()))

    npv(ŷ, y)

Negative predictive value for observations `ŷ` and ground truth
`y`. Assigns `true` to first element of `levels(y)`. To reverse roles,
use `NPV(rev=true)` instead of `npv`.

For more information, run `info(npv)`.

"""
const npv = NPV()
const negativepredictive_value = NPV()

"""
    ppv

$(docstring(Precision()))

    ppv(ŷ, y)

Positive predictive value for observations `ŷ` and ground truth
`y`. Assigns `true` to first element of `levels(y)`. To reverse roles,
use `Precision(rev=true)` instead of `ppv`.

For more information, run `info(ppv)`.

"""
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

* [0.0 - thresh[1]]
* [thresh[1] - thresh[2]]
* ...
* [thresh[k] - 1]

consequently, `tprs` and `fprs` are of length `k+1` if `ts` is of length `k`.

To draw the curve using your favorite plotting backend, do `plot(fprs, tprs)`.
"""
function roc_curve(ŷ::Vec{<:UnivariateFinite},
                   y::Vec{<:CategoricalElement})

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
