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
docstring(::Type{<:CrossEntropy}) =
    "Cross entropy loss; aliases: `cross_entropy`"
target_scitype(::Type{<:CrossEntropy}) = AbstractVector{<:Finite}
prediction_type(::Type{<:CrossEntropy}) = :probabilistic
orientation(::Type{<:CrossEntropy}) = :loss
reports_each_observation(::Type{<:CrossEntropy}) = true
is_feature_dependent(::Type{<:CrossEntropy}) = false
supports_weights(::Type{<:CrossEntropy}) = false
distribution_type(::Type{<:CrossEntropy}) = UnivariateFinite

# for single observation:
_cross_entropy(d, y) = -log(clamp(pdf(d, y), eps(), 1.0-eps()))

function (::CrossEntropy)(ŷ::AbstractVector{<:UnivariateFinite},
                          y::AbstractVector{<:CategoricalElement})
    check_dimensions(ŷ, y)
    check_pools(ŷ, y)
    return broadcast(_cross_entropy, ŷ, y)
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

name(::Type{<:BrierScore{D}}) where D = "BrierScore($(string(D)))"
docstring(::Type{<:BrierScore{D}}) where D =
    "Brier proper scoring rule for distributions of type $D; "*
    "aliases: `BrierScore($D)`"
docstring(::Type{<:BrierScore{<:UnivariateFinite}}) =
    "Brier proper scoring rule for `MultiClass` data; "*
    "aliases: `BrierScore()`, `BrierScore(UnivariateFinite)`"
target_scitype(::Type{<:BrierScore{D}}) where D = AbstractVector{<:Finite}
prediction_type(::Type{<:BrierScore}) = :probabilistic
orientation(::Type{<:BrierScore}) = :score
reports_each_observation(::Type{<:BrierScore}) = true
is_feature_dependent(::Type{<:BrierScore}) = false
supports_weights(::Type{<:BrierScore}) = true
distribution_type(::Type{<:BrierScore{D}}) where D = UnivariateFinite

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
function (::BrierScore{<:UnivariateFinite})(
    ŷ::AbstractVector{<:UnivariateFinite},
    y::AbstractVector{<:CategoricalElement})
    check_dimensions(ŷ, y)
    check_pools(ŷ, y)
    return broadcast(brier_score, ŷ, y)
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

name(::Type{<:MisclassificationRate}) = "misclassification_rate"
docstring(::Type{<:MisclassificationRate}) =
    "misclassification rate; aliases: `misclassification_rate`, `mcr`"
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
$INVARIANT_LABEL_MULTICLASS

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
    supports_weights=true,
    docstring="accuracy; aliases: `accuracy`")


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
    name="balanced_accuracy",
    target_scitype=AbstractVector{<:Finite},
    prediction_type=:deterministic,
    orientation=:score,
    reports_each_observation=false,
    is_feature_dependent=false,
    supports_weights=true,
    docstring="balanced accuracy; aliases: "*
              "`balanced_accuracy`, `bacc`, `bac`")


## Binary but order independent

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

(m::MCC)(ŷ::AbstractVector{<:CategoricalElement},
         y::AbstractVector{<:CategoricalElement}) =
             confmat(ŷ, y, warn=false) |> m

metadata_measure(MatthewsCorrelation;
    name="matthews_correlation",
    target_scitype=AbstractVector{<:Finite{2}},
    prediction_type=:deterministic,
    orientation=:score,
    reports_each_observation=false,
    is_feature_dependent=false,
    supports_weights=false,
    docstring="Matthew's correlation; aliases: `matthews_correlation`, `mcc`")


struct AUC <: Measure end

"""
auc(ŷ, y)

Return the Area Under the (ROC) Curve for probabilistic prediction `ŷ` given true
observations `y`.
$INVARIANT_LABEL_BINARY

For more information, run `info(auc)`.
"""
const auc = AUC()

function (::AUC)(ŷ::AbstractVector{<:UnivariateFinite},
                 y::AbstractVector{<:CategoricalElement})
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
    name="auc",
    target_scitype=AbstractVector{<:Finite},
    prediction_type=:probabilistic,
    orientation=:score,
    reports_each_observation=false,
    is_feature_dependent=false,
    supports_weights=false,
    docstring = "area under the curve; aliases: `auc`")

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
const negativepredictive_value = NPV()
const positivepredictive_value = Precision()
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

metadata_measure.((FalsePositive, FalseNegative);
    target_scitype=AbstractVector{<:Finite},
    prediction_type=:deterministic,
    orientation=:loss,
    reports_each_observation=false,
    aggregation=Sum(),
    is_feature_dependent=false,
    supports_weights=false)

metadata_measure.((FPR, FNR, FDR);
    target_scitype=AbstractVector{<:Finite},
    prediction_type=:deterministic,
    orientation=:loss,
    reports_each_observation=false,
    is_feature_dependent=false,
    supports_weights=false)

metadata_measure.((TruePositive, TrueNegative);
    target_scitype=AbstractVector{<:Finite},
    prediction_type=:deterministic,
    orientation=:score,
    reports_each_observation=false,
    aggregation=Sum(),
    is_feature_dependent=false,
    supports_weights=false)

metadata_measure.((TPR, TNR, Precision, FScore, NPV);
    target_scitype=AbstractVector{<:Finite},
    prediction_type=:deterministic,
    orientation=:score,
    reports_each_observation=false,
    is_feature_dependent=false,
    supports_weights=false)

# adjustments
name(::Type{<:TruePositive})  = "tp"
docstring(::Type{<:TruePositive})  = "number of true positives; aliases: "*
    "`truepositive`, `tp`"
name(::Type{<:TrueNegative})  = "tn"
docstring(::Type{<:TrueNegative})  = "number of true negatives; aliases: "*
    "`truenegative`, `tn`"
name(::Type{<:FalsePositive}) = "fp"
docstring(::Type{<:FalsePositive}) = "number of false positives; aliases: "*
    "`falsepositive`, `fp`"
name(::Type{<:FalseNegative}) = "fn"
docstring(::Type{<:FalseNegative}) = "number of false negatives; aliases: "*
    "`falsenegative`, `fn`"

name(::Type{<:TPR}) = "tpr"
docstring(::Type{<:TPR}) = "true postive rate; aliases: "*
    "`truepositive_rate`, `tpr`, `sensitivity`, `recall`, `hit_rate`"
name(::Type{<:TNR}) = "tnr"
docstring(::Type{<:TNR}) = "true negative rate; aliases: "*
    "`truenegative_rate`, `tnr`, `specificity`, `selectivity`"
name(::Type{<:FPR}) = "fpr"
docstring(::Type{<:FPR}) = "false positive rate; aliases: "*
    "`falsepositive_rate`, `fpr`, `fallout`"
name(::Type{<:FNR}) = "fnr"
docstring(::Type{<:FNR}) = "false negative rate; aliases: "*
    "`falsenegative_rate`, `fnr`, `miss_rate`"

name(::Type{<:FDR}) = "fdr"
docstring(::Type{<:FDR}) = "false discovery rate; aliases: "*
    "`falsediscovery_rate`, `fdr`"
name(::Type{<:NPV}) = "npv"
docstring(::Type{<:NPV}) = "negative predictive value; aliases: "*
    "`negativepredictive_value`, `npv`"

name(::Type{<:Precision}) = "ppv"
docstring(::Type{<:Precision}) = "precision; aliases: "*
    "`precision`, `positivepredictive_value`, `ppv`"
name(::Type{<:FScore{β}}) where β = "FScore($β)"
name(::Type{<:FScore}) = "FScore(β)" # for registry
docstring(::Type{<:FScore}) = "F_β score; aliases: "*
    "`FScore(β)`, `f1=f1score=FScore(1)`"


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


## ROC computation

"""
_idx_unique_sorted(v)

Internal function to return the index of unique elements in `v` under the assumption
that the vector `v` is sorted in decreasing order.
"""
function _idx_unique_sorted(v::AbstractVector{<:Real})
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

Return the ROC curve for a two-class probabilistic prediction `ŷ` given the ground  truth `y`.
The true positive rates, false positive rates over a range of thresholds `ts` are returned.
Note that if there are `k` unique scores, there are correspondingly  `k` thresholds and `k+1` "bins" over which the FPR and TPR are constant:

* [0.0 - thresh[1]]
* [thresh[1] - thresh[2]]
* ...
* [thresh[k] - 1]

consequently, `tprs` and `fprs` are of length `k+1` if `ts` is of length `k`.

To draw the curve using your favorite plotting backend, do `plot(fprs, tprs)`.
"""
function roc_curve(ŷ::AbstractVector{<:UnivariateFinite},
                   y::AbstractVector{<:CategoricalElement})

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
