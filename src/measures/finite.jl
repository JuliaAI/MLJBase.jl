# ============================================================
# PROBABILISTIC PREDICTIONS

# -----------------------------------------------------
# LogLoss

struct LogLoss{R} <: Measure where R <: Real
    tol::R
end
LogLoss(;eps=eps(), tol=eps) = LogLoss(tol)

metadata_measure(LogLoss;
                 instances                = ["log_loss", "cross_entropy"],
                 target_scitype           = Vec{<:Finite},
                 prediction_type          = :probabilistic,
                 orientation              = :loss,
                 reports_each_observation = true,
                 is_feature_dependent     = false,
                 supports_weights         = false,
                 distribution_type        = UnivariateFinite)

const CrossEntropy = LogLoss
@create_aliases LogLoss

@create_docs(LogLoss,
body=
"""
Since the score is undefined in the case that the true observation is
predicted to occur with probability zero, probablities are clipped
between `tol` and `1-tol`, where `tol` is a constructor key-word
argument.

If `sᵢ` is the predicted probability for the true class `yᵢ` then
the score for that example is given by

    -log(clamp(sᵢ, tol), 1 - tol)

A score is reported for every observation.
""",
scientific_type=DOC_FINITE)

# for single observation:
_cross_entropy(d::UnivariateFinite{S,V,R,P}, y, tol) where {S,V,R,P} =
    -log(clamp(pdf(d, y), P(tol), P(1) - P(tol)))

# multiple observations:
function (c::LogLoss)(ŷ::Vec{<:UnivariateFinite},
                           y::Vec)
    check_dimensions(ŷ, y)
    check_pools(ŷ, y)
    return broadcast(_cross_entropy, ŷ, y, c.tol)
end
# performant in case of UnivariateFiniteArray:
function (c::LogLoss)(ŷ::UnivariateFiniteVector{S,V,R,P},
                           y::Vec) where {S,V,R,P}
    check_dimensions(ŷ, y)
    check_pools(ŷ, y)
    return -log.(clamp.(broadcast(pdf, ŷ, y), P(c.tol), P(1) - P(c.tol)))
end

# -----------------------------------------------------
# BrierScore

struct BrierScore <: Measure end

metadata_measure(BrierScore;
                 human_name = "Brier score (a.k.a. quadratic score)",
                 instances                = ["brier_score",],
                 target_scitype           = Vec{<:Finite},
                 prediction_type          = :probabilistic,
                 orientation              = :score,
                 reports_each_observation = true,
                 is_feature_dependent     = false,
                 supports_weights         = true,
                 distribution_type        = UnivariateFinite)

@create_aliases BrierScore

@create_docs(BrierScore,
body=
"""
If `p(y)` is the predicted probability for a *single*
observation `y`, and `C` all possible classes, then the corresponding
Brier score for that observation is given by

``2p(y) - \\left(\\sum_{η ∈ C} p(η)^2\\right) - 1``

*Warning.* `BrierScore()` is a "score" in the sense that bigger is
better (with `0` optimal, and all other values negative). In Brier's
original 1950 paper, and many other places, it has the opposite sign,
despite the name. Moreover, the present implementation does not treat
the binary case as special, so that the score may differ, in that
case, by a factor of two from usage elsewhere.
""",
scientific_type=DOC_FINITE)

# calling on single observations (no checks):
function _brier_score(d::UnivariateFinite{S,V,R,P}, y) where {S,V,R,P}
    levels = classes(d)
    pvec = broadcast(pdf, d, levels)
    offset = P(1) + sum(pvec.^2)
    return P(2) * pdf(d, y) - offset
end

# calling on multiple observations:
function (::BrierScore)(ŷ::Vec{<:UnivariateFinite},
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

# Performant version in case of UnivariateFiniteArray:
function (::BrierScore)(
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

# -----------------------------------------------------
# BrierLoss

struct BrierLoss <: Measure end

metadata_measure(BrierLoss;
                 human_name = "Brier loss (a.k.a. quadratic loss)",
                 instances                = ["brier_loss",],
                 target_scitype           = Vec{<:Finite},
                 prediction_type          = :probabilistic,
                 orientation              = :score,
                 reports_each_observation = true,
                 is_feature_dependent     = false,
                 supports_weights         = true,
                 distribution_type        = UnivariateFinite)

@create_aliases BrierLoss

@create_docs(BrierLoss,
body=
"""
If `p(y)` is the predicted probability for a *single*
observation `y`, and `C` all possible classes, then the corresponding
Brier score for that observation is given by

``\\left(\\sum_{η ∈ C} p(η)^2\\right) - 2p(y) + 1``

*Warning.* In Brier's original 1950 paper, what is implemented here is
called a "loss". It is, however, a "score" in the contemporary use of
that term: smaller is better (with `0` optimal, and all other values
positive).  Note also the present implementation does not treat the
binary case as special, so that the loss may differ, in that case, by
a factor of two from usage elsewhere.
""",
scientific_type=DOC_FINITE)

# calling on single observations (no checks):
function _brier_loss(d::UnivariateFinite{S,V,R,P}, y) where {S,V,R,P}
    levels = classes(d)
    pvec = broadcast(pdf, d, levels)
    offset = P(1) + sum(pvec.^2)
    return P(2) * pdf(d, y) - offset
end

(m::BrierLoss)(ŷ::Vec{<:UnivariateFinite},
               y::Vec,
               w::Union{Nothing,Vec{<:Real}}=nothing) =
                   - brier_score(ŷ, y, w)




const INVARIANT_LABEL =
    "This metric is invariant to class reordering."
const VARIANT_LABEL =
    "This metric is *not* invariant to class re-ordering"

# =============================================================
# DETERMINISTIC FINITE PREDICTIONS

# ---------------------------------------------------
# misclassification rate

struct MisclassificationRate <: Measure end

metadata_measure(MisclassificationRate;
                 instances  = ["misclassification_rate", "mcr"],
                 target_scitype           = Vec{<:Finite},
                 prediction_type          = :deterministic,
                 orientation              = :loss,
                 reports_each_observation = false,
                 is_feature_dependent     = false,
                 supports_weights         = true)

const MCR = MisclassificationRate
@create_aliases MCR

@create_docs(MisclassificationRate,
body=
"""
A confusion matrix can also be passed as argument.
$INVARIANT_LABEL
""",
scientific_type=DOC_FINITE)

# calling behaviour:
(::MCR)(ŷ::Vec{<:CategoricalValue},
        y::Vec{<:CategoricalValue}) = mean(y .!= ŷ)
(::MCR)(ŷ::Vec{<:CategoricalValue},
        y::Vec{<:CategoricalValue},
        w::Vec{<:Real}) = sum((y .!= ŷ) .* w) / length(y)
(::MCR)(cm::ConfusionMatrixObject) = 1.0 - sum(diag(cm.mat)) / sum(cm.mat)

# -------------------------------------------------------------
# accuracy

struct Accuracy <: Measure end

metadata_measure(Accuracy;
                 instances = ["accuracy",],
                 target_scitype           = Vec{<:Finite},
                 prediction_type          = :deterministic,
                 orientation              = :score,
                 reports_each_observation = false,
                 is_feature_dependent     = false,
                 supports_weights         = true),

@create_aliases Accuracy

@create_docs(Accuracy,
body=
"""
Accuracy is proportion of correct predictions `ŷ[i]` that match the
ground truth `y[i]` observations. $INVARIANT_LABEL
""",
scientific_type=DOC_FINITE)

# calling behaviour:
(::Accuracy)(args...) = 1.0 - misclassification_rate(args...)
(::Accuracy)(m::ConfusionMatrixObject) = sum(diag(m.mat)) / sum(m.mat)

# -----------------------------------------------------------
# balanced accuracy

struct BalancedAccuracy <: Measure end

metadata_measure(BalancedAccuracy;
                 instances = ["balanced_accuracy", "bacc", "bac"],
                 target_scitype           = Vec{<:Finite},
                 prediction_type          = :deterministic,
                 orientation              = :score,
                 reports_each_observation = false,
                 is_feature_dependent     = false,
                 supports_weights         = true)

const BACC = BalancedAccuracy
@create_aliases BACC

@create_docs(BalancedAccuracy,
body=
"""
Balanced accuracy compensates standard [`Accuracy`](@ref) for class imbalance.
See [https://en.wikipedia.org/wiki/Precision_and_recall#Imbalanced_data](https://en.wikipedia.org/wiki/Precision_and_recall#Imbalanced_data).
$INVARIANT_LABEL
""",
scientific_type=DOC_FINITE)

# calling behavior:
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
## DETERMINISTIC BINARY PREDICTIONS - ORDER-INDEPENDENT

# ------------------------------------------------------------------
# Matthew's correlation

struct MatthewsCorrelation <: Measure end

metadata_measure(MatthewsCorrelation;
                 instances = ["matthews_correlation", "mcc"],
                 target_scitype           = Vec{<:Finite{2}},
                 prediction_type          = :deterministic,
                 orientation              = :score,
                 reports_each_observation = false,
                 is_feature_dependent     = false,
                 supports_weights         = false)
const MCC = MatthewsCorrelation
@create_aliases MCC

@create_docs(MatthewsCorrelation,
body=
"""
[https://en.wikipedia.org/wiki/Matthews_correlation_coefficient](https://en.wikipedia.org/wiki/Matthews_correlation_coefficient)
$INVARIANT_LABEL
""",
scientific_type=DOC_FINITE_BINARY)

# calling behaviour:
function (::MCC)(cm::ConfusionMatrixObject{C}) where C
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
             _confmat(ŷ, y, warn=false) |> m

# ---------------------------------------------------------
# AreaUnderCurve

#. Implementation drawn from
# https://www.ibm.com/developerworks/community/blogs/jfp/entry/Fast_Computation_of_AUC_ROC_score?lang=en
# but this link is now broken. Author contacted here:
# https://www.kaggle.com/c/microsoft-malware-prediction/discussion/76013.

struct AreaUnderCurve <: Measure end

metadata_measure(AreaUnderCurve;
                 human_name = "area under the ROC",
                 instances = ["area_under_curve", "auc"],
                 target_scitype           = Vec{<:Finite{2}},
                 prediction_type          = :probabilistic,
                 orientation              = :score,
                 reports_each_observation = false,
                 is_feature_dependent     = false,
                 supports_weights         = false)

const AUC = AreaUnderCurve
@create_aliases AreaUnderCurve

@create_docs(AreaUnderCurve,
body=
"""
Returns the area under the ROC ([receiver operator
characteristic](https://en.wikipedia.org/wiki/Receiver_operating_characteristic))
$INVARIANT_LABEL
""",
scitpye = DOC_FINITE_BINARY)

# core algorithm:
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

# calling behaviour:
(::AUC)(ŷ::Vec{<:UnivariateFinite}, y::Vec) = _auc(Float64, ŷ, y)

# performant version for UnivariateFiniteVector:
(::AUC)(ŷ::Vec{<:UnivariateFinite{S,V,R,P}}, y::Vec) where {S,V,R,P} =
    _auc(P, ŷ, y)


# ==========================================================================
# DETERMINISTIC BINARY PREDICTIONS - ORDER DEPENDENT

const CM2 = ConfusionMatrixObject{2}

# --------------------------------------------------------------------------
# FScore

struct FScore{T<:Real} <: Measure
    β::T
    rev::Union{Nothing,Bool}
end

FScore(; β=1.0, rev=nothing) = FScore(β, rev)

metadata_measure(FScore;
                 human_name = "F-Score",
                 instances = ["f1score",],
                 target_scitype           = Vec{<:Finite{2}},
                 prediction_type          = :deterministic,
                 orientation              = :score,
                 reports_each_observation = false,
                 is_feature_dependent     = false,
                 supports_weights         = false)

@create_aliases FScore

@create_docs(FScore,
body=
"""
This is the one-parameter generalization, ``F_β``, of the F-measure or
balanced F-score.

[https://en.wikipedia.org/wiki/F1_score](https://en.wikipedia.org/wiki/F1_score)

Constructor signature: `FScore(; β=1.0, rev=true)`.

By default, the second element of `levels(y)` is designated as
`true`. To reverse roles, specify `rev=true`.
""",
scientific_type=DOC_ORDERED_FACTOR_BINARY,
footer="Constructor signature: `FScore(β=1.0, rev=false)`. ")

# calling on conf matrix:
function (score::FScore)(m::CM2)
    β = score.β
    β2   = β^2
    prec = precision(m)
    rec  = recall(m)
    return (1 + β2) * (prec * rec) / (β2 * prec + rec)
end

# calling on vectors:
(m::FScore)(ŷ, y) = _confmat(ŷ, y; rev=m.rev) |> m

# -------------------------------------------------------------------------
# TruePositive and its cousins - struct and metadata declerations

const TRUE_POSITIVE_AND_COUSINS =
    (:TruePositive, :TrueNegative, :FalsePositive, :FalseNegative,
     :TruePositiveRate, :TrueNegativeRate, :FalsePositiveRate,
     :FalseNegativeRate, :FalseDiscoveryRate, :Precision,
     :NegativePredictiveValue)

for M in TRUE_POSITIVE_AND_COUSINS
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

metadata_measure.((TruePositiveRate, TrueNegativeRate, Precision,
                   NegativePredictiveValue);
    target_scitype           = Vec{<:Finite{2}},
    prediction_type          = :deterministic,
    orientation              = :score,
    reports_each_observation = false,
    is_feature_dependent     = false,
    supports_weights         = false)

# adjustments:
instances(::Type{<:TruePositive}) = ["true_positive", "truepositive"]
human_name(::Type{<:TruePositive})  = "number of true positives"

instances(::Type{<:TrueNegative}) = ["true_negative", "truenegative"]
human_name(::Type{<:TrueNegative}) = "number of true negatives"

instances(::Type{<:FalsePositive}) = ["false_positive", "falsepositive"]
human_name(::Type{<:FalsePositive}) = "number of false positives"

instances(::Type{<:FalseNegative}) = ["false_negative", "falsenegative"]
human_name(::Type{<:FalseNegative}) = "number of false negatives"

instances(::Type{<:TruePositiveRate}) =
    ["true_positive_rate", "truepositive_rate",
     "tpr", "sensitivity", "recall", "hit_rate"]
human_name(::Type{<:TruePositiveRate}) =
    "true positive rate (a.k.a recall)"

instances(::Type{<:TrueNegativeRate}) =
    ["true_negative_rate", "truenegative_rate", "tnr",
     "specificity", "selectivity"]

instances(::Type{<:FalsePositiveRate}) =
    ["false_positive_rate", "falsepositive_rate",
     "fpr", "fallout"]
                               "."
instances(::Type{<:FalseNegativeRate}) =
    ["false_negative_rate", "falsenegative_rate", "fnr", "miss_rate"]
                               "."
instances(::Type{<:FalseDiscoveryRate}) =
    ["false_discovery_rate", "falsediscovery_rate", "fdr"]

instances(::Type{<:NegativePredictiveValue}) =
    ["negative_predictive_value", "negativepredictive_value", "npv"]

instances(::Type{<:Precision}) =
    ["positive_predictive_value", "ppv", "positivepredictive_value", "precision"]
human_name(::Type{<:Precision}) =
    "precision (a.k.a. positive predictive value)"


# ---------------------------------------------------------------------
# TruePositive and its cousins - doc-string building and alias creation

for M in TRUE_POSITIVE_AND_COUSINS
    eval(quote
         $M == Precision || @create_aliases $M # precision handled separately

         @create_docs($M,
         body=
         """
         Assigns `false` to first element of `levels(y)`. To reverse roles,
         use `$(name($M))(rev=true)`.
         """,
         scientific_type=DOC_ORDERED_FACTOR_BINARY)
         end)
end

# type aliases:
const TNR = TrueNegativeRate
const Specificity = TrueNegativeRate
const TPR = TruePositiveRate
const Recall = TPR
const FPR = FalsePositiveRate
const FNR = FalseNegativeRate
const FDR = FalseDiscoveryRate
const NPV = NegativePredictiveValue
const PPV = Precision

# special case of precision; cannot generate alias's automatically due
# to conflict with Base.precision:
const positive_predictive_value = Precision()
const ppv = Precision()
const positivepredictive_value = Precision()

# ----------------------------------------------------------------------
# TruePositive and its cousins - helper functions for confusion matrices

_tp(m::CM2) = m[2,2]
_tn(m::CM2) = m[1,1]
_fp(m::CM2) = m[2,1]
_fn(m::CM2) = m[1,2]

_tpr(m::CM2) = _tp(m) / (_tp(m) + _fn(m))
_tnr(m::CM2) = _tn(m) / (_tn(m) + _fp(m))
_fpr(m::CM2) = 1 - _tnr(m)
_fnr(m::CM2) = 1 - _tpr(m)

_fdr(m::CM2) = _fp(m) / (_tp(m) + _fp(m))
_npv(m::CM2) = _tn(m) / (_tn(m) + _fn(m))

# ----------------------------------------------------------------------
# TruePositive and its cousins - calling behaviour

# NOTE: here we assume the CM was constructed a priori with the
# proper ordering so the field `rev` in the measure is ignored

# on confusion matrices:
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

# on vectors (ŷ, y):
for M_ex in TRUE_POSITIVE_AND_COUSINS
    local M = eval(M_ex)
    (m::M)(ŷ, y) = _confmat(ŷ, y; rev=m.rev) |> m
end

# special `precision` case (conflict with Base.precision):
Base.precision(m::CM2) = m |> Precision()
Base.precision(ŷ, y)   = _confmat(ŷ, y) |> Precision()


# =================================================================
#MULTICLASS AND ORDER INDEPENDENT

const CM = ConfusionMatrixObject{N} where N

abstract type MulticlassAvg end
struct MacroAvg <: MulticlassAvg end
struct MicroAvg <: MulticlassAvg end
struct NoAvg <: MulticlassAvg end

const macro_avg = MacroAvg()
const micro_avg = MicroAvg()
const no_avg    = NoAvg()

const DS_AVG_RET = "Options for `average` are: `no_avg`, `macro_avg` "*
    "(default) and `micro_avg`. Options for `return_type`, "*
    "applying in the `no_avg` case, are: `LittleDict` (default) or "*
    "`Vector`. "

const DS_RET = "Options for `return_type` are: "*
    "`LittleDict`(default) or "*
    "`Vector`. "

const CLASS_W = "An optional `AbstractDict`, denoted `class_w` above, "*
    "keyed on `levels(y)`, specifies class weights. It applies if "*
    "`average=macro_avg` or `average=no_avg`."

"""
    MulticlassFScore(; β=1.0, average=macro_avg, return_type=LittleDict)

One-parameter generalization, ``F_β``, of the F-measure or balanced F-score for
multiclass observations.

    MulticlassFScore()(ŷ, y)
    MulticlassFScore()(ŷ, y, class_w)

Evaluate the default score on multiclass observations, `ŷ`, given
ground truth values, `y`. $DS_AVG_RET $CLASS_W

For more information, run `info(MulticlassFScore)`.

"""
struct MulticlassFScore{T<:Real,
                        M<:MulticlassAvg,
                        U<:Union{Vector, LittleDict}} <:Measure
    β::T
    average::M
    return_type::Type{U}
end

MulticlassFScore(; β=1.0, average=macro_avg, return_type=LittleDict) =
    MulticlassFScore(β, average, return_type)

metadata_measure(MulticlassFScore;
    target_scitype           = Vec{<:Finite{N}} where N,
    prediction_type          = :deterministic,
    orientation              = :score,
    reports_each_observation = false,
    is_feature_dependent     = false,
    supports_weights         = false,
    supports_class_weights   = true)

MLJModelInterface.docstring(::Type{<:MulticlassFScore}) =
    "Multiclass F_β score; aliases: " *
    "`macro_f1score=MulticlassFScore()`, "*
    "`multiclass_f1score=MulticlassFScore()` " *
    "`micro_f1score=MulticlassFScore(average=micro_avg)`."

const micro_f1score      = MulticlassFScore(average=micro_avg)
const macro_f1score      = MulticlassFScore(average=macro_avg)
const multiclass_f1score = MulticlassFScore(average=macro_avg)

for M in (:MulticlassTruePositive, :MulticlassTrueNegative,
          :MulticlassFalsePositive, :MulticlassFalseNegative)
    ex = quote
        struct $M{U<:Union{Vector, LittleDict}} <: Measure
            return_type::Type{U}
        end
#        $M(return_type::Type{U}) where {U} = $M(return_type)
        $M(; return_type=LittleDict) = $M(return_type)
    end
    eval(ex)
end

const _mtp_vec = MulticlassTruePositive(return_type=Vector)
const _mfn_vec = MulticlassFalseNegative(return_type=Vector)
const _mfp_vec = MulticlassFalsePositive(return_type=Vector)
const _mtn_vec = MulticlassTrueNegative(return_type=Vector)

for M in (:MulticlassTruePositiveRate, :MulticlassTrueNegativeRate,
          :MulticlassFalsePositiveRate, :MulticlassFalseNegativeRate,
          :MulticlassFalseDiscoveryRate, :MulticlassPrecision,
          :MulticlassNegativePredictiveValue)
    ex = quote
        struct $M{T<:MulticlassAvg, U<:Union{Vector, LittleDict}} <: Measure
            average::T
            return_type::Type{U}
        end
        # @ablaom says next line looks redundant:
        $M(average::T, return_type::Type{U}) where {T, U} =
            $M(average, return_type)
        $M(; average=macro_avg,
           return_type=LittleDict) where T<:MulticlassAvg where
           U<:Union{Vector, LittleDict} = $M(average, return_type)
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

metadata_measure.((MulticlassTrueNegativeRate, MulticlassNegativePredictiveValue);
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

# MMI.name(::Type{<:MulticlassTruePositive})       = "multiclass_true_positive"
MMI.docstring(::Type{<:MulticlassTruePositive})  =
    "Number of true positives; " *
    "aliases: `multiclass_true_positive`, `multiclass_truepositive`."
# MMI.name(::Type{<:MulticlassTrueNegative})       = "multiclass_true_negative"
MMI.docstring(::Type{<:MulticlassTrueNegative})  =
    "Number of true negatives; " *
    "aliases: `multiclass_true_negative`, `multiclass_truenegative`."
# MMI.name(::Type{<:MulticlassFalsePositive})      = "multiclass_false_positive"
MMI.docstring(::Type{<:MulticlassFalsePositive}) =
    "Number of false positives; " *
    "aliases: `multiclass_false_positive`, `multiclass_falsepositive`."
# MMI.name(::Type{<:MulticlassFalseNegative})      = "multiclass_false_negative"
MMI.docstring(::Type{<:MulticlassFalseNegative}) =
    "Number of false negatives; " *
    "aliases: `multiclass_false_negative`, `multiclass_falsenegative`."

# MMI.name(::Type{<:MulticlassTruePositiveRate}) = "multiclass_true_positive_rate"
MMI.docstring(::Type{<:MulticlassTruePositiveRate}) =
    "multiclass true positive rate; aliases: " *
    "`multiclass_true_positive_rate`, `multiclass_tpr`, " *
    "`multiclass_sensitivity`, `multiclass_recall`, " *
    "`multiclass_hit_rate`, `multiclass_truepositive_rate`, "
# MMI.name(::Type{<:MulticlassTrueNegativeRate}) = "multiclass_true_negative_rate"
MMI.docstring(::Type{<:MulticlassTrueNegativeRate}) =
    "multiclass true negative rate; aliases: " *
    "`multiclass_true_negative_rate`, `multiclass_tnr` " *
    " `multiclass_specificity`, `multiclass_selectivity`, " *
    "`multiclass_truenegative_rate`."
# MMI.name(::Type{<:MulticlassFalsePositiveRate}) = "multiclass_false_positive_rate"
MMI.docstring(::Type{<:MulticlassFalsePositiveRate}) =
                       "multiclass false positive rate; aliases: " *
                       "`multiclass_false_positive_rate`, `multiclass_fpr` " *
                       "`multiclass_fallout`, `multiclass_falsepositive_rate`."
# MMI.name(::Type{<:MulticlassFalseNegativeRate}) = "multiclass_false_negative_rate"
MMI.docstring(::Type{<:MulticlassFalseNegativeRate}) =
    "multiclass false negative rate; aliases: " *
    "`multiclass_false_negative_rate`, `multiclass_fnr`, " *
    "`multiclass_miss_rate`, `multiclass_falsenegative_rate`."
# MMI.name(::Type{<:MulticlassFalseDiscoveryRate}) = "multiclass_false_discovery_rate"
MMI.docstring(::Type{<:MulticlassFalseDiscoveryRate}) =
    "multiclass false discovery rate; "*
    "aliases: `multiclass_false_discovery_rate`, " *
    "`multiclass_falsediscovery_rate`, `multiclass_fdr`."
# MMI.name(::Type{<:MulticlassNegativePredictiveValue}) = "multiclass_negative_predictive_value"
MMI.docstring(::Type{<:MulticlassNegativePredictiveValue}) =
    "multiclass negative predictive value; aliases: " *
    "`multiclass_negative_predictive_value`, " *
    "`multiclass_negativepredictive_value`, `multiclass_npv`."
# MMI.name(::Type{<:MulticlassPrecision}) = "multiclass_positive_predictive_value"
MMI.docstring(::Type{<:MulticlassPrecision}) =
  "multiclass positive predictive value (aka precision);"*
  " aliases: `multiclass_positive_predictive_value`, `multiclass_ppv`, " *
  "`MulticlassPrecision()`, `multiclass_positivepredictive_value`, " *
  "`multiclass_recall`."

const W_KEY_MISMATCH = "Encountered target with levels different from the " *
                       "keys of user-specified dictionary of class weights."
const W_PROMOTE_WARN = "Using macro averaging instead of micro averaging, as "*
    "class weights specified. "


# ----------------------------------------------------
# MulticlassTruePositive

"""
    MulticlassTruePositive(; return_type=LittleDict)

$(docstring(MulticlassTruePositive()))

    MulticlassTruePositive()(ŷ, y)

Number of true positives for multiclass observations `ŷ` and ground
truth `y`, using default return type. $DS_RET

For more information, run `info(MulticlassTruePositive)`.

"""
function MulticlassTruePositive end
const multiclass_true_positive  = MulticlassTruePositive()
const multiclass_truepositive   = MulticlassTruePositive()
const mtp = MulticlassTruePositive()


# ----------------------------------------------------
# MulticlassTrueNegative

"""
    MulticlassTrueNegative(; return_type=LittleDict)

$(docstring(MulticlassTrueNegative()))

    MulticlassTrueNegative()(ŷ, y)

Number of true negatives for multiclass observations `ŷ` and ground truth
`y`, using default return type. $DS_RET

For more information, run `info(MulticlassTrueNegative)`.

"""
function MulticlassTrueNegative end
const multiclass_true_negative  = MulticlassTrueNegative()
const multiclass_truenegative   = MulticlassTrueNegative()
const mtn = MulticlassTrueNegative()


# ----------------------------------------------------
# MulticlassFalsePositive

"""
    MulticlassFalsePositive(; return_type=LittleDict)

$(docstring(MulticlassFalsePositive()))

    MulticlassFalsePositive()(ŷ, y)

Number of false positives for multiclass observations `ŷ` and ground
truth `y`, using default return type. $DS_RET

For more information, run `info(MulticlassFalsePositive)`.

"""
function MulticlassPositive end
const multiclass_false_positive = MulticlassFalsePositive()
const multiclass_falsepositive  = MulticlassFalsePositive()
const mfp = MulticlassFalsePositive()


# ----------------------------------------------------
# MulticlassFalseNegative

"""
    MulticlassFalseNegative(; return_type=LittleDict)

$(docstring(MulticlassFalseNegative()))

    MulticlassFalseNegative()(ŷ, y)

Number of false negatives for multiclass observations `ŷ` and ground
truth `y`, using default return type. $DS_RET

For more information, run `info(MulticlassFalseNegative)`.

"""
function MulticlassNegative end
const multiclass_false_negative = MulticlassFalseNegative()
const multiclass_falsenegative  = MulticlassFalseNegative()
const mfn = MulticlassFalseNegative()


# ----------------------------------------------------
# MulticlassTruePositiveRate

"""
    MulticlassTruePositiveRate(; average=macro_avg, return_type=LittleDict)

$(docstring(MulticlassTruePositiveRate()))

    MulticlassTruePositiveRate(ŷ, y)
    MulticlassTruePositiveRate(ŷ, y, class_w)

True positive rate (a.k.a. sensitivity, recall, hit rate) for
multiclass observations `ŷ` and ground truth `y`, using default
averaging and return type. $DS_AVG_RET $CLASS_W

For more information, run `info(MulticlassTruePositiveRate)`.

"""
function MulticlassTruePositiveRate end
const multiclass_true_positive_rate = MulticlassTruePositiveRate()
const multiclass_truepositive_rate  = MulticlassTruePositiveRate()
const multiclass_tpr                = MulticlassTruePositiveRate()
const multiclass_sensitivity        = MulticlassTruePositiveRate()
const multiclass_hit_rate           = MulticlassTruePositiveRate()
const MTPR                          = MulticlassTruePositiveRate
const multiclass_recall             = MulticlassTruePositiveRate()
const MulticlassRecall              = MulticlassTruePositiveRate


# ----------------------------------------------------
# MulticlassTrueNegativeRate

"""
    MulticlassTrueNegativeRate(; average=macro_avg, return_type=LittleDict)

$(docstring(MulticlassTrueNegativeRate()))

    MulticlassTrueNegativeRate()(ŷ, y)
    MulticlassTrueNegativeRate()(ŷ, y, class_w)

True negative rate for multiclass observations `ŷ` and ground truth
`y`, using default averaging and return type. $DS_AVG_RET $CLASS_W

For more information, run `info(MulticlassTrueNegativeRate)`.

"""
function MulticlassTrueNegativeRate end
const multiclass_true_negative_rate = MulticlassTrueNegativeRate()
const multiclass_truenegative_rate  = MulticlassTrueNegativeRate()
const multiclass_tnr                = MulticlassTrueNegativeRate()
const multiclass_specificity        = MulticlassTrueNegativeRate()
const multiclass_selectivity        = MulticlassTrueNegativeRate()
const MulticlassSpecificity         = MulticlassTrueNegativeRate
const MTNR                          = MulticlassTrueNegativeRate


# ----------------------------------------------------
# MulticlassFalsePositiveRate

"""
    MulticlassFalsePositiveRate(; average=macro_avg, return_type=LittleDict)

$(docstring(MulticlassFalsePositiveRate()))

    MulticlassFalsePositiveRate()(ŷ, y)
    MulticlassFalsePositiveRate()(ŷ, y, class_w)

False positive rate for multiclass observations `ŷ` and ground truth
`y`, using default averaging and return type.  $DS_AVG_RET $CLASS_W

For more information, run `info(MulticlassFalsePositiveRate)`.

"""
function MulticlassFalsePositiveRate end
const multiclass_false_positive_rate = MulticlassFalsePositiveRate()
const multiclass_falsepositive_rate  = MulticlassFalsePositiveRate()
const multiclass_fpr                 = MulticlassFalsePositiveRate()
const MFPR                           = MulticlassFalsePositiveRate
const multiclass_fallout             = MFPR()


# ----------------------------------------------------
# MulticlassFalseNegativeRate

"""
    MulticlassFalseNegativeRate(; average=macro_avg, return_type=LittleDict)

$(docstring(MulticlassFalseNegativeRate()))

    MulticlassFalseNegativeRate()(ŷ, y)
    MulticlassFalseNegativeRate()(ŷ, y, class_w)

False negative rate for multiclass observations `ŷ` and ground truth
`y`, using default averaging and return type.  $DS_AVG_RET $CLASS_W

For more information, run `info(MulticlassFalseNegativeRate)`.

"""
function MulticlassFalseNegativeRate end
const multiclass_false_negative_rate = MulticlassFalseNegativeRate()
const multiclass_falsenegative_rate  = MulticlassFalseNegativeRate()
const multiclass_fnr                 = MulticlassFalseNegativeRate()
const MFNR                           = MulticlassFalseNegativeRate
const multiclass_miss_rate           = MFNR()


# ----------------------------------------------------
# MulticlassFalseDiscoveryRate

"""
    MulticlassFalseDiscoveryRate(; average=macro_avg, return_type=LittleDict)

$(docstring(MulticlassFalseDiscoveryRate()))

    MulticlassFalseDiscoveryRate()(ŷ, y)
    MulticlassFalseDiscoveryRate()(ŷ, y, class_w)

False discovery rate for multiclass observations `ŷ` and ground truth
`y`, using default averaging and return type.  $DS_AVG_RET $CLASS_W

For more information, run `info(MulticlassFalseDiscoveryRate)`.

"""
function MulticlassFalseDiscoveryRate end
const multiclass_false_discovery_rate = MulticlassFalseDiscoveryRate()
const multiclass_falsediscovery_rate  = MulticlassFalseDiscoveryRate()
const multiclass_fdr                  = MulticlassFalseDiscoveryRate()
const MFDR                            = MulticlassFalseDiscoveryRate


# ----------------------------------------------------
# MulticlassPrecision

"""
    MulticlassPrecision(; average=macro_avg, return_type=LittleDict)

$(docstring(MulticlassPrecision()))

    MulticlassPrecision()(ŷ, y)
    MulticlassPrecision()(ŷ, y, class_w)

Precision for multiclass observations `ŷ` and ground truth `y`, using
default averaging and return type. $DS_AVG_RET $CLASS_W

For more information, run `info(MulticlassPrecision)`.

"""
function MulticlassPrecision end
const multiclass_precision                 = MulticlassPrecision()
const multiclass_ppv                       = MulticlassPrecision()
const multiclass_positive_predictive_value = MulticlassPrecision()
const multiclass_positivepredictive_value  = MulticlassPrecision()
const MPPV                                 = MulticlassPrecision


# ----------------------------------------------------
# MulticlassNegativePredictiveValue

"""
    MulticlassNegativePredictiveValue(; average=macro_avg, return_type=LittleDict)

$(docstring(MulticlassNegativePredictiveValue()))

    MulticlassNegativePredictiveValue()(ŷ, y)
    MulticlassNegativePredictiveValue()(ŷ, y, class_w)

Negative predictive value for multiclass observations `ŷ` and ground truth
`y`, using default averaging and return type. $DS_AVG_RET $CLASS_W

For more information, run `info(MulticlassNegativePredictiveValue)`.

"""
function MulticlassNegativePredictiveValue end
const multiclass_npv                       = MulticlassNegativePredictiveValue()
const multiclass_negative_predictive_value = MulticlassNegativePredictiveValue()
const multiclass_negativepredictive_value  = MulticlassNegativePredictiveValue()
const MNPV                                 = MulticlassNegativePredictiveValue


# -----------------------------------------------------
## INTERNAL FUCNTIONS ON MULTICLASS CONFUSION MATRIX

_mtp(m::CM, return_type::Type{Vector}) = diag(m.mat)
_mtp(m::CM, return_type::Type{LittleDict}) =
    LittleDict(m.labels, diag(m.mat))

_mfp(m::CM, return_type::Type{Vector}) =
    (col_sum = vec(sum(m.mat, dims=2)); col_sum .-= diag(m.mat))

_mfp(m::CM, return_type::Type{LittleDict}) =
    (col_sum = vec(sum(m.mat, dims=2)); col_sum .-= diag(m.mat);
     LittleDict(m.labels, col_sum))

_mfn(m::CM, return_type::Type{Vector}) =
    (row_sum = vec(collect(transpose(sum(m.mat, dims=1))));
     row_sum .-= diag(m.mat))

_mfn(m::CM, return_type::Type{LittleDict}) =
    (row_sum = vec(collect(transpose(sum(m.mat, dims=1))));
     row_sum .-= diag(m.mat); LittleDict(m.labels, row_sum))

function _mtn(m::CM, return_type::Type{Vector})
    _sum  = sum(m.mat, dims=2)
    _sum .= sum(m.mat) .- (_sum .+= sum(m.mat, dims=1)'.+ diag(m.mat))
    return vec(_sum)
end

function _mtn(m::CM, return_type::Type{LittleDict})
    _sum  = sum(m.mat, dims=2)
    _sum .= sum(m.mat) .- (_sum .+= sum(m.mat, dims=1)'.+ diag(m.mat))
    return LittleDict(m.labels, vec(_sum))
end

@inline function _mean(x::Vec{<:Real})
    for i in eachindex(x)
        @inbounds x[i] = ifelse(isnan(x[i]), zero(eltype(x)), x[i])
    end
    return mean(x)
end

@inline function _class_w(level_m::Vec{<:String},
                          class_w::AbstractDict{<:Any, <:Real})
    class_w_labels = levels(keys(class_w))
    string.(class_w_labels) == level_m || throw(ArgumentError(W_KEY_MISMATCH))
    return [class_w[l] for l in class_w_labels]
end

@inline function _mc_helper(m::CM, a::Vec{<:Real}, b::Vec{<:Real},
                            average::NoAvg, return_type::Type{Vector})
    return vec(a ./ (a + b))
end

@inline function _mc_helper(m::CM, a::Vec{<:Real}, b::Vec{<:Real},
                            average::NoAvg, return_type::Type{LittleDict})
    return LittleDict(m.labels, _mc_helper(m, a, b, average, Vector))
end

@inline function _mc_helper(m::CM, a::Vec{<:Real}, b::Vec{<:Real},
                            average::MacroAvg, return_type)
    return _mean(_mc_helper(m, a, b, no_avg, Vector))
end

@inline function _mc_helper(m::CM, a::Vec{<:Real}, b::Vec{<:Real},
                            average::MicroAvg, return_type)
    a_sum, b_sum = sum(a), sum(b)
    return a_sum / (a_sum + b_sum)
end

@inline function _mc_helper(m::CM, a::Vec{<:Real}, b::Vec{<:Real},
                            class_w::AbstractDict{<:Any, <:Real},
                            average::NoAvg, return_type::Type{Vector})
    level_w = _class_w(m.labels, class_w)
    return _mc_helper(m, a, b, no_avg, return_type) .* level_w
end

@inline function _mc_helper(m::CM, a::Vec{<:Real}, b::Vec{<:Real},
                            class_w::AbstractDict{<:Any, <:Real},
                            average::MacroAvg, return_type::Type{Vector})
    return _mean(_mc_helper(m, a, b, class_w, no_avg, return_type))
end

@inline function _mc_helper(m::CM, a::Vec{<:Real}, b::Vec{<:Real},
                            class_w::AbstractDict{<:Any, <:Real},
                            average::MicroAvg, return_type)
    @warn W_PROMOTE_WARN
    return _mc_helper(m, a, b, class_w, macro_avg, Vector)
end

@inline function _mc_helper_b(m::CM, helper_name,
                              class_w::AbstractDict{<:Any, <:Real},
                              average::NoAvg, return_type::Type{Vector})
    level_w = _class_w(m.labels, class_w)
    return (1 .- helper_name(m, no_avg, return_type)) .* level_w
end

@inline function _mc_helper_b(m::CM, helper_name,
                              class_w::AbstractDict{<:Any, <:Real},
                              average::NoAvg, return_type::Type{LittleDict})
    level_w = _class_w(m.labels, class_w)
    return LittleDict(m.labels, ((1 .- helper_name(m, no_avg, Vector)) .* level_w))
end

@inline function _mc_helper_b(m::CM, helper_name,
                              class_w::AbstractDict{<:Any, <:Real},
                              average::MacroAvg, return_type)
    return _mean(_mc_helper_b(m, helper_name, class_w, no_avg, Vector))
end

@inline function _mc_helper_b(m::CM, helper_name,
                              class_w::AbstractDict{<:Any, <:Real},
                              average::MicroAvg, return_type)
    @warn W_PROMOTE_WARN
    return _mc_helper_b(m, helper_name, class_w, macro_avg, Vector)
end

@inline function _mc_helper_b(m::CM, helper_name, average::NoAvg,
                              return_type::Type{LittleDict})
    return LittleDict(m.labels, 1.0 .- helper_name(m, average, Vector))
end

@inline function _mc_helper_b(m::CM, helper_name, average::NoAvg,
                              return_type::Type{Vector})
    return 1.0 .- helper_name(m, average, Vector)
end

@inline function _mc_helper_b(m::CM, helper_name, average::MacroAvg,
                              return_type)
    return 1.0 .- helper_name(m, average, Vector)
end

@inline function _mc_helper_b(m::CM, helper_name, average::MicroAvg,
                              return_type)
    return 1.0 .- helper_name(m, average, Vector)
end

@inline function _mc_helper(m::CM, a::Vec{<:Real}, b::Vec{<:Real},
                            class_w::AbstractDict{<:Any, <:Real},
                            average::NoAvg, return_type::Type{LittleDict})
    level_w = _class_w(m.labels, class_w)
    return LittleDict(m.labels, _mc_helper(m, a, b, class_w, no_avg, Vector))
end

@inline function _mc_helper(m::CM, a::Vec{<:Real}, b::Vec{<:Real},
                            class_w::AbstractDict{<:Any, <:Real},
                            average::MacroAvg, return_type::Type{U}) where U
    return _mean(_mc_helper(m, a, b, class_w, no_avg, Vector))
end

@inline function _mc_helper(m::CM, a::Vec{<:Real}, b::Vec{<:Real},
                            class_w::AbstractDict{<:Any, <:Real},
                            average::MicroAvg, return_type::Type{U}) where U
    @warn W_PROMOTE_WARN
    return _mc_helper(m, a, b, class_w, macro_avg, return_type)
end

function _mtpr(m::CM, average::A, return_type::Type{U}) where {A, U}
    mtp_val, mfn_val = _mtp_vec(m), _mfn_vec(m)
    return _mc_helper(m, mtp_val, mfn_val, average, return_type)
end

function _mtpr(m::CM, class_w::AbstractDict{<:Any, <:Real}, average::A,
               return_type::Type{U}) where {A, U}
    mtp_val, mfn_val = _mtp_vec(m), _mfn_vec(m)
    return _mc_helper(m, mtp_val, mfn_val, class_w, average, return_type)
end

function _mtnr(m::CM, average::A, return_type::Type{U}) where {A, U}
    mtn_val, mfp_val = _mtn_vec(m), _mfp_vec(m)
    return _mc_helper(m, mtn_val, mfp_val, average, return_type)
end

function _mtnr(m::CM, class_w::AbstractDict{<:Any, <:Real}, average::A,
               return_type::Type{U}) where {A, U}
    mtn_val, mfp_val = _mtn_vec(m), _mfp_vec(m)
    return _mc_helper(m, mtn_val, mfp_val, class_w, average, return_type)
end

_mfpr(m::CM, average::A, return_type::Type{U}) where {A, U} =
    _mc_helper_b(m, _mtnr, average, return_type)

function _mfpr(m::CM, class_w::AbstractDict{<:Any, <:Real}, average::A,
               return_type::Type{U}) where {A, U}
    return _mc_helper_b(m, _mtnr, class_w, average, return_type)
end

_mfnr(m::CM, average::A, return_type::Type{U}) where {A, U} =
    _mc_helper_b(m, _mtpr, average, return_type)

function _mfnr(m::CM, class_w::AbstractDict{<:Any, <:Real}, average::A,
               return_type::Type{U}) where {A, U}
    return _mc_helper_b(m, _mtpr, class_w, average, return_type)
end

function _mfdr(m::CM, average::A, return_type::Type{U}) where {A, U}
    mfp_val, mtp_val = _mfp_vec(m), _mtp_vec(m)
    return _mc_helper(m, mfp_val, mtp_val, average, return_type)
end

function _mfdr(m::CM, class_w::AbstractDict{<:Any, <:Real}, average::A,
               return_type::Type{U}) where {A, U}
    mfp_val, mtp_val = _mfp_vec(m), _mtp_vec(m)
    return _mc_helper(m, mfp_val, mtp_val, class_w, average, return_type)
end

function _mnpv(m::CM, average::A, return_type::Type{U}) where {A, U}
    mtn_val, mfn_val = _mtn_vec(m), _mfn_vec(m)
    return _mc_helper(m, mtn_val, mfn_val, average, return_type)
end

function _mnpv(m::CM, class_w::AbstractDict{<:Any, <:Real}, average::A,
               return_type::Type{U}) where {A, U}
    mtn_val, mfn_val = _mtn_vec(m), _mfn_vec(m)
    return _mc_helper(m, mtn_val, mfn_val, class_w, average, return_type)
end

## CALLABLES ON MULTICLASS CONFUSION MATRIX

(p::MulticlassTruePositive)(m::CM)  = _mtp(m, p.return_type)
(n::MulticlassTrueNegative)(m::CM)  = _mtn(m, n.return_type)
(p::MulticlassFalsePositive)(m::CM) = _mfp(m, p.return_type)
(n::MulticlassFalseNegative)(m::CM) = _mfn(m, n.return_type)

(r::MTPR)(m::CM) = _mtpr(m, r.average, r.return_type)
(r::MTPR)(m::CM, w::AbstractDict{<:Any, <:Real}) =
    _mtpr(m, w, r.average, r.return_type)

(r::MTNR)(m::CM) = _mtnr(m, r.average, r.return_type)
(r::MTNR)(m::CM, w::AbstractDict{<:Any, <:Real}) =
    _mtnr(m, w, r.average, r.return_type)

(r::MFPR)(m::CM) = _mfpr(m, r.average, r.return_type)
(r::MFPR)(m::CM, w::AbstractDict{<:Any, <:Real}) =
    _mfpr(m, w, r.average, r.return_type)

(r::MFNR)(m::CM) = _mfnr(m, r.average, r.return_type)
(r::MFNR)(m::CM, w::AbstractDict{<:Any, <:Real}) =
    _mfnr(m, w, r.average, r.return_type)

(r::MFDR)(m::CM) = _mfdr(m, r.average, r.return_type)
(r::MFDR)(m::CM, w::AbstractDict{<:Any, <:Real}) =
    _mfdr(m, w, r.average, r.return_type)

(v::MNPV)(m::CM) = _mnpv(m, v.average, v.return_type)
(v::MNPV)(m::CM, w::AbstractDict{<:Any, <:Real}) =
    _mnpv(m, w, v.average, v.return_type)

(p::MulticlassPrecision)(m::CM) =
    _mc_helper_b(m, _mfdr, p.average, p.return_type)
function (p::MulticlassPrecision)(m::CM, class_w::AbstractDict{<:Any, <:Real})
    return _mc_helper_b(m, _mfdr, class_w, p.average, p.return_type)
end

@inline function _fs_helper(m::CM, β::Real, rec::Vec{<:Real}, prec::Vec{<:Real},
                    average::NoAvg, return_type::Type{LittleDict})
    β2 = β^2
    return LittleDict(m.labels, (1 + β2) .* (prec .* rec) ./ (β2 .* prec .+ rec))
end

@inline function _fs_helper(m::CM, β::Real, rec::Vec{<:Real}, prec::Vec{<:Real},
                    average::NoAvg, return_type::Type{Vector})
    β2 = β^2
    return (1 + β2) .* (prec .* rec) ./ (β2 .* prec .+ rec)
end

@inline function _fs_helper(m::CM, β::Real, rec::Vec{<:Real}, prec::Vec{<:Real},
                    average::MacroAvg, return_type::Type{U}) where U
    return _mean(_fs_helper(m, β, rec, prec, no_avg, Vector))
end

@inline function _fs_helper(m::CM, β::Real, rec::Real, prec::Real,
                            average::MicroAvg, return_type::Type{U}) where U
    β2 = β^2
    return (1 + β2) * (prec * rec) / (β2 * prec + rec)
end

function (f::MulticlassFScore)(m::CM)
    if f.average == micro_avg
        rec = MulticlassRecall(; average=f.average, return_type=Vector)(m)
        f.β == 1.0 && return rec
        return _fs_helper(m, f.β, rec, rec, f.average, f.return_type)
    end
    rec = MulticlassRecall(; average=no_avg, return_type=Vector)(m)
    prec = MulticlassPrecision(; average=no_avg, return_type=Vector)(m)
    return _fs_helper(m, f.β, rec, prec, f.average, f.return_type)
end

@inline function _fs_helper(m::CM, w::AbstractDict{<:Any, <:Real}, β::Real,
                    average::NoAvg, return_type::Type{LittleDict})
    level_w = _class_w(m.labels, w)
    return LittleDict(m.labels,
                      MulticlassFScore(β=β,
                                       average=no_avg,
                                       return_type=Vector)(m) .* level_w)
end

@inline function _fs_helper(m::CM, w::AbstractDict{<:Any, <:Real}, β::Real,
                    average::NoAvg, return_type::Type{Vector})
    level_w = _class_w(m.labels, w)
    return MulticlassFScore(β=β,
                            average=no_avg,
                            return_type=Vector)(m) .* level_w
end

@inline function _fs_helper(m::CM, w::AbstractDict{<:Any, <:Real}, β::Real,
                            average::MacroAvg, return_type::Type{U}) where U
    return _mean(_fs_helper(m, w, β, no_avg, Vector))
end

@inline function _fs_helper(m::CM, w::AbstractDict{<:Any, <:Real}, β::Real,
                            average::MicroAvg, return_type::Type{U}) where U
    @warn W_PROMOTE_WARN
    return _fs_helper(m, w, β, macro_avg, return_type)
end

function (f::MulticlassFScore)(m::CM, class_w::AbstractDict{<:Any, <:Real})
    return _fs_helper(m, class_w, f.β, f.average, f.return_type)
end

## Callables on vectors

for M in (MulticlassTruePositive, MulticlassTrueNegative,
          MulticlassFalsePositive, MulticlassFalseNegative)
    (m::M)(ŷ, y) = m(_confmat(ŷ, y, warn=false))
end

for M in (MTPR, MTNR, MFPR, MFNR, MFDR, MulticlassPrecision, MNPV,
          MulticlassFScore)
    @eval (m::$M)(ŷ, y) = m(_confmat(ŷ, y, warn=false))
    @eval (m::$M)(ŷ, y, class_w::AbstractDict{<:Any, <:Real}) =
                          m(_confmat(ŷ, y, warn=false), class_w)
end


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
