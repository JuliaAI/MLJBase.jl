const FiniteArrMissing{N} = Union{
    AbstractArray{<:Union{Missing,Multiclass{N}}},
    AbstractArray{<:Union{Missing,OrderedFactor{N}}}}

# ---------------------------------------------------
# misclassification rate

struct MisclassificationRate <: Aggregated end

metadata_measure(MisclassificationRate;
                 instances  = ["misclassification_rate", "mcr"],
                 target_scitype           = FiniteArrMissing,
                 prediction_type          = :deterministic,
                 orientation              = :loss)

const MCR = MisclassificationRate
@create_aliases MCR

@create_docs(MisclassificationRate,
body=
"""
A confusion matrix can also be passed as argument.
$INVARIANT_LABEL
""",
scitype=DOC_FINITE)

# calling behaviour:
call(::MCR, ŷ, y) where {V,N} = (y .!= ŷ) |> Mean()
call(::MCR, ŷ, y, w) where {V,N} = (y .!= ŷ) .* w |> Mean()
(::MCR)(cm::ConfusionMatrixObject) = 1.0 - sum(diag(cm.mat)) / sum(cm.mat)

# -------------------------------------------------------------
# accuracy

struct Accuracy <: Aggregated end

metadata_measure(Accuracy;
                 instances = ["accuracy",],
                 target_scitype           = FiniteArrMissing,
                 prediction_type          = :deterministic,
                 orientation              = :score)

@create_aliases Accuracy

@create_docs(Accuracy,
body=
"""
Accuracy is proportion of correct predictions `ŷ[i]` that match the
ground truth `y[i]` observations. $INVARIANT_LABEL
""",
scitype=DOC_FINITE)

# calling behaviour:
call(::Accuracy, args...) = 1.0 - call(misclassification_rate, args...)
(::Accuracy)(m::ConfusionMatrixObject) = sum(diag(m.mat)) / sum(m.mat)

# -----------------------------------------------------------
# balanced accuracy

struct BalancedAccuracy <: Aggregated
    adjusted::Bool
end
BalancedAccuracy(; adjusted=false) = BalancedAccuracy(adjusted)

metadata_measure(BalancedAccuracy;
                 instances = ["balanced_accuracy", "bacc", "bac"],
                 target_scitype           = FiniteArrMissing,
                 prediction_type          = :deterministic,
                 orientation              = :score)

const BACC = BalancedAccuracy
@create_aliases BACC

@create_docs(BalancedAccuracy,
body=
"""
Balanced accuracy compensates standard [`Accuracy`](@ref) for class imbalance.
See [https://en.wikipedia.org/wiki/Precision_and_recall#Imbalanced_data](https://en.wikipedia.org/wiki/Precision_and_recall#Imbalanced_data). 

Setting `adjusted=true` rescales the score in the way prescribed in
[L. Mosley (2013): A balanced approach to the multi-class imbalance
problem. PhD thesis, Iowa State
University](https://lib.dr.iastate.edu/etd/13537/). In the binary
case, the adjusted balanced accuracy is also known as *Youden’s J
statistic*, or *informedness*.

$INVARIANT_LABEL
""",
scitype=DOC_FINITE)

function call(m::BACC, ŷm, ym, wm::Union{Nothing,Arr{<:Real}}=nothing)

    ŷ, y, w = _skipinvalid(ŷm, ym, wm)

    if w === nothing
        n_given_class = StatsBase.countmap(y)
        freq(i) = @inbounds n_given_class[y[i]]
        ŵ = 1 ./ freq.(eachindex(y))
    else # following sklearn, which is non-linear
        ŵ = similar(w)
        @inbounds for i in eachindex(w)
            ŵ[i] = w[i] / sum(w .* (y .== y[i]))
        end
    end
    s = sum(ŵ)
    score = sum((ŷ .== y) .* ŵ) / sum(ŵ)
    if m.adjusted
        n_classes = length(levels(y))
        chance = 1 / n_classes
        score -= chance
        score /= 1 - chance
    end
    return score
end

# ---------------------------------------------------
# kappa

struct Kappa <: Aggregated end

metadata_measure(Kappa;
                 instances  = ["kappa"],
                 target_scitype           = FiniteArrMissing,
                 prediction_type          = :deterministic,
                 orientation              = :score,
                 supports_weights         = false)

@create_aliases Kappa

@create_docs(Kappa,
body=
"""
A metric to measure agreement between predicted labels and the ground truth. 
See [https://en.wikipedia.org/wiki/Cohen%27s_kappa](https://en.wikipedia.org/wiki/Cohen%27s_kappa)

$INVARIANT_LABEL
""",
scitype=DOC_FINITE)

# calling behaviour:
function (::Kappa)(cm::ConfusionMatrixObject{C}) where C
    # relative observed agreement - same as accuracy
    p₀ = sum(diag(cm.mat))/sum(cm.mat)

    # probability of agreement due to chance - for each class cᵢ, this would be: (#predicted=cᵢ)/(#instances) x (#observed=cᵢ)/(#instances)
    pₑ = sum(sum(cm[j, :]) * sum(cm[:, j]) for j in 1:C)/sum(cm.mat)^2

    # Kappa calculation
    κ = (p₀ - pₑ)/(1 - pₑ)

    return κ
end

call(k::Kappa, ŷ, y) = _confmat(ŷ, y, warn=false) |> k


# ==================================================================
## DETERMINISTIC BINARY PREDICTIONS - ORDER-INDEPENDENT

# ------------------------------------------------------------------
# Matthew's correlation

struct MatthewsCorrelation <: Aggregated end

metadata_measure(MatthewsCorrelation;
                 instances = ["matthews_correlation", "mcc"],
                 target_scitype           = FiniteArrMissing{2},
                 prediction_type          = :deterministic,
                 orientation              = :score,
                 supports_weights         = false)
const MCC = MatthewsCorrelation
@create_aliases MCC

@create_docs(MatthewsCorrelation,
body=
"""
[https://en.wikipedia.org/wiki/Matthews_correlation_coefficient](https://en.wikipedia.org/wiki/Matthews_correlation_coefficient)
$INVARIANT_LABEL
""",
scitype=DOC_FINITE_BINARY)

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
    mcc = num / sqrt(float(den1) * float(den2))

    isnan(mcc) && return 0
    return mcc
end

call(m::MCC, ŷ, y) = _confmat(ŷ, y, warn=false) |> m


# ==========================================================================
# DETERMINISTIC BINARY PREDICTIONS - ORDER DEPENDENT

const CM2 = ConfusionMatrixObject{2}

# --------------------------------------------------------------------------
# FScore

struct FScore{T<:Real} <: Aggregated
    β::T
    rev::Union{Nothing,Bool}
end

FScore(; β=1.0, rev=nothing) = FScore(β, rev)

metadata_measure(FScore;
                 human_name = "F-Score",
                 instances = ["f1score",],
                 target_scitype           = FiniteArrMissing{2},
                 prediction_type          = :deterministic,
                 orientation              = :score,
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
scitype=DOC_ORDERED_FACTOR_BINARY,
footer="Constructor signature: `FScore(β=1.0, rev=false)`. ")

# calling on conf matrix:
function (score::FScore)(m::CM2)
    β = score.β
    β2   = β^2
    tp = _tp(m)
    fn = _fn(m)
    fp = _fp(m)
    return (1 + β2) * tp / ((1 + β2)*tp + β2*fn + fp)
end

# calling on arrays:
call(m::FScore, ŷ, y) = _confmat(ŷ, y; rev=m.rev) |> m

# -------------------------------------------------------------------------
# TruePositive and its cousins - struct and metadata declerations

const TRUE_POSITIVE_AND_COUSINS =
    (:TruePositive, :TrueNegative, :FalsePositive, :FalseNegative,
     :TruePositiveRate, :TrueNegativeRate, :FalsePositiveRate,
     :FalseNegativeRate, :FalseDiscoveryRate, :Precision,
     :NegativePredictiveValue)

for M in TRUE_POSITIVE_AND_COUSINS
    ex = quote
        struct $M <: Aggregated rev::Union{Nothing,Bool} end
        $M(; rev=nothing) = $M(rev)
    end
    eval(ex)
end

metadata_measure.((FalsePositive, FalseNegative);
    target_scitype           = FiniteArrMissing{2},
    prediction_type          = :deterministic,
    orientation              = :loss,
    aggregation              = Sum(),
    supports_weights         = false)

metadata_measure.((FalsePositiveRate, FalseNegativeRate, FalseDiscoveryRate);
    target_scitype           = FiniteArrMissing{2},
    prediction_type          = :deterministic,
    orientation              = :loss,
    supports_weights         = false)

metadata_measure.((TruePositive, TrueNegative);
    target_scitype           = FiniteArrMissing{2},
    prediction_type          = :deterministic,
    orientation              = :score,
    aggregation              = Sum(),
    supports_weights         = false)

metadata_measure.((TruePositiveRate, TrueNegativeRate, Precision,
                   NegativePredictiveValue);
    target_scitype           = FiniteArrMissing{2},
    prediction_type          = :deterministic,
    orientation              = :score,
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
         scitype=DOC_ORDERED_FACTOR_BINARY)
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

# on arrays (ŷ, y):
for M_ex in TRUE_POSITIVE_AND_COUSINS
    @eval call(m::$M_ex, ŷ, y) = _confmat(ŷ, y; rev=m.rev) |> m
end

# since Base.precision exists (as single argument function) we
# manually overload Base.precision:
Base.precision(m::CM2) = m |> Precision()
function Base.precision(ŷ, y)
    _check(Precision(), ŷ, y)
    call(Precision(), ŷ, y)
end


# =================================================================
# MULTICLASS AND ORDER INDEPENDENT

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
                        U<:Union{Vector, LittleDict}} <:Aggregated
    β::T
    average::M
    return_type::Type{U}
end

MulticlassFScore(; β=1.0, average=macro_avg, return_type=LittleDict) =
    MulticlassFScore(β, average, return_type)

metadata_measure(MulticlassFScore;
                 instances = ["macro_f1score", "micro_f1score",
                              "multiclass_f1score"],
                 target_scitype           = FiniteArrMissing,
                 prediction_type          = :deterministic,
                 orientation              = :score,
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
        struct $M{U<:Union{Vector, LittleDict}} <: Aggregated
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
        struct $M{T<:MulticlassAvg, U<:Union{Vector, LittleDict}} <: Aggregated
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
    target_scitype           = FiniteArrMissing,
    prediction_type          = :deterministic,
    orientation              = :loss,
    aggregation               = Sum(),
    is_feature_dependent     = false,
    supports_weights         = false,
    supports_class_weights   = false)

metadata_measure.((MulticlassFalsePositiveRate, MulticlassFalseNegativeRate,
                   MulticlassFalseDiscoveryRate);
    target_scitype           = FiniteArrMissing,
    prediction_type          = :deterministic,
    orientation              = :loss,
    is_feature_dependent     = false,
    supports_weights         = false,
    supports_class_weights   = true)

metadata_measure.((MulticlassTruePositive, MulticlassTrueNegative);
    target_scitype           = FiniteArrMissing,
    prediction_type          = :deterministic,
    orientation              = :score,
    aggregation              = Sum(),
    is_feature_dependent     = false,
    supports_weights         = false,
    supports_class_weights   = false)

metadata_measure.((MulticlassTrueNegativeRate, MulticlassNegativePredictiveValue);
    target_scitype           = FiniteArrMissing,
    prediction_type          = :deterministic,
    orientation              = :score,
    is_feature_dependent     = false,
    supports_weights         = false,
    supports_class_weights   = true)

metadata_measure.((MulticlassTruePositiveRate, MulticlassPrecision);
    target_scitype           = FiniteArrMissing,
    prediction_type          = :deterministic,
    orientation              = :score,
    is_feature_dependent     = false,
    supports_weights         = false,
    supports_class_weights   = true)

MMI.docstring(::Type{<:MulticlassTruePositive})  =
    "Number of true positives; " *
    "aliases: `multiclass_true_positive`, `multiclass_truepositive`."
instances(::Type{<:MulticlassTruePositive})  =
    ["multiclass_true_positive", "multiclass_truepositive"]
MMI.docstring(::Type{<:MulticlassTrueNegative})  =
    "Number of true negatives; " *
    "aliases: `multiclass_true_negative`, `multiclass_truenegative`."
instances(::Type{<:MulticlassTrueNegative})  =
    ["multiclass_true_negative", "multiclass_truenegative"]
MMI.docstring(::Type{<:MulticlassFalsePositive}) =
    "Number of false positives; " *
    "aliases: `multiclass_false_positive`, `multiclass_falsepositive`."
instances(::Type{<:MulticlassFalsePositive}) =
    ["multiclass_false_positive", "multiclass_falsepositive"]
MMI.docstring(::Type{<:MulticlassFalseNegative}) =
    "Number of false negatives; " *
    "aliases: `multiclass_false_negative`, `multiclass_falsenegative`."
instances(::Type{<:MulticlassFalseNegative}) =
    ["multiclass_false_negative", "multiclass_falsenegative"]

MMI.docstring(::Type{<:MulticlassTruePositiveRate}) =
    "multiclass true positive rate; aliases: " *
    "`multiclass_true_positive_rate`, `multiclass_tpr`, " *
    "`multiclass_sensitivity`, `multiclass_recall`, " *
    "`multiclass_hit_rate`, `multiclass_truepositive_rate`, "
instances(::Type{<:MulticlassTruePositiveRate}) =
    ["multiclass_true_positive_rate", "multiclass_tpr",
    "multiclass_sensitivity", "multiclass_recall",
    "multiclass_hit_rate", "multiclass_truepositive_rate"]
MMI.docstring(::Type{<:MulticlassTrueNegativeRate}) =
    "multiclass true negative rate; aliases: " *
    "`multiclass_true_negative_rate`, `multiclass_tnr` " *
    " `multiclass_specificity`, `multiclass_selectivity`, " *
    "`multiclass_truenegative_rate`."
instances(::Type{<:MulticlassTrueNegativeRate}) =
    ["multiclass_true_negative_rate", "multiclass_tnr",
    "multiclass_specificity", "multiclass_selectivity",
    "multiclass_truenegative_rate"]
MMI.docstring(::Type{<:MulticlassFalsePositiveRate}) =
                       "multiclass false positive rate; aliases: " *
                       "`multiclass_false_positive_rate`, `multiclass_fpr` " *
                       "`multiclass_fallout`, `multiclass_falsepositive_rate`."
instances(::Type{<:MulticlassFalsePositiveRate}) =
    ["multiclass_false_positive_rate", "multiclass_fpr",
     "multiclass_fallout", "multiclass_falsepositive_rate"]
MMI.docstring(::Type{<:MulticlassFalseNegativeRate}) =
    "multiclass false negative rate; aliases: " *
    "`multiclass_false_negative_rate`, `multiclass_fnr`, " *
    "`multiclass_miss_rate`, `multiclass_falsenegative_rate`."
instances(::Type{<:MulticlassFalseNegativeRate}) =
    ["multiclass_false_negative_rate", "multiclass_fnr",
    "multiclass_miss_rate", "multiclass_falsenegative_rate"]
MMI.docstring(::Type{<:MulticlassFalseDiscoveryRate}) =
    "multiclass false discovery rate; "*
    "aliases: `multiclass_false_discovery_rate`, " *
    "`multiclass_falsediscovery_rate`, `multiclass_fdr`."
instances(::Type{<:MulticlassFalseDiscoveryRate}) =
    ["multiclass_falsediscovery_rate", "multiclass_fdr",
     "multiclass_false_discovery_rate"]
MMI.docstring(::Type{<:MulticlassNegativePredictiveValue}) =
    "multiclass negative predictive value; aliases: " *
    "`multiclass_negative_predictive_value`, " *
    "`multiclass_negativepredictive_value`, `multiclass_npv`."
instances(::Type{<:MulticlassNegativePredictiveValue}) =
    ["multiclass_negative_predictive_value",
    "multiclass_negativepredictive_value", "multiclass_npv"]
MMI.docstring(::Type{<:MulticlassPrecision}) =
  "multiclass positive predictive value (aka precision);"*
  " aliases: `multiclass_positive_predictive_value`, `multiclass_ppv`, " *
  "`multiclass_positivepredictive_value`, " *
  "`multiclass_precision`."
instances(::Type{<:MulticlassPrecision}) =
    ["multiclass_positive_predictive_value", "multiclass_ppv",
     "multiclass_positivepredictive_value", "multiclass_precision"]

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
## INTERNAL FUNCTIONS ON MULTICLASS CONFUSION MATRIX

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
    _sum = sum(m.mat, dims=2)
    _sum .= sum(m.mat) .- (_sum .+= sum(m.mat, dims=1)'.- diag(m.mat))
    return vec(_sum)
end

function _mtn(m::CM, return_type::Type{LittleDict})
    _sum = sum(m.mat, dims=2)
    _sum .= sum(m.mat) .- (_sum .+= sum(m.mat, dims=1)'.- diag(m.mat))
    return LittleDict(m.labels, vec(_sum))
end

@inline function _mean(x::Arr{<:Real})
    for i in eachindex(x)
        @inbounds x[i] = ifelse(isnan(x[i]), zero(eltype(x)), x[i])
    end
    return mean(x)
end

@inline function _class_w(level_m::Arr{<:String},
                          class_w::AbstractDict{<:Any, <:Real})
    class_w_labels = levels(keys(class_w))
    string.(class_w_labels) == level_m || throw(ArgumentError(W_KEY_MISMATCH))
    return [class_w[l] for l in class_w_labels]
end

@inline function _mc_helper(m::CM, a::Arr{<:Real}, b::Arr{<:Real},
                            average::NoAvg, return_type::Type{Vector})
    return vec(a ./ (a + b))
end

@inline function _mc_helper(m::CM, a::Arr{<:Real}, b::Arr{<:Real},
                            average::NoAvg, return_type::Type{LittleDict})
    return LittleDict(m.labels, _mc_helper(m, a, b, average, Vector))
end

@inline function _mc_helper(m::CM, a::Arr{<:Real}, b::Arr{<:Real},
                            average::MacroAvg, return_type)
    return _mean(_mc_helper(m, a, b, no_avg, Vector))
end

@inline function _mc_helper(m::CM, a::Arr{<:Real}, b::Arr{<:Real},
                            average::MicroAvg, return_type)
    a_sum, b_sum = sum(a), sum(b)
    return a_sum / (a_sum + b_sum)
end

@inline function _mc_helper(m::CM, a::Arr{<:Real}, b::Arr{<:Real},
                            class_w::AbstractDict{<:Any, <:Real},
                            average::NoAvg, return_type::Type{Vector})
    level_w = _class_w(m.labels, class_w)
    return _mc_helper(m, a, b, no_avg, return_type) .* level_w
end

@inline function _mc_helper(m::CM, a::Arr{<:Real}, b::Arr{<:Real},
                            class_w::AbstractDict{<:Any, <:Real},
                            average::MacroAvg, return_type::Type{Vector})
    return _mean(_mc_helper(m, a, b, class_w, no_avg, return_type))
end

@inline function _mc_helper(m::CM, a::Arr{<:Real}, b::Arr{<:Real},
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

@inline function _mc_helper(m::CM, a::Arr{<:Real}, b::Arr{<:Real},
                            class_w::AbstractDict{<:Any, <:Real},
                            average::NoAvg, return_type::Type{LittleDict})
    level_w = _class_w(m.labels, class_w)
    return LittleDict(m.labels, _mc_helper(m, a, b, class_w, no_avg, Vector))
end

@inline function _mc_helper(m::CM, a::Arr{<:Real}, b::Arr{<:Real},
                            class_w::AbstractDict{<:Any, <:Real},
                            average::MacroAvg, return_type::Type{U}) where U
    return _mean(_mc_helper(m, a, b, class_w, no_avg, Vector))
end

@inline function _mc_helper(m::CM, a::Arr{<:Real}, b::Arr{<:Real},
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
(p::MulticlassPrecision)(m::CM, class_w::AbstractDict{<:Any, <:Real}) =
    _mc_helper_b(m, _mfdr, class_w, p.average, p.return_type)

@inline function _fs_helper(m::CM, β::Real, mtp_val::Arr{<:Real}, mfp_val::Arr{<:Real}, mfn_val::Arr{<:Real},
                    average::NoAvg, return_type::Type{LittleDict})
    β2 = β^2
    return LittleDict(m.labels, (1 + β2) * mtp_val ./ ((1 + β2) * mtp_val + β2 * mfn_val + mfp_val))
end

@inline function _fs_helper(m::CM, β::Real, mtp_val::Arr{<:Real}, mfp_val::Arr{<:Real}, mfn_val::Arr{<:Real},
                    average::NoAvg, return_type::Type{Vector})
    β2 = β^2
    return (1 + β2) * mtp_val ./ ((1 + β2) * mtp_val + β2 * mfn_val + mfp_val)
end

@inline function _fs_helper(m::CM, β::Real, mtp_val::Arr{<:Real}, mfp_val::Arr{<:Real}, mfn_val::Arr{<:Real},
                            average::MacroAvg, return_type::Type{U}) where U
    return _mean(_fs_helper(m, β, mtp_val, mfp_val, mfn_val, no_avg, Vector))
end

function (f::MulticlassFScore)(m::CM)
    f.average == micro_avg && return MulticlassRecall(; average=micro_avg, return_type=f.return_type)(m)
    mtp_val = _mtp(m, Vector)
    mfp_val = _mfp(m, Vector)
    mfn_val = _mfn(m, Vector)
    return _fs_helper(m, f.β, mtp_val, mfp_val, mfn_val, f.average, f.return_type)
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

## Callables on arrays

for M_ex in (:MulticlassTruePositive, :MulticlassTrueNegative,
          :MulticlassFalsePositive, :MulticlassFalseNegative)
    @eval call(m::$M_ex, ŷ, y) = m(_confmat(ŷ, y, warn=false))
end

for M_ex in (:MTPR, :MTNR, :MFPR, :MFNR, :MFDR, :MulticlassPrecision, :MNPV,
          :MulticlassFScore)
    @eval call(m::$M_ex, ŷ, y) = m(_confmat(ŷ, y, warn=false))
    @eval call(m::$M_ex, ŷ, y, class_w::AbstractDict{<:Any, <:Real}) =
        m(_confmat(ŷ, y, warn=false), class_w)
end
