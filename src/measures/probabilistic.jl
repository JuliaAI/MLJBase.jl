const DOC_DISTRIBUTIONS =
"""
In the case the predictions `ŷ` are continuous probability
distributions, such as `Distributions.Normal`, replace the above sum
with an integral, and interpret `p` as the probablity density
function. In case of discrete distributions over the integers, such as
`Distributions.Poisson`, sum over all integers instead of `C`.
"""
const UD = Distributions.UnivariateDistribution
const WITH_L2NORM_CONTINUOUS =
    [@eval(Distributions.$d) for d in [
        :Chisq,
        :Gamma,
        :Beta,
        :Chi,
        :Cauchy,
        :Normal,
        :Uniform,
        :Logistic,
        :Exponential]]

const WITH_L2NORM_COUNT =
    [@eval(Distributions.$d) for d in [
        :Poisson,
        :DiscreteUniform,
        :DiscreteNonParametric]]

const WITH_L2NORM_INFINITE = vcat(WITH_L2NORM_CONTINUOUS,
                                  WITH_L2NORM_COUNT)

# ========================================================
# AGGREGATED MEASURES

# ---------------------------------------------------------
# AreaUnderCurve

#. Implementation drawn from
# https://www.ibm.com/developerworks/community/blogs/jfp/entry/Fast_Computation_of_AUC_ROC_score?lang=en
# but this link is now broken. Author contacted here:
# https://www.kaggle.com/c/microsoft-malware-prediction/discussion/76013.

struct AreaUnderCurve <: Aggregated end

metadata_measure(AreaUnderCurve;
                 human_name = "area under the ROC",
                 instances = ["area_under_curve", "auc"],
                 target_scitype           = Arr{<:Union{Missing,Finite{2}}},
                 prediction_type          = :probabilistic,
                 orientation              = :score,
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
function _auc(::Type{P}, ŷm, ym) where P<:Real # type of probabilities
    ŷ, y    = skipinvalid(ŷm, ym)
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
call(::AUC, ŷ::Arr{<:UnivariateFinite}, y) = _auc(Float64, ŷ, y)

# performant version for UnivariateFiniteVector:
call(::AUC, ŷ::Arr{<:UnivariateFinite{S,V,R,P}}, y) where {S,V,R,P} =
    _auc(P, ŷ, y)


# ========================================================
# UNAGGREGATED MEASURES

# ---------------------------------------------------------------------
# LogScore

struct LogScore{R <: Real} <: Unaggregated
    tol::R
end
LogScore(;eps=eps(), tol=eps) = LogScore(tol)

metadata_measure(LogScore;
                 instances                = ["log_score", ],
                 target_scitype           = Arr{<:Union{Missing,Finite}},
                 prediction_type          = :probabilistic,
                 orientation              = :loss,
                 is_feature_dependent     = false,
                 supports_weights         = false,
                 distribution_type        = UnivariateFinite)

@create_aliases LogScore

@create_docs(LogScore,
body=
"""
Since the score is undefined in the case that the true observation is
predicted to occur with probability zero, probablities are clipped
between `tol` and `1-tol`, where `tol` is a constructor key-word
argument.

If `sᵢ` is the predicted probability for the true class `yᵢ` then
the score for that example is given by

    log(clamp(sᵢ, tol), 1 - tol)

A score is reported for every observation. See also [`LogLoss`](@ref),
which differs only in sign.
""",
scitype=DOC_FINITE)

# workaround for https://github.com/JuliaLang/julia/issues/41939:
@static if VERSION < v"1.1"
    Base.clamp(::Missing, lo::Any, hi::Any) = missing
end

# for single finite observation:
single(c::LogScore, d::UnivariateFinite{S,V,R,P}, η) where {S,V,R,P} =
    log(clamp(pdf(d, η), P(c.tol), P(1) - P(c.tol)))

# performant broadasting in case of UnivariateFiniteArray:
call(c::LogScore, ŷ::UnivariateFiniteArray{S,V,R,P,N},
     y::ArrMissing{V,N}) where {S,V,R,P,N} =
    log.(clamp.(broadcast(pdf, ŷ, y), P(c.tol), P(1) - P(c.tol)))


# ---------------------------------------------------------------------
# LogLoss

struct LogLoss{R <: Real} <: Unaggregated
    tol::R
end
LogLoss(;eps=eps(), tol=eps) = LogLoss(tol)

metadata_measure(LogLoss;
                 instances                = ["log_loss", "cross_entropy"],
                 target_scitype           = Arr{<:Union{Missing,Finite}},
                 prediction_type          = :probabilistic,
                 orientation              = :loss,
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

A score is reported for every observation. See also
[`LogScore`](@ref), which differs only in sign.
""",
scitype=DOC_FINITE)

# for single finite observation:
single(c::LogLoss, d::UnivariateFinite{S,V,R,P}, η) where {S,V,R,P} =
    -single(LogScore(tol=c.tol), d, η)

# performant broadasting in case of UnivariateFiniteArray:
call(c::LogLoss, ŷ::UnivariateFiniteArray{S,V,R,P,N},
     y::ArrMissing{V,N}) where {S,V,R,P,N} =
    -call(LogScore(tol=c.tol), ŷ, y)


# -----------------------------------------------------
# BrierScore

struct BrierScore <: Unaggregated end

metadata_measure(BrierScore;
                 human_name = "Brier score (a.k.a. quadratic score)",
                 instances                = ["brier_score",],
                 target_scitype           = Arr{<:Union{Missing,Finite}},
                 prediction_type          = :probabilistic,
                 orientation              = :score,
                 is_feature_dependent     = false,
                 supports_weights         = true,
                 distribution_type        = UnivariateFinite)

@create_aliases BrierScore

@create_docs(BrierScore,
body=
"""
Convention as in $PROPER_SCORING_RULES: If `p(y)` is the predicted
probability for a *single* observation `y`, and `C` all possible
classes, then the corresponding Brier score for that observation is
given by

``2p(y) - \\left(\\sum_{η ∈ C} p(η)^2\\right) - 1``

*Warning.* `BrierScore()` is a "score" in the sense that bigger is
better (with `0` optimal, and all other values negative). In Brier's
original 1950 paper, and many other places, it has the opposite sign,
despite the name. Moreover, the present implementation does not treat
the binary case as special, so that the score may differ, in that
case, by a factor of two from usage elsewhere.
""",
scitype=DOC_FINITE)

# calling on single observations:
function single(::BrierScore, d::UnivariateFinite{S,V,R,P}, y) where {S,V,R,P}
    levels = classes(d)
    pvec = broadcast(pdf, d, levels)
    offset = P(1) + sum(pvec.^2)
    return P(2) * pdf(d, y) - offset
end

# Performant version in case of UnivariateFiniteArray:
function call(::BrierScore,
              ŷ::UnivariateFiniteArray{S,V,R,P,N},
              y::ArrMissing{V,N},
              w::Union{Nothing,Arr{<:Real,N}}=nothing) where {S,V,R,P<:Real,N}

    probs = pdf(ŷ, classes(first(ŷ)))
    offset = P(1) .+ vec(sum(probs.^2, dims=2))

    unweighted = P(2) .* broadcast(pdf, ŷ, y) .- offset

    if w === nothing
        return unweighted
    end
    return w.*unweighted
end

# -----------------------------------------------------
# BrierLoss

struct BrierLoss <: Unaggregated end

metadata_measure(BrierLoss;
                 human_name = "Brier loss (a.k.a. quadratic loss)",
                 instances                = ["brier_loss",],
                 target_scitype           = Arr{<:Union{Missing,Finite}},
                 prediction_type          = :probabilistic,
                 orientation              = :score,
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
scitype=DOC_FINITE)

# calling on single observations (no checks):
single(m::BrierLoss, d::UnivariateFinite{S,V,R,P}, y) where {S,V,R,P} =
    - single(BrierScore(), d, y)

# to get performant broadcasting in case of UnivariateFiniteArray
call(m::BrierLoss,
     ŷ::UnivariateFiniteArray{S,V,R,P,N},
     y::ArrMissing{V,N},
     w::Union{Nothing,Arr{<:Real,N}}=nothing) where {S,V,R,P<:Real,N} =
         - call(BrierScore(), ŷ, y, w)

# -----------------------------------------------------
# SphericalScore

struct SphericalScore{T<:Real} <: Unaggregated
    alpha::T
end
SphericalScore(; alpha=2) = SphericalScore(alpha)

metadata_measure(SphericalScore;
                 human_name               = "Spherical score",
                 instances                = ["spherical_score",],
                 target_scitype           = Arr{<:Union{Missing,Finite}},
                 prediction_type          = :probabilistic,
                 orientation              = :score,
                 is_feature_dependent     = false,
                 supports_weights         = true,
                 distribution_type        = UnivariateFinite)

@create_aliases SphericalScore

@create_docs(SphericalScore,
body=
"""
Convention as in $PROPER_SCORING_RULES: If `η` takes on a finite
number of classes `C` and ``p(η)` is the predicted probability for a
*single* observation `η`, then the corresponding Brier score for that
observation is given by

``p(y)^α / \\left(\\sum_{η ∈ C} p(η)^α\\right)^{1-α} - 1``

where `α` is the measure parameter `alpha`.

$DOC_DISTRIBUTIONS

""",
scitype=DOC_FINITE)

# calling on single observations:
function single(s::SphericalScore,
                d::UnivariateFinite{S,V,R,P}, y) where {S,V,R,P}
    α = s.alpha
    levels = classes(d)
    pvec = broadcast(pdf, d, levels)
    return (pdf(d, y)/norm(pvec, α))^(α - 1)
end

# to compute the α-norm along last dimension:
_norm(A::AbstractArray{<:Any,N}, α) where N =
    sum(x -> x^α, A, dims=N)^(1/α)

# Performant version in case of UnivariateFiniteArray:
function call(s::SphericalScore,
              ŷ::UnivariateFiniteArray{S,V,R,P,N},
              y::ArrMissing{V,N},
              w::Union{Nothing,Arr{<:Real,N}}=nothing) where {S,V,R,P<:Real,N}

    α = s.alpha
    alphanorm(A) = _norm(A, α)

    predicted_probs = pdf(ŷ, classes(first(ŷ)))

    unweighted = (broadcast(pdf, ŷ, y) ./ alphanorm(predicted_probs)).^(α - 1)

    if w === nothing
        return unweighted
    end
    return w.*unweighted
end

# -------------------------
# Infinite

const FORMULA_INFINITE_BRIER = "``2d(η) - ∫ d(t)^2 dt``"
_infinite_brier(d, y) = 2*pdf(d, y) - Distributions.pdfsquaredL2norm(d)

const FORMULA_INFINITE_SPHERICAL =
    "``d(η) / \\left(∫ d(t)^2 dt\\right)^{1/2}``"
_infinite_spherical(d, y) = pdf(d, y)/sqrt(Distributions.pdfsquaredL2norm(d))

const FORMULA_INFINITE_LOG = "``log(d(η))``"
# _infinite_log(d, y) = logpdf(d, y)

# helper to broadcast single observation versions:
function _broadcast_measure(measure, f, ŷm, ym, wm)

    ŷ, y, w = skipinvalid(ŷm, ym, wm)

    check_dimensions(ŷ, y)
    w === nothing || check_dimensions(w, y)

    isempty(ŷ) || check_distribution_supported(measure, first(ŷ))

    unweighted = broadcast(f, ŷ, y)

    if w === nothing
        return unweighted
    end
    return w.*unweighted
end

doc_body(formula) =
"""
Note here that `ŷ` is an array of *probabilistic* predictions. For
example, predictions might be `Normal` or `Poisson` distributions.

Convention as in $PROPER_SCORING_RULES: If `d` is a *single* predicted
probability density or mass function, and `η` the corresponding ground
truth observation, then the score for that observation is

$formula

"""

for (Measure, FORMULA, f, human_name) in [
    (:InfiniteBrierScore,     FORMULA_INFINITE_BRIER,     _infinite_brier,
     "Brier (or quadratic) score"),
    (:InfiniteSphericalScore, FORMULA_INFINITE_SPHERICAL, _infinite_spherical,
     "Spherical score"),
    (:InfiniteLogScore,       FORMULA_INFINITE_LOG,       logpdf,
     "Logarithmic score")]

    measure_str = string(Measure)
    instance_str = StatisticalTraits.snakecase(measure_str)

    quote

        struct $Measure <: Unaggregated end

        err_distribution(::$Measure, d) = ArgumentError(
            "Distribution $d is not supported by `"*$measure_str*"`. "*
            "Supported distributions are "*
            join(string.(map(s->"`$s`", WITH_L2NORM_INFINITE)), ", ", ", and "))

        check_distribution_supported(measure::$Measure, d) =
            d isa Union{WITH_L2NORM_INFINITE...} ||
            throw(err_distribution(measure, d))

        metadata_measure($Measure;
                         target_scitype = Arr{<:Union{Missing,Infinite}},
                         prediction_type          = :probabilistic,
                         orientation              = :score,
                         is_feature_dependent     = false,
                         supports_weights         = true,
                         distribution_type        =
                         Distributions.UnivariateDistribution)

        StatisticalTraits.instances(::Type{<:$Measure})  = [$instance_str,]
        StatisticalTraits.human_name(::Type{<:$Measure}) =
            $human_name*" for a continuous or unbounded discrete target"

        @create_aliases $Measure

        @create_docs($Measure, body=doc_body($FORMULA), scitype=DOC_FINITE)

        (measure::$Measure)(
            ŷ::Arr{<:Union{Missing,Distributions.UnivariateDistribution}},
            y::Arr{<:Any,N},
            w::Union{Nothing,Arr{<:Real,N}}=nothing) where N =
                _broadcast_measure(measure, $f, ŷ, y, w)

    end |> eval
end
