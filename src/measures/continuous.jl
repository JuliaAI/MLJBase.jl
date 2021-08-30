# ===========================================================
## DETERMINISTIC PREDICTIONS

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

# -----------------------------------------------------------
# MeanAbsoluteError

struct MeanAbsoluteError <: Measure end

metadata_measure(MeanAbsoluteError;
                 instances = ["mae", "mav", "mean_absolute_error",
                              "mean_absolute_value"],
                 target_scitype           = Union{Vec{Continuous},Vec{Count}},
                 prediction_type          = :deterministic,
                 orientation              = :loss,
                 reports_each_observation = false,
                 is_feature_dependent     = false,
                 supports_weights         = true)

const MAE = MeanAbsoluteError
const MAV = MeanAbsoluteError
@create_aliases MeanAbsoluteError

@create_docs(MeanAbsoluteError,
body=
"""
``\\text{mean absolute error} =  n^{-1}∑ᵢ|yᵢ-ŷᵢ|`` or
``\\text{mean absolute error} = n^{-1}∑ᵢwᵢ|yᵢ-ŷᵢ|``
""")

function (::MeanAbsoluteError)(ŷ::Vec{<:Real}, y::Vec{<:Real})
    check_dimensions(ŷ, y)
    ret = zero(eltype(y))
    for i in eachindex(y)
        dev = abs(y[i] - ŷ[i])
        ret += dev
    end
    return ret / length(y)
end

function (::MeanAbsoluteError)(ŷ::Vec{<:Real}, y::Vec{<:Real},
                 w::Vec{<:Real})
    check_dimensions(ŷ, y)
    check_dimensions(y, w)
    ret = zero(eltype(y))
    for i in eachindex(y)
        dev = abs(y[i] - ŷ[i])
        ret += w[i]*dev
    end
    return ret / length(y)
end

# ----------------------------------------------------------------
# RootMeanSquaredError

struct RootMeanSquaredError <: Measure end

metadata_measure(RootMeanSquaredError;
                 instances                = ["rms", "rmse",
                                             "root_mean_squared_error"],
                 target_scitype           = Union{Vec{Continuous},Vec{Count}},
                 prediction_type          = :deterministic,
                 orientation              = :loss,
                 reports_each_observation = false,
                 aggregation              = RootMeanSquare(),
                 is_feature_dependent     = false,
                 supports_weights         = true)

const RMS = RootMeanSquaredError
@create_aliases RootMeanSquaredError

@create_docs(RootMeanSquaredError,
body=
"""
``\\text{root mean squared error} = \\sqrt{n^{-1}∑ᵢ|yᵢ-ŷᵢ|^2}`` or
``\\text{root mean squared error} = \\sqrt{\\frac{∑ᵢwᵢ|yᵢ-ŷᵢ|^2}{∑ᵢwᵢ}}``
""")

function (::RootMeanSquaredError)(ŷ::Vec{<:Real}, y::Vec{<:Real})
    check_dimensions(ŷ, y)
    ret = zero(eltype(y))
    for i in eachindex(y)
        dev = (y[i] - ŷ[i])^2
        ret += dev
    end
    return sqrt(ret / length(y))
end

function (::RootMeanSquaredError)(ŷ::Vec{<:Real}, y::Vec{<:Real},
                 w::Vec{<:Real})
    check_dimensions(ŷ, y)
    ret = zero(eltype(y))
    for i in eachindex(y)
        dev = (y[i] - ŷ[i])^2
        ret += w[i]*dev
    end
    return sqrt(ret / length(y))
end

# -------------------------------------------------------------------
# LP

struct LPLoss{T<:Real} <: Measure
    p::T
end

LPLoss(; p=2.0) = LPLoss(p)

metadata_measure(LPLoss;
                 instances = ["l1", "l2"],
                 target_scitype           = Union{Vec{Continuous},Vec{Count}},
                 prediction_type          = :deterministic,
                 orientation              = :loss,
                 reports_each_observation = true,
                 is_feature_dependent     = false,
                 supports_weights         = true)

const l1 = LPLoss(1)
const l2 = LPLoss(2)

@create_docs(LPLoss,
body=
"""
Constructor signature: `LPLoss(p=2)`. Reports
`|ŷ[i] - y[i]|^p` for every index `i`.
""")

function (m::LPLoss)(ŷ::Vec{<:Real}, y::Vec{<:Real})
    check_dimensions(ŷ, y)
    return abs.((y - ŷ)).^(m.p)
end

function (m::LPLoss)(ŷ::Vec{<:Real}, y::Vec{<:Real},
                w::Vec{<:Real})
    check_dimensions(ŷ, y)
    check_dimensions(w, y)
    return w .* abs.((y - ŷ)).^(m.p)
end

# ----------------------------------------------------------------------------
# RootMeanSquaredLogError

struct RootMeanSquaredLogError <: Measure end

metadata_measure(RootMeanSquaredLogError;
                 instances = ["rmsl", "rmsle", "root_mean_squared_log_error"],
                 target_scitype           = Union{Vec{Continuous},Vec{Count}},
                 prediction_type          = :deterministic,
                 orientation              = :loss,
                 reports_each_observation = false,
                 aggregation              = RootMeanSquare(),
                 is_feature_dependent     = false,
                 supports_weights         = false)

const RMSL = RootMeanSquaredLogError
@create_aliases RootMeanSquaredLogError

@create_docs(RootMeanSquaredLogError,
body=
"""
``\\text{root mean squared log error} =
n^{-1}∑ᵢ\\log\\left({yᵢ \\over ŷᵢ}\\right)``
""",
footer="See also [`rmslp1`](@ref).")

function (::RootMeanSquaredLogError)(ŷ::Vec{<:Real}, y::Vec{<:Real})
    check_dimensions(ŷ, y)
    ret = zero(eltype(y))
    for i in eachindex(y)
        dev = (log(y[i]) - log(ŷ[i]))^2
        ret += dev
    end
    return sqrt(ret / length(y))
end

# ---------------------------------------------------------------------------
#  RootMeanSquaredLogProportionalError

struct RootMeanSquaredLogProportionalError{T<:Real} <: Measure
    offset::T
end

RootMeanSquaredLogProportionalError(; offset=1.0) =
    RootMeanSquaredLogProportionalError(offset)

metadata_measure(RootMeanSquaredLogProportionalError;
                 instances                = ["rmslp1", ],
                 target_scitype           = Union{Vec{Continuous},Vec{Count}},
                 prediction_type          = :deterministic,
                 orientation              = :loss,
                 reports_each_observation = false,
                 aggregation              = RootMeanSquare(),
                 is_feature_dependent     = false,
                 supports_weights         = false)

const RMSLP = RootMeanSquaredLogProportionalError
@create_aliases RootMeanSquaredLogProportionalError

@create_docs(RootMeanSquaredLogProportionalError,
body=
"""
Constructor signature: `RootMeanSquaredLogProportionalError(; offset = 1.0)`.

``\\text{root mean squared log proportional error} =
n^{-1}∑ᵢ\\log\\left({yᵢ + \\text{offset} \\over ŷᵢ + \\text{offset}}\\right)``
""",
footer="See also [`rmsl`](@ref). ")

function (m::RMSLP)(ŷ::Vec{<:Real}, y::Vec{<:Real})
    check_dimensions(ŷ, y)
    ret = zero(eltype(y))
    for i in eachindex(y)
        dev = (log(y[i] + m.offset) - log(ŷ[i] + m.offset))^2
        ret += dev
    end
    return sqrt(ret / length(y))
end

# --------------------------------------------------------------------------
# RootMeanSquaredProportionalError

struct RootMeanSquaredProportionalError{T<:Real} <: Measure
    tol::T
end

RootMeanSquaredProportionalError(; tol=eps()) =
    RootMeanSquaredProportionalError(tol)

metadata_measure(RootMeanSquaredProportionalError;
    instances                = ["rmsp", ],
    target_scitype           = Union{Vec{Continuous},Vec{Count}},
    prediction_type          = :deterministic,
    orientation              = :loss,
    reports_each_observation = false,
    aggregation              = RootMeanSquare(),
    is_feature_dependent     = false,
    supports_weights         = false)

const RMSP = RootMeanSquaredProportionalError
@create_aliases RMSP

@create_docs(RootMeanSquaredProportionalError,
body=
"""
Constructor keyword arguments: `tol` (default = `eps()`).

``\\text{root mean squared proportional error} =
m^{-1}∑ᵢ \\left({yᵢ-ŷᵢ \\over yᵢ}\\right)^2``

where the sum is over indices such that `abs(yᵢ) > tol` and `m` is the number
of such indices.

""")

function (m::RootMeanSquaredProportionalError)(ŷ::Vec{<:Real}, y::Vec{<:Real})
    check_dimensions(ŷ, y)
    ret = zero(eltype(y))
    count = 0
    @inbounds for i in eachindex(y)
        ayi = abs(y[i])
        if ayi > m.tol
            dev = ((y[i] - ŷ[i]) / ayi)^2
            ret += dev
            count += 1
        end
    end
    return sqrt(ret / count)
end

# -----------------------------------------------------------------------
# MeanAbsoluteProportionalError

struct MeanAbsoluteProportionalError{T} <: Measure
    tol::T
end

MeanAbsoluteProportionalError(; tol=eps()) = MeanAbsoluteProportionalError(tol)

metadata_measure(MeanAbsoluteProportionalError;
    instances                = ["mape", ],
    target_scitype           = Union{Vec{Continuous},Vec{Count}},
    prediction_type          = :deterministic,
    orientation              = :loss,
    reports_each_observation = false,
    is_feature_dependent     = false,
    supports_weights         = false,
    docstring                = "Mean Absolute Proportional Error; "*
                 "aliases: `mape=MAPE()`.")

const MAPE = MeanAbsoluteProportionalError
@create_aliases MAPE

@create_docs(MeanAbsoluteProportionalError,
body=
"""
Constructor key-word arguments: `tol` (default = `eps()`).

``\\text{mean absolute proportional error} =  m^{-1}∑ᵢ|{(yᵢ-ŷᵢ) \\over yᵢ}|``

where the sum is over indices such that `abs(yᵢ) > tol` and `m` is the number
of such indices.
""")

function (m::MeanAbsoluteProportionalError)(ŷ::Vec{<:Real}, y::Vec{<:Real})
    check_dimensions(ŷ, y)
    ret = zero(eltype(y))
    count = 0
    @inbounds for i in eachindex(y)
        ayi = abs(y[i])
        if ayi > m.tol
        #if y[i] != zero(eltype(y))
            dev = abs((y[i] - ŷ[i]) / ayi)
            #dev = abs((y[i] - ŷ[i]) / y[i])
            ret += dev
            count += 1
        end
    end
    return ret / count
end

# -------------------------------------------------------------------------
# LogCoshLoss

struct LogCoshLoss <: Measure end

metadata_measure(LogCoshLoss;
    instances                = ["log_cosh", "log_cosh_loss"],
    target_scitype           = Union{Vec{Continuous},Vec{Count}},
    prediction_type          = :deterministic,
    orientation              = :loss,
    reports_each_observation = true,
    is_feature_dependent     = false,
    supports_weights         = false,
    docstring                = "log cosh loss; aliases: `log_cosh`.")

const LogCosh = LogCoshLoss
@create_aliases LogCoshLoss

@create_docs(LogCoshLoss,
body="Reports ``\\log(\\cosh(ŷᵢ-yᵢ))`` for each index `i`. ")

_softplus(x::T) where T<:Real = x > zero(T) ? x + log1p(exp(-x)) : log1p(exp(x))
_log_cosh(x::T) where T<:Real = x + _softplus(-2x) - log(convert(T, 2))

function (log_cosh::LogCoshLoss)(ŷ::Vec{<:T}, y::Vec{<:T}) where T <:Real
    check_dimensions(ŷ, y)
    return _log_cosh.(ŷ - y)
end

# -------------------------------------------------------------------------
# ContinuousBrierScore and CountBrierScore

const DOC_BODY_CONTINUOUS =
"""
This measure assigns a score for each observation of a continuous
variable, for which probabilistic predictions are being made. For
example, predictions might be `Normal` distributions.

If `ŷ` were a *single* predicted probability density function, and `y` the
corresponding ground truth observation, then the Brier score for that
observation is

``2ŷ(y) - ∫ ŷ(t)^2 dt``

*Note.* `ContinuousBrierScore()` is a "score" in the sense that bigger
is better. The usage here differs by a sign, factor of one half,
and/or a constant shift, from usage elsewhere.
"""

const DOC_BODY_COUNT =
"""
This measure assigns a score for each observation of a unbounded integer
variable, for which probabilistic predictions are being made. For
example, predictions might be `Poisson` distributions.

If `ŷ` were a *single* predicted probability mass function, and `y` the
corresponding ground truth observation, then the Brier score for that
observation is

``2ŷ(y) - ∑_j ŷ(j)^2``

*Note.* `CountBrierScore()` is a "score" in the sense that bigger
is better. The usage here differs by a sign, factor of one half,
and/or a constant shift from usage elsewhere.
"""

for (Measure, DISTRIBUTIONS,
     Scitype, Scitype_str, distribution_type,
     DOC_BODY, DOC_SCITYPE) in
    [(:ContinuousBrierScore, :WITH_L2NORM_CONTINUOUS,
      Continuous, "Continuous", Distributions.ContinuousUnivariateDistribution,
      DOC_BODY_CONTINUOUS, DOC_CONTINUOUS),
     (:CountBrierScore, :WITH_L2NORM_COUNT,
      Count, "Count", Distributions.DiscreteUnivariateDistribution,
      DOC_BODY_COUNT, DOC_COUNT)]

    measure_str = string(Measure)
    scitype_str = lowercase(Scitype_str)

    quote

        struct $Measure <: Measure end

        err_brier_distribution(::$Measure, d) = ArgumentError(
            "Distribution $d is not supported by `"*$measure_str*"`. "*
            "Supported distributions are "*
            join(string.(map(s->"`$s`", $DISTRIBUTIONS)), ", ", ", and "))

        check_distribution_supported(measure::$Measure, d) =
            d isa Union{$DISTRIBUTIONS...} ||
            throw(err_brier_distribution(measure, d))

        metadata_measure($Measure;
                         target_scitype = Arr{<:Union{Missing,$Scitype}},
                         prediction_type          = :probabilistic,
                         orientation              = :score,
                         reports_each_observation = true,
                         is_feature_dependent     = false,
                         supports_weights         = true,
                         distribution_type        = $distribution_type)

        StatisticalTraits.instances(::Type{<:$Measure}) =
            [$scitype_str*"_brier_score",]
        StatisticalTraits.human_name(::Type{<:$Measure}) =
            $Scitype_str*" Brier (or quadratic) score"

        @create_aliases $Measure

        @create_docs($Measure, body=$DOC_BODY, scitype=$DOC_SCITYPE)

    end |> eval
end

const InfiniteBrierScore = Union{ContinuousBrierScore,CountBrierScore}

# calling on single observations (no checks):
function _infinite_brier_score(d, y)
    return 2*pdf(d, y) - Distributions.pdfsquaredL2norm(d)
end

# calling on multiple observations:
function _infinite_brier_score(measure, ŷm, ym, wm)
    ŷ, y, w = _skipmissing(ŷm, ym, wm)

    check_dimensions(ŷ, y)
    w === nothing || check_dimensions(w, y)

    isempty(ŷ) || check_distribution_supported(measure, first(ŷ))

    unweighted = broadcast(_infinite_brier_score, ŷ, y)

    if w === nothing
        return unweighted
    end
    return w.*unweighted
end

(measure::ContinuousBrierScore)(
    ŷ::Arr{<:Union{Missing,Distributions.UnivariateDistribution}},
    y::Arr{<:Any,N},
    w::Union{Nothing,Arr{<:Real,N}}=nothing) where N =
        _infinite_brier_score(measure, ŷ, y, w)

(measure::CountBrierScore)(
    ŷ::Arr{<:Union{Missing,Distributions.UnivariateDistribution}},
    y::Arr{<:Any,N},
    w::Union{Nothing,Arr{<:Real,N}}=nothing) where N =
        _infinite_brier_score(measure, ŷ, y, w)
