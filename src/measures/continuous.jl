# ===========================================================
## DETERMINISTIC PREDICTIONS

# -----------------------------------------------------------
# MeanAbsoluteError

struct MeanAbsoluteError <: Measure end

metadata_measure(MeanAbsoluteError;
                 instances = ["mae", "mav", "mean_absolute_error",
                              "mean_absolute_value"],
                 target_scitype           = Union{Arr{Continuous},Arr{Count}},
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

function (::MeanAbsoluteError)(ŷ::Arr{<:Real}, y::Arr{<:Real})
    check_dimensions(ŷ, y)
    ret = zero(eltype(y))
    for i in eachindex(y)
        dev = abs(y[i] - ŷ[i])
        ret += dev
    end
    return ret / length(y)
end

function (::MeanAbsoluteError)(ŷ::Arr{<:Real}, y::Arr{<:Real},
                 w::Arr{<:Real})
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
                 target_scitype           = Union{Arr{Continuous},Arr{Count}},
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

function (::RootMeanSquaredError)(ŷ::Arr{<:Real}, y::Arr{<:Real})
    check_dimensions(ŷ, y)
    ret = zero(eltype(y))
    for i in eachindex(y)
        dev = (y[i] - ŷ[i])^2
        ret += dev
    end
    return sqrt(ret / length(y))
end

function (::RootMeanSquaredError)(ŷ::Arr{<:Real}, y::Arr{<:Real},
                 w::Arr{<:Real})
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
                 target_scitype           = Union{Arr{Continuous},Arr{Count}},
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

function (m::LPLoss)(ŷ::Arr{<:Real}, y::Arr{<:Real})
    check_dimensions(ŷ, y)
    return abs.((y - ŷ)).^(m.p)
end

function (m::LPLoss)(ŷ::Arr{<:Real}, y::Arr{<:Real},
                w::Arr{<:Real})
    check_dimensions(ŷ, y)
    check_dimensions(w, y)
    return w .* abs.((y - ŷ)).^(m.p)
end

# ----------------------------------------------------------------------------
# RootMeanSquaredLogError

struct RootMeanSquaredLogError <: Measure end

metadata_measure(RootMeanSquaredLogError;
                 instances = ["rmsl", "rmsle", "root_mean_squared_log_error"],
                 target_scitype           = Union{Arr{Continuous},Arr{Count}},
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

function (::RootMeanSquaredLogError)(ŷ::Arr{<:Real}, y::Arr{<:Real})
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
                 target_scitype           = Union{Arr{Continuous},Arr{Count}},
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

function (m::RMSLP)(ŷ::Arr{<:Real}, y::Arr{<:Real})
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
    target_scitype           = Union{Arr{Continuous},Arr{Count}},
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

function (m::RootMeanSquaredProportionalError)(ŷ::Arr{<:Real}, y::Arr{<:Real})
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
    target_scitype           = Union{Arr{Continuous},Arr{Count}},
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

function (m::MeanAbsoluteProportionalError)(ŷ::Arr{<:Real}, y::Arr{<:Real})
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
    target_scitype           = Union{Arr{Continuous},Arr{Count}},
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

function (log_cosh::LogCoshLoss)(ŷ::Arr{<:T}, y::Arr{<:T}) where T <:Real
    check_dimensions(ŷ, y)
    return _log_cosh.(ŷ - y)
end

# ===========================================================
## PROBABLISTIC PREDICTIONS

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

        struct $Measure <: Measure end

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
                         reports_each_observation = true,
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
