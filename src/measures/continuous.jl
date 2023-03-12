const InfiniteArrMissing = Union{
    AbstractArray{<:Union{Missing,Continuous}},
    AbstractArray{<:Union{Missing,Count}}}

# -----------------------------------------------------------
# MeanAbsoluteError

struct MeanAbsoluteError <: Aggregated end

metadata_measure(MeanAbsoluteError;
                 instances = ["mae", "mav", "mean_absolute_error",
                              "mean_absolute_value"],
                 target_scitype           = InfiniteArrMissing,
                 prediction_type          = :deterministic,
                 orientation              = :loss),

const MAE = MeanAbsoluteError
const MAV = MeanAbsoluteError
@create_aliases MeanAbsoluteError

@create_docs(MeanAbsoluteError,
body=
"""
``\\text{mean absolute error} =  n^{-1}∑ᵢ|yᵢ-ŷᵢ|`` or
``\\text{mean absolute error} = n^{-1}∑ᵢwᵢ|yᵢ-ŷᵢ|``
""",
scitype=DOC_INFINITE)

call(::MeanAbsoluteError, ŷ, y) = abs.(ŷ .- y) |> skipinvalid |> mean
call(::MeanAbsoluteError, ŷ, y, w) = abs.(ŷ .- y) .* w |> skipinvalid |> mean

# ----------------------------------------------------------------
# RootMeanSquaredError

struct RootMeanSquaredError <: Aggregated end

metadata_measure(RootMeanSquaredError;
                 instances                = ["rms", "rmse",
                                             "root_mean_squared_error"],
                 target_scitype           = InfiniteArrMissing,
                 prediction_type          = :deterministic,
                 orientation              = :loss,
                 aggregation              = RootMeanSquare())

const RMS = RootMeanSquaredError
@create_aliases RootMeanSquaredError

@create_docs(RootMeanSquaredError,
body=
"""
``\\text{root mean squared error} = \\sqrt{n^{-1}∑ᵢ|yᵢ-ŷᵢ|^2}`` or
``\\text{root mean squared error} = \\sqrt{\\frac{∑ᵢwᵢ|yᵢ-ŷᵢ|^2}{∑ᵢwᵢ}}``
""",
scitype=DOC_INFINITE)

call(::RootMeanSquaredError, ŷ, y) = (y .- ŷ).^2 |> skipinvalid |> mean |> sqrt
call(::RootMeanSquaredError, ŷ, y, w) = (y .- ŷ).^2 .* w |> skipinvalid |> mean |> sqrt

# -------------------------------------------------------------------------
# R-squared (coefficient of determination)

struct RSquared <: Aggregated end

metadata_measure(RSquared;
                 instances               = ["rsq", "rsquared"],
                 target_scitype          = InfiniteArrMissing,
                 prediction_type         = :deterministic,
                 orientation             = :score,
                 supports_weights        = false)

const RSQ = RSquared
@create_aliases RSquared

@create_docs(RSquared,
body=
"""
The R² (also known as R-squared or coefficient of determination) is suitable for
interpreting linear regression analysis (Chicco et al., [2021](https://doi.org/10.7717/peerj-cs.623)).

Let ``\\overline{y}`` denote the mean of ``y``, then

``\\text{R^2} = 1 - \\frac{∑ (\\hat{y} - y)^2}{∑ \\overline{y} - y)^2}.``
""",
scitype=DOC_INFINITE)

function call(::RSquared, ŷ, y)
    num = (ŷ .- y).^2 |> skipinvalid |> sum
    mean_y = mean(y)
    denom = (mean_y .- y).^2 |> skipinvalid |> sum
    return 1 - (num / denom)
end

# -------------------------------------------------------------------
# LP

struct LPLoss{T<:Real} <: Unaggregated
    p::T
end

LPLoss(; p=2.0) = LPLoss(p)

metadata_measure(LPLoss;
                 instances = ["l1", "l2"],
                 target_scitype           = InfiniteArrMissing,
                 prediction_type          = :deterministic,
                 orientation              = :loss)

const l1 = LPLoss(1)
const l2 = LPLoss(2)

@create_docs(LPLoss,
body=
"""
Constructor signature: `LPLoss(p=2)`. Reports
`|ŷ[i] - y[i]|^p` for every index `i`.
""",
scitype=DOC_INFINITE)

single(m::LPLoss, ŷ, y) =  abs(y - ŷ)^(m.p)

# ----------------------------------------------------------------------------
# RootMeanSquaredLogError

struct RootMeanSquaredLogError <: Aggregated end

metadata_measure(RootMeanSquaredLogError;
                 instances = ["rmsl", "rmsle", "root_mean_squared_log_error"],
                 target_scitype           = InfiniteArrMissing,
                 prediction_type          = :deterministic,
                 orientation              = :loss,
                 aggregation              = RootMeanSquare())

const RMSL = RootMeanSquaredLogError
@create_aliases RootMeanSquaredLogError

@create_docs(RootMeanSquaredLogError,
body=
"""
``\\text{root mean squared log error} =
\\sqrt{n^{-1}∑ᵢ\\log\\left({yᵢ \\over ŷᵢ}\\right)^2}``
""",
footer="See also [`rmslp1`](@ref).",
scitype=DOC_INFINITE)

call(::RootMeanSquaredLogError, ŷ, y) =
    (log.(y) - log.(ŷ)).^2 |> skipinvalid |> mean |> sqrt
call(::RootMeanSquaredLogError, ŷ, y, w) =
    (log.(y) - log.(ŷ)).^2 .* w |> skipinvalid |> mean |> sqrt

# ---------------------------------------------------------------------------
#  RootMeanSquaredLogProportionalError

struct RootMeanSquaredLogProportionalError{T<:Real} <: Aggregated
    offset::T
end

RootMeanSquaredLogProportionalError(; offset=1.0) =
    RootMeanSquaredLogProportionalError(offset)

metadata_measure(RootMeanSquaredLogProportionalError;
                 instances                = ["rmslp1", ],
                 target_scitype           = InfiniteArrMissing,
                 prediction_type          = :deterministic,
                 orientation              = :loss,
                 aggregation              = RootMeanSquare())

const RMSLP = RootMeanSquaredLogProportionalError
@create_aliases RootMeanSquaredLogProportionalError

@create_docs(RootMeanSquaredLogProportionalError,
body=
"""
Constructor signature: `RootMeanSquaredLogProportionalError(; offset = 1.0)`.

``\\text{root mean squared log proportional error} =
\\sqrt{n^{-1}∑ᵢ\\log\\left({yᵢ + \\text{offset} \\over ŷᵢ + \\text{offset}}\\right)}``
""",
footer="See also [`rmsl`](@ref). ",
scitype=DOC_INFINITE)

call(m::RMSLP, ŷ, y) =
    (log.(y .+ m.offset) - log.(ŷ .+ m.offset)).^2 |>
    skipinvalid |> mean |> sqrt

call(m::RMSLP, ŷ, y, w) =
    (log.(y .+ m.offset) - log.(ŷ .+ m.offset)).^2 .* w |>
    skipinvalid |> mean |> sqrt

# --------------------------------------------------------------------------
# RootMeanSquaredProportionalError

struct RootMeanSquaredProportionalError{T<:Real} <: Aggregated
    tol::T
end

RootMeanSquaredProportionalError(; tol=eps()) =
    RootMeanSquaredProportionalError(tol)

metadata_measure(RootMeanSquaredProportionalError;
    instances                = ["rmsp", ],
    target_scitype           = InfiniteArrMissing,
    prediction_type          = :deterministic,
    orientation              = :loss,
    aggregation              = RootMeanSquare())

const RMSP = RootMeanSquaredProportionalError
@create_aliases RMSP

@create_docs(RootMeanSquaredProportionalError,
body=
"""
Constructor keyword arguments: `tol` (default = `eps()`).

``\\text{root mean squared proportional error} =
\\sqrt{m^{-1}∑ᵢ \\left({yᵢ-ŷᵢ \\over yᵢ}\\right)^2}``

where the sum is over indices such that `abs(yᵢ) > tol` and `m` is the number
of such indices.

""", scitype=DOC_INFINITE)

function call(
    m::RootMeanSquaredProportionalError,
    ŷ,
    y,
    w=nothing,
    )
    ret = 0
    count = 0
    @inbounds for i in eachindex(y)
        (isinvalid(y[i]) || isinvalid(ŷ[i])) && continue
        ayi = abs(y[i])
        if ayi > m.tol
            dev = ((y[i] - ŷ[i]) / ayi)^2
            ret += dev
            ret = _scale(ret, w, i)
            count += 1
        end
    end
    return sqrt(ret / count)
end

# -----------------------------------------------------------------------
# MeanAbsoluteProportionalError

struct MeanAbsoluteProportionalError{T} <: Aggregated
    tol::T
end

MeanAbsoluteProportionalError(; tol=eps()) = MeanAbsoluteProportionalError(tol)

metadata_measure(MeanAbsoluteProportionalError;
    instances                = ["mape", ],
    target_scitype           = InfiniteArrMissing,
    prediction_type          = :deterministic,
    orientation              = :loss)

const MAPE = MeanAbsoluteProportionalError
@create_aliases MAPE

@create_docs(MeanAbsoluteProportionalError,
body=
"""
Constructor key-word arguments: `tol` (default = `eps()`).

``\\text{mean absolute proportional error} =  m^{-1}∑ᵢ|{(yᵢ-ŷᵢ) \\over yᵢ}|``

where the sum is over indices such that `abs(yᵢ) > tol` and `m` is the number
of such indices.
""", scitype=DOC_INFINITE)

function call(
    m::MeanAbsoluteProportionalError,
    ŷ,
    y,
    w=nothing,
    )
    ret = 0
    count = 0
    @inbounds for i in eachindex(y)
        (isinvalid(y[i]) || isinvalid(ŷ[i])) && continue
        ayi = abs(y[i])
        if ayi > m.tol
        #if y[i] != zero(eltype(y))
            dev = abs((y[i] - ŷ[i]) / ayi)
            ret += dev
            ret =_scale(ret, w, i)
            count += 1
        end
    end
    return ret / count
end

# -------------------------------------------------------------------------
# LogCoshLoss

struct LogCoshLoss <: Unaggregated end

metadata_measure(LogCoshLoss;
    instances                = ["log_cosh", "log_cosh_loss"],
    target_scitype           = InfiniteArrMissing,
    prediction_type          = :deterministic,
    orientation              = :loss)

const LogCosh = LogCoshLoss
@create_aliases LogCoshLoss

@create_docs(LogCoshLoss,
             body="Reports ``\\log(\\cosh(ŷᵢ-yᵢ))`` for each index `i`. ",
             scitype=DOC_INFINITE)

_softplus(x::T) where T<:Real = x > zero(T) ? x + log1p(exp(-x)) : log1p(exp(x))
_log_cosh(x::T) where T<:Real = x + _softplus(-2x) - log(convert(T, 2))

single(::LogCoshLoss, ŷ, y) = _log_cosh(ŷ - y)
