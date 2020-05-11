## REGRESSOR METRICS (FOR DETERMINISTIC PREDICTIONS)

struct MAE <: Measure end

"""
    mae(ŷ, y)
    mae(ŷ, y, w)

Mean absolute error.

``\\text{MAE} =  n^{-1}∑ᵢ|yᵢ-ŷᵢ|`` or ``\\text{MAE} = n^{-1}∑ᵢwᵢ|yᵢ-ŷᵢ|``

For more information, run `info(mae)`.
"""
const mae = MAE()
const mav = MAE()

metadata_measure(MAE;
    name                     = "mae",
    target_scitype           = Union{Vec{Continuous},Vec{Count}},
    prediction_type          = :deterministic,
    orientation              = :loss,
    reports_each_observation = false,
    is_feature_dependent     = false,
    supports_weights         = true,
    docstring                = "mean absolute error; aliases: `mae`, `mav`")

function (::MAE)(ŷ::Vec{<:Real}, y::Vec{<:Real})
    check_dimensions(ŷ, y)
    ret = zero(eltype(y))
    for i in eachindex(y)
        dev = abs(y[i] - ŷ[i])
        ret += dev
    end
    return ret / length(y)
end

function (::MAE)(ŷ::Vec{<:Real}, y::Vec{<:Real},
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

struct RMS <: Measure end
"""
    rms(ŷ, y)
    rms(ŷ, y, w)

Root mean squared error:

``\\text{RMS} = \\sqrt{n^{-1}∑ᵢ|yᵢ-ŷᵢ|^2}`` or ``\\text{RMS} = \\sqrt{\\frac{∑ᵢwᵢ|yᵢ-ŷᵢ|^2}{∑ᵢwᵢ}}``

For more information, run `info(rms)`.
"""
const rms = RMS()

metadata_measure(RMS;
    name                     = "rms",
    target_scitype           = Union{Vec{Continuous},Vec{Count}},
    prediction_type          = :deterministic,
    orientation              = :loss,
    reports_each_observation = false,
    aggregation              = RootMeanSquare(),
    is_feature_dependent     = false,
    supports_weights         = true,
    docstring                = "root mean squared; aliases: `rms`.")

function (::RMS)(ŷ::Vec{<:Real}, y::Vec{<:Real})
    check_dimensions(ŷ, y)
    ret = zero(eltype(y))
    for i in eachindex(y)
        dev = (y[i] - ŷ[i])^2
        ret += dev
    end
    return sqrt(ret / length(y))
end

function (::RMS)(ŷ::Vec{<:Real}, y::Vec{<:Real},
                 w::Vec{<:Real})
    check_dimensions(ŷ, y)
    ret = zero(eltype(y))
    for i in eachindex(y)
        dev = (y[i] - ŷ[i])^2
        ret += w[i]*dev
    end
    return sqrt(ret / length(y))
end

struct L2 <: Measure end

"""
    l2(ŷ, y)
    l2(ŷ, y, w)

L2 per-observation loss.

For more information, run `info(l2)`.
"""
const l2 = L2()

metadata_measure(L2;
    name                     = "l2",
    target_scitype           = Union{Vec{Continuous},Vec{Count}},
    prediction_type          = :deterministic,
    orientation              = :loss,
    reports_each_observation = true,
    is_feature_dependent     = false,
    supports_weights         = true,
    docstring                = "squared deviations; aliases: `l2`.")

function (::L2)(ŷ::Vec{<:Real}, y::Vec{<:Real})
    check_dimensions(ŷ, y)
    return (y - ŷ).^2
end

function (::L2)(ŷ::Vec{<:Real}, y::Vec{<:Real},
                w::Vec{<:Real})
    check_dimensions(ŷ, y)
    check_dimensions(w, y)
    return w .* (y - ŷ).^2
end

struct L1 <: Measure end

"""
    l1(ŷ, y)
    l1(ŷ, y, w)

L1 per-observation loss.

For more information, run `info(l1)`.
"""
const l1 = L1()

metadata_measure(L1;
    name                     = "l1",
    target_scitype           = Union{Vec{Continuous},Vec{Count}},
    prediction_type          = :deterministic,
    orientation              = :loss,
    reports_each_observation = true,
    is_feature_dependent     = false,
    supports_weights         = true,
    docstring                = "absolute deviations; aliases: `l1`.")

function (::L1)(ŷ::Vec{<:Real}, y::Vec{<:Real})
    check_dimensions(ŷ, y)
    return abs.(y - ŷ)
end

function (::L1)(ŷ::Vec{<:Real}, y::Vec{<:Real},
                w::Vec{<:Real})
    check_dimensions(ŷ, y)
    check_dimensions(w, y)
    return w .* abs.(y - ŷ)
end

struct RMSL <: Measure end

"""
    rmsl(ŷ, y)

Root mean squared logarithmic error:

``\\text{RMSL} = n^{-1}∑ᵢ\\log\\left({yᵢ \\over ŷᵢ}\\right)``

For more information, run `info(rmsl)`.

See also [`rmslp1`](@ref).
"""
const rmsl = RMSL()

metadata_measure(RMSL;
    name                     = "rmsl",
    target_scitype           = Union{Vec{Continuous},Vec{Count}},
    prediction_type          = :deterministic,
    orientation              = :loss,
    reports_each_observation = false,
    aggregation              = RootMeanSquare(),
    is_feature_dependent     = false,
    supports_weights         = false,
    docstring                = "root mean square logarithm; aliases: `rmsl`.")

function (::RMSL)(ŷ::Vec{<:Real}, y::Vec{<:Real})
    check_dimensions(ŷ, y)
    ret = zero(eltype(y))
    for i in eachindex(y)
        dev = (log(y[i]) - log(ŷ[i]))^2
        ret += dev
    end
    return sqrt(ret / length(y))
end

struct RMSLP1 <: Measure end
"""
    rmslp1(ŷ, y)

Root mean squared logarithmic error with an offset of 1:

``\\text{RMSLP1} = n^{-1}∑ᵢ\\log\\left({yᵢ + 1 \\over ŷᵢ + 1}\\right)``

For more information, run `info(rmslp1)`.

See also [`rmsl`](@ref).
"""
const rmslp1 = RMSLP1()

metadata_measure(RMSLP1;
    name                     = "rmslp1",
    target_scitype           = Union{Vec{Continuous},Vec{Count}},
    prediction_type          = :deterministic,
    orientation              = :loss,
    reports_each_observation = false,
    aggregation              = RootMeanSquare(),
    is_feature_dependent     = false,
    supports_weights         = false,
    docstring                = "root mean squared logarithm plus one; " *
                               "aliases: `rmslp1`.")

function (::RMSLP1)(ŷ::Vec{<:Real}, y::Vec{<:Real})
    check_dimensions(ŷ, y)
    ret = zero(eltype(y))
    for i in eachindex(y)
        dev = (log(y[i] + 1) - log(ŷ[i] + 1))^2
        ret += dev
    end
    return sqrt(ret / length(y))
end

struct RMSP <: Measure end

"""
    rmsp(ŷ, y)

Root mean squared percentage loss:

``\\text{RMSP} = m^{-1}∑ᵢ \\left({yᵢ-ŷᵢ \\over yᵢ}\\right)^2``

where the sum is over indices such that `yᵢ≂̸0` and `m` is the number
of such indices.

For more information, run `info(rmsp)`.
"""
const rmsp = RMSP()

metadata_measure(RMSP;
    name                     = "rmsp",
    target_scitype           = Union{Vec{Continuous},Vec{Count}},
    prediction_type          = :deterministic,
    orientation              = :loss,
    reports_each_observation = false,
    aggregation              = RootMeanSquare(),
    is_feature_dependent     = false,
    supports_weights         = false,
    docstring                = "root mean square proportions; aliases: `rmsp`.")

function (::RMSP)(ŷ::Vec{<:Real}, y::Vec{<:Real}, tol=eps())
    check_dimensions(ŷ, y)
    ret = zero(eltype(y))
    count = 0
    @inbounds for i in eachindex(y)
        ayi = abs(y[i])
        if ayi > tol
            dev = ((y[i] - ŷ[i]) / ayi)^2
            ret += dev
            count += 1
        end
    end
    return sqrt(ret / count)
end

struct MAPE <: Measure
    tol::Real
end

MAPE(; tol=eps()) = MAPE(tol)

"""
     MAPE(; tol=esp())

Mean Absolute Percentage Error:

``\\text{MAPE} =  m^{-1}∑ᵢ|{(yᵢ-ŷᵢ) \\over yᵢ}|``
where the sum is over indices such that `yᵢ > tol` and `m` is the number
of such indices.

For more information, run `info(mape)`.
"""
const mape = MAPE()

metadata_measure(MAPE;
    name                     = "mape",
    target_scitype           = Union{Vec{Continuous},Vec{Count}},
    prediction_type          = :deterministic,
    orientation              = :loss,
    reports_each_observation = false,
    is_feature_dependent     = false,
    supports_weights         = false,
                 docstring                = "Mean Absolute Percentage Error; "*
                 "aliases: `mape=MAPE()`.")

function (m::MAPE)(ŷ::Vec{<:Real}, y::Vec{<:Real})
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
