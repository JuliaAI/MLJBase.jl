## REGRESSOR METRICS (FOR DETERMINISTIC PREDICTIONS)

mutable struct MAV<: Measure end
"""
    mav(ŷ, y)
    mav(ŷ, y, w)

Mean absolute error (also known as MAE).

``\\text{MAV} =  n^{-1}∑ᵢ|yᵢ-ŷᵢ|`` or ``\\text{MAV} =  ∑ᵢwᵢ|yᵢ-ŷᵢ|/∑ᵢwᵢ``

For more information, run `info(mav)`.

"""
mav = MAV()
name(::Type{<:MAV}) = "mav"

target_scitype(::Type{<:MAV}) = Union{AbstractVector{Continuous},AbstractVector{Count}}
prediction_type(::Type{<:MAV}) = :deterministic
orientation(::Type{<:MAV}) = :loss
reports_each_observation(::Type{<:MAV}) = false
is_feature_dependent(::Type{<:MAV}) = false
supports_weights(::Type{<:MAV}) = true

function (::MAV)(ŷ::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    check_dimensions(ŷ, y)
    ret = 0.0
    for i in eachindex(y)
        dev = y[i] - ŷ[i]
        ret += abs(dev)
    end
    return ret / length(y)
end

function (::MAV)(ŷ::AbstractVector{<:Real}, y::AbstractVector{<:Real}, w::AbstractVector{<:Real})
    check_dimensions(ŷ, y)
    check_dimensions(y, w)
    ret = 0.0
    for i in eachindex(y)
        dev = w[i]*(y[i] - ŷ[i])
        ret += abs(dev)
    end
    return ret / sum(w)
end

# synonym
"""
mae(ŷ, y)

See also [`mav`](@ref).
"""
const mae = mav


struct RMS <: Measure end
"""
    rms(ŷ, y)
    rms(ŷ, y, w)

Root mean squared error:

``\\text{RMS} = \\sqrt{n^{-1}∑ᵢ|yᵢ-ŷᵢ|^2}`` or ``\\text{RMS} = \\sqrt{\\frac{∑ᵢwᵢ|yᵢ-ŷᵢ|^2}{∑ᵢwᵢ}}``

For more information, run `info(rms)`.

"""
rms = RMS()
name(::Type{<:RMS}) = "rms"
target_scitype(::Type{<:RMS}) = Union{AbstractVector{Continuous},AbstractVector{Count}}
prediction_type(::Type{<:RMS}) = :deterministic
orientation(::Type{<:RMS}) = :loss
reports_each_observation(::Type{<:RMS}) = false
is_feature_dependent(::Type{<:RMS}) = false
supports_weights(::Type{<:RMS}) = true

function (::RMS)(ŷ::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    check_dimensions(ŷ, y)
    ret = 0.0
    for i in eachindex(y)
        dev = y[i] - ŷ[i]
        ret += dev * dev
    end
    return sqrt(ret / length(y))
end

function (::RMS)(ŷ::AbstractVector{<:Real}, y::AbstractVector{<:Real}, w::AbstractVector{<:Real})
    check_dimensions(ŷ, y)
    ret = 0.0
    for i in eachindex(y)
        dev = y[i] - ŷ[i]
        ret += w[i]*dev*dev
    end
    return sqrt(ret / sum(w))
end

struct L2 <: Measure end
"""
    l2(ŷ, y)
    l2(ŷ, y, w)

L2 per-observation loss.

For more information, run `info(l2)`.

"""
l2 = L2()
name(::Type{<:L2}) = "l2"
target_scitype(::Type{<:L2}) = Union{AbstractVector{Continuous},AbstractVector{Count}}
prediction_type(::Type{<:L2}) = :deterministic
orientation(::Type{<:L2}) = :loss
reports_each_observation(::Type{<:L2}) = true
is_feature_dependent(::Type{<:L2}) = false
supports_weights(::Type{<:L2}) = true

function (::L2)(ŷ::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    (check_dimensions(ŷ, y); (y - ŷ).^2)
end

function (::L2)(ŷ::AbstractVector{<:Real}, y::AbstractVector{<:Real}, w::AbstractVector{<:Real})
    check_dimensions(ŷ, y)
    check_dimensions(w, y)
    return (y - ŷ).^2 .* w ./ (sum(w)/length(y))
end

struct L1 <: Measure end
"""
    l1(ŷ, y)
    l1(ŷ, y, w)

L1 per-observation loss.

For more information, run `info(l1)`.

"""
l1 = L1()
name(::Type{<:L1}) = "l1"
target_scitype(::Type{<:L1}) = Union{AbstractVector{Continuous},AbstractVector{Count}}
prediction_type(::Type{<:L1}) = :deterministic
orientation(::Type{<:L1}) = :loss
reports_each_observation(::Type{<:L1}) = true
is_feature_dependent(::Type{<:L1}) = false
supports_weights(::Type{<:L1}) = true

function (::L1)(ŷ::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    (check_dimensions(ŷ, y); abs.(y - ŷ))
end

function (::L1)(ŷ::AbstractVector{<:Real}, y::AbstractVector{<:Real}, w::AbstractVector{<:Real})
    check_dimensions(ŷ, y)
    check_dimensions(w, y)
    return abs.(y - ŷ) .* w ./ (sum(w)/length(y))
end

struct RMSL <: Measure end
"""
    rmsl(ŷ, y)

Root mean squared logarithmic error:

``\\text{RMSL} = n^{-1}∑ᵢ\\log\\left({yᵢ \\over ŷᵢ}\\right)``

For more information, run `info(rmsl)`.

See also [`rmslp1`](@ref).

"""
rmsl = RMSL()
name(::Type{<:RMSL}) = "rmsl"
target_scitype(::Type{<:RMSL}) = Union{AbstractVector{Continuous},AbstractVector{Count}}
prediction_type(::Type{<:RMSL}) = :deterministic
orientation(::Type{<:RMSL}) = :loss
reports_each_observation(::Type{<:RMSL}) = false
is_feature_dependent(::Type{<:RMSL}) = false
supports_weights(::Type{<:RMSL}) = false

function (::RMSL)(ŷ::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    check_dimensions(ŷ, y)
    ret = 0.0
    for i in eachindex(y)
        dev = log(y[i]) - log(ŷ[i])
        ret += dev * dev
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
rmslp1 = RMSLP1()
name(::Type{<:RMSLP1}) = "rmslp1"
target_scitype(::Type{<:RMSLP1}) = Union{AbstractVector{Continuous},AbstractVector{Count}}
prediction_type(::Type{<:RMSLP1}) = :deterministic
orientation(::Type{<:RMSLP1}) = :loss
reports_each_observation(::Type{<:RMSLP1}) = false
is_feature_dependent(::Type{<:RMSLP1}) = false
supports_weights(::Type{<:RMSLP1}) = false

function (::RMSLP1)(ŷ::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    check_dimensions(ŷ, y)
    ret = 0.0
    for i in eachindex(y)
        dev = log(y[i] + 1) - log(ŷ[i] + 1)
        ret += dev * dev
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
rmsp = RMSP()
name(::Type{<:RMSP}) = "rmsp"
target_scitype(::Type{<:RMSP}) = Union{AbstractVector{Continuous},AbstractVector{Count}}
prediction_type(::Type{<:RMSP}) = :deterministic
orientation(::Type{<:RMSP}) = :loss
reports_each_observation(::Type{<:RMSP}) = false
is_feature_dependent(::Type{<:RMSP}) = false
supports_weights(::Type{<:RMSP}) = false

function (::RMSP)(ŷ::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    check_dimensions(ŷ, y)
    ret = 0.0
    count = 0
    for i in eachindex(y)
        if y[i] != 0.0
            dev = (y[i] - ŷ[i])/y[i]
            ret += dev * dev
            count += 1
        end
    end
    return sqrt(ret/count)
end
