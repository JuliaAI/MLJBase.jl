## TRAITS FOR MEASURES

is_measure(::Any) = false

const MEASURE_TRAITS =
    [:name, :target_scitype, :supports_weights, :prediction_type, :orientation,
     :reports_each_observation, :is_feature_dependent]

# already defined in model_traits.jl:
# name              - fallback for non-MLJType is string(M) where M is arg
# target_scitype    - fallback value = Unknown
# supports_weights  - fallback value = false
# prediction_type   - fallback value = :unknown (also: :deterministic,
#                                           :probabilistic, :interval)

# specfic to measures:
orientation(::Type) = :loss  # other options are :score, :other
reports_each_observation(::Type) = false
is_feature_dependent(::Type) = false

# extend to instances:
orientation(m) = orientation(typeof(m))
reports_each_observation(m) = reports_each_observation(typeof(m))
is_feature_dependent(m) = is_feature_dependent(typeof(m))

# specific to probabilistic measures:
distribution_type(::Type) = missing


## DISPATCH FOR EVALUATION

# yhat - predictions (point or probabilisitic)
# X - features
# y - target observations
# w - per-observation weights

value(measure, yhat, X, y, w) = value(measure, yhat, X, y, w,
                                      Val(is_feature_dependent(measure)),
                                      Val(supports_weights(measure)))


## DEFAULT EVALUATION INTERFACE

#  is feature independent, weights not supported:
value(measure, yhat, X, y, w, ::Val{false}, ::Val{false}) = measure(yhat, y)

#  is feature dependent:, weights not supported:
value(measure, yhat, X, y, w, ::Val{true}, ::Val{false}) = measure(yhat, X, y)


#  is feature independent, weights supported:
value(measure, yhat, X, y, w, ::Val{false}, ::Val{true}) = measure(yhat, y, w)
value(measure, yhat, X, y, ::Nothing, ::Val{false}, ::Val{true}) = measure(yhat, y)

#  is feature dependent, weights supported:
value(measure, yhat, X, y, w, ::Val{true}, ::Val{true}) = measure(yhat, X, y, w)
value(measure, yhat, X, y, ::Nothing, ::Val{true}, ::Val{true}) = measure(yhat, X, y)


## HELPERS

"""
    check_dimension(ŷ, y)

Check that two vectors have compatible dimensions
"""
function check_dimensions(ŷ::AbstractVector, y::AbstractVector)
    length(y) == length(ŷ) ||
        throw(DimensionMismatch("Differing numbers of observations and "*
                                "predictions. "))
    return nothing
end

function check_pools(ŷ, y)
    levels(y) == levels(ŷ[1]) ||
        error("Conflicting categorical pools found "*
              "in observations and predictions. ")
    return nothing
end


## FOR BUILT-IN MEASURES

abstract type Measure <: MLJType end
is_measure(::Measure) = true


Base.show(stream::IO, ::MIME"text/plain", m::Measure) = print(stream, "$(name(m)) (callable Measure)")
Base.show(stream::IO, m::Measure) = print(stream, name(m))

MLJBase.info(measure, ::Val{:measure}) =
    (name=name(measure),
     target_scitype=target_scitype(measure),
     prediction_type=prediction_type(measure),
     orientation=orientation(measure),
     reports_each_observation=reports_each_observation(measure),
     is_feature_dependent=is_feature_dependent(measure),
     supports_weights=supports_weights(measure))


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


## CLASSIFICATION METRICS (FOR DETERMINISTIC PREDICTIONS)

struct MisclassificationRate <: Measure end

"""
    misclassification_rate(ŷ, y)
    misclassification_rate(ŷ, y, w)

Returns the rate of misclassification of the (point) predictions `ŷ`,
given true observations `y`, optionally weighted by the weights
`w`. All three arguments must be abstract vectors of the same length.

For more information, run `info(misclassification_rate)`.

"""
misclassification_rate = MisclassificationRate()
name(::Type{<:MisclassificationRate}) = "misclassification_rate"
target_scitype(::Type{<:MisclassificationRate}) = AbstractVector{<:Finite}
prediction_type(::Type{<:MisclassificationRate}) = :deterministic
orientation(::Type{<:MisclassificationRate}) = :loss
reports_each_observation(::Type{<:MisclassificationRate}) = false
is_feature_dependent(::Type{<:MisclassificationRate}) = false
supports_weights(::Type{<:MisclassificationRate}) = true

(::MisclassificationRate)(ŷ::AbstractVector{<:CategoricalElement},
                          y::AbstractVector{<:CategoricalElement}) =
                              mean(y .!= ŷ)
(::MisclassificationRate)(ŷ::AbstractVector{<:CategoricalElement},
                          y::AbstractVector{<:CategoricalElement},
                          w::AbstractVector{<:Real}) =
                              sum((y .!= ŷ) .*w) / sum(w)


## CLASSIFICATION METRICS (FOR PROBABILISTIC PREDICTIONS)

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
target_scitype(::Type{<:CrossEntropy}) = AbstractVector{<:Finite}
prediction_type(::Type{<:CrossEntropy}) = :probabilistic
orientation(::Type{<:CrossEntropy}) = :loss
reports_each_observation(::Type{<:CrossEntropy}) = true
is_feature_dependent(::Type{<:CrossEntropy}) = false
supports_weights(::Type{<:CrossEntropy}) = false

# for single observation:
_cross_entropy(d, y) = -log(pdf(d, y))

function (::CrossEntropy)(ŷ::AbstractVector{<:UnivariateFinite},
                          y::AbstractVector{<:CategoricalElement})
    check_dimensions(ŷ, y)
    check_pools(ŷ, y)
    return broadcast(_cross_entropy, Any[ŷ...], y)
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

name(::Type{<:BrierScore{D}}) where D = "BrierScore{$(string(D))}"
target_scitype(::Type{<:BrierScore{D}}) where D = AbstractVector{<:Finite}
prediction_type(::Type{<:BrierScore}) = :probabilistic
orientation(::Type{<:BrierScore}) = :score
reports_each_observation(::Type{<:BrierScore}) = true
is_feature_dependent(::Type{<:BrierScore}) = false
supports_weights(::Type{<:BrierScore}) = true

# For single observations (no checks):

# UnivariateFinite:
function brier_score(d::UnivariateFinite, y)
    levels = classes(d)
    pvec = broadcast(pdf, d, levels)
    offset = 1 + sum(pvec.^2)
    return 2*pdf(d, y) - offset
end

# For multiple observations:

# UnivariateFinite:
function (::BrierScore{<:UnivariateFinite})(
    ŷ::AbstractVector{<:UnivariateFinite},
    y::AbstractVector{<:CategoricalElement})
    check_dimensions(ŷ, y)
    check_pools(ŷ, y)
    return broadcast(brier_score, Any[ŷ...], y)
end

function (score::BrierScore{<:UnivariateFinite})(
    ŷ, y, w::AbstractVector{<:Real})
    check_dimensions(y, w)
    return w .* score(ŷ, y) ./ (sum(w)/length(y))
end


## DEFAULT MEASURES

default_measure(model::M) where M<:Supervised =
    default_measure(model, target_scitype(M))
default_measure(model, ::Any) = nothing
default_measure(model::Deterministic,
                ::Type{<:Union{AbstractVector{Continuous},
                               AbstractVector{Count}}}) = rms
# default_measure(model::Probabilistic,
#                 ::Type{<:Union{AbstractVector{Continuous},
#                                AbstractVector{Count}}}) = rms
default_measure(model::Deterministic,
                ::Type{<:AbstractVector{<:Finite}}) =
                    misclassification_rate
default_measure(model::Probabilistic,
                ::Type{<:AbstractVector{<:Finite}}) =
                    cross_entropy
