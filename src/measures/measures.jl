## TRAITS FOR MEASURES

is_measure_type(::Any) = false

const MEASURE_TRAITS =
    [:name, :target_scitype, :supports_weights, :prediction_type, :orientation,
     :reports_each_observation, :aggregation, :is_feature_dependent]

# already defined in model_traits.jl:
# name              - fallback for non-MLJType is string(M) where M is arg
# target_scitype    - fallback value = Unknown
# supports_weights  - fallback value = false
# prediction_type   - fallback value = :unknown (also: :deterministic,
#                                           :probabilistic, :interval)

# specfic to measures:
orientation(::Type) = :loss  # other options are :score, :other
reports_each_observation(::Type) = false
aggregation(::Type) = Mean()  # other option is Sum() or callable object
is_feature_dependent(::Type) = false

# extend to instances:
orientation(m) = orientation(typeof(m))
reports_each_observation(m) = reports_each_observation(typeof(m))
aggregation(m) = aggregation(typeof(m))
is_feature_dependent(m) = is_feature_dependent(typeof(m))

# specific to probabilistic measures:
distribution_type(::Type) = missing


## AGGREGATION

abstract type AggregationMode end

struct Sum <: AggregationMode end
(::Sum)(v) = sum(v)

struct Mean <: AggregationMode end
(::Mean)(v) = mean(v)

# for rms and it's cousins:
struct RootMeanSquare <: AggregationMode end
(::RootMeanSquare)(v) = sqrt(mean(v.^2))

aggregate(v, measure) = aggregation(measure)(v)


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
is_measure_type(::Type{<:Measure}) = true
is_measure(m) = is_measure_type(typeof(m))

Base.show(stream::IO, ::MIME"text/plain", m::Measure) =
    print(stream, "$(name(m)) (callable Measure)")
Base.show(stream::IO, m::Measure) = print(stream, name(m))

function MLJBase.info(M, ::Val{:measure_type})
    values = Tuple(@eval($trait($M)) for trait in MEASURE_TRAITS)
    return NamedTuple{Tuple(MEASURE_TRAITS)}(values)
end

MLJBase.info(m, ::Val{:measure}) = info(typeof(m))



## INCLUDE SPECIFIC MEASURES AND TOOLS

include("continuous.jl")
include("confusion_matrix.jl")
include("finite.jl")
include("loss_functions_interface.jl")


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
