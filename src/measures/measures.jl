## TRAITS FOR MEASURES

is_measure_type(::Any) = false

const MEASURE_TRAITS =
    [:name, :target_scitype, :supports_weights, :prediction_type, :orientation,
     :reports_each_observation, :aggregation, :is_feature_dependent, :docstring,
     :distribution_type]

# already defined in model_traits.jl:
# name              - fallback for non-MLJType is string(M) where M is arg
# target_scitype    - fallback value = Unknown
# supports_weights  - fallback value = false
# prediction_type   - fallback value = :unknown (also: :deterministic,
#                                           :probabilistic, :interval)
# docstring         - fallback value is value of `name` trait.

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

# aggregation is no-op on scalars:
const MeasureValue = Union{Real,Tuple{<:Real,<:Real}} # number or interval
aggregate(x::MeasureValue, measure) = x


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


## helper

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


## DISPLAY AND INFO

Base.show(stream::IO, ::MIME"text/plain", m::Measure) =
    print(stream, "$(name(m)) (callable Measure)")
Base.show(stream::IO, m::Measure) = print(stream, name(m))

function MLJScientificTypes.info(M, ::Val{:measure_type})
    values = Tuple(@eval($trait($M)) for trait in MEASURE_TRAITS)
    return NamedTuple{Tuple(MEASURE_TRAITS)}(values)
end

MLJScientificTypes.info(m, ::Val{:measure}) = info(typeof(m))


"""
    metadata_measure(T; kw...)

Helper function to write the metadata for a single measure.
"""
function metadata_measure(T; name::String="",
                          target_scitype=Unknown,
                          prediction_type::Symbol=:unknown,
                          orientation::Symbol=:unknown,
                          reports_each_observation::Bool=true,
                          aggregation=Mean(),
                          is_feature_dependent::Bool=false,
                          supports_weights::Bool=false,
                          docstring::String="",
                          distribution_type=missing)
    pred_str        = "$prediction_type"
    orientation_str = "$orientation"
    dist = ifelse(ismissing(distribution_type), missing, "$distribution_type")
    ex = quote
        if !isempty($name)
            MMI.name(::Type{<:$T}) = $name
        end
        if !isempty($docstring)
            MMI.docstring(::Type{<:$T}) = $docstring
        end
        # traits common with models
        MMI.target_scitype(::Type{<:$T}) = $target_scitype
        MMI.prediction_type(::Type{<:$T}) = Symbol($pred_str)
        MMI.supports_weights(::Type{<:$T}) = $supports_weights
        # traits specific to measures
        orientation(::Type{<:$T}) = Symbol($orientation_str)
        reports_each_observation(::Type{<:$T}) = $reports_each_observation
        aggregation(::Type{<:$T}) = $aggregation
        is_feature_dependent(::Type{<:$T}) = $is_feature_dependent
        distribution_type(::Type{<:$T}) = $dist
    end
    parentmodule(T).eval(ex)
end

## INCLUDE SPECIFIC MEASURES AND TOOLS

include("continuous.jl")
include("confusion_matrix.jl")
include("finite.jl")
include("loss_functions_interface.jl")


## DEFAULT MEASURES
default_measure(T, S) = nothing

# Deterministic + Continuous / Count ==> RMS
default_measure(::Type{<:Deterministic},
                ::Type{<:Union{Vec{<:Continuous}, Vec{<:Count}}}) = rms

# Deterministic + Finite ==> Misclassification rate
default_measure(::Type{<:Deterministic},
                ::Type{<:Vec{<:Finite}}) = misclassification_rate

# default_measure(::Type{Probabilistic},
#                 ::Type{<:Union{Vec{<:Continuous},
#                                Vec{<:Count}}}) = ???

# Probabilistic + Finite ==> Cross entropy
default_measure(::Type{<:Probabilistic},
                ::Type{<:Vec{<:Finite}}) = cross_entropy

# Fallbacks
default_measure(M::Type{<:Supervised}) = default_measure(M, target_scitype(M))
default_measure(::M) where M <: Supervised = default_measure(M)
