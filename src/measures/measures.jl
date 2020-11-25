## TRAITS

is_measure_type(::Any) = false

const MEASURE_TRAITS =
    [:name, :instances, :human_name, :target_scitype, :supports_weights,
     :prediction_type, :orientation,
     :reports_each_observation, :aggregation, :is_feature_dependent, :docstring,
     :distribution_type, :supports_class_weights]

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
instances(::Type) = String[]
human_name(::Type) = snakecase(name(M), delim=' ')

# specific to `Finite` measures:
supports_class_weights(::Type) = false

# specific to probabilistic measures:
distribution_type(::Type) = missing

# extend to instances:
for trait in [:orientation, :reports_each_observation, :aggregation,
              :is_feature_dependent, :instances, :supports_class_weights,
              :distribution_type]
    eval(:($trait(m) = $trait(typeof(m))))
end


## FOR BUILT-IN MEASURES (subtyping Measure)

abstract type Measure <: MLJType end
is_measure_type(::Type{<:Measure}) = true
is_measure(m) = is_measure_type(typeof(m))

# docstring fall-back:
_decorate(s::AbstractString) = "`$s`"
_decorate(v::Vector{<:AbstractString}) = join(_decorate.(v), ", ")
function MMI.docstring(M::Type{<:Measure})
    list = _decorate(instances(M))
    ret = "`$(name(M))` - $(human_name(M)) type"
    isempty(list) || (ret *= " with instances $list")
    ret *= ". "
    return ret
end

# display:
show_as_constructed(::Type{<:Measure}) = true

# info (see also src/init.jl):
function MLJScientificTypes.info(M, ::Val{:measure_type})
    values = Tuple(@eval($trait($M)) for trait in MEASURE_TRAITS)
    return NamedTuple{Tuple(MEASURE_TRAITS)}(values)
end
MLJScientificTypes.info(m, ::Val{:measure}) = info(typeof(m))


## AGGREGATION

abstract type AggregationMode end

struct Sum <: AggregationMode end
(::Sum)(v) = sum(v)
(::Sum)(v::LittleDict) = sum(values(v))

struct Mean <: AggregationMode end
(::Mean)(v) = mean(v)
(::Mean)(v::LittleDict) = mean(values(v))

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

function value(measure, yhat, X, y, w)
    vfdep     = Val(is_feature_dependent(measure))
    vsweights = Val(supports_weights(measure))
    return value(measure, yhat, X, y, w, vfdep, vsweights)
end


## DEFAULT EVALUATION INTERFACE

#  is feature independent, weights not supported:
value(m, yhat, X, y, w, ::Val{false}, ::Val{false}) = m(yhat, y)

#  is feature dependent:, weights not supported:
value(m, yhat, X, y, w, ::Val{true}, ::Val{false}) = m(yhat, X, y)

#  is feature independent, weights supported:
value(m, yhat, X, y, w,         ::Val{false}, ::Val{true}) = m(yhat, y, w)
value(m, yhat, X, y, ::Nothing, ::Val{false}, ::Val{true}) = m(yhat, y)

#  is feature dependent, weights supported:
value(m, yhat, X, y, w,         ::Val{true}, ::Val{true}) = m(yhat, X, y, w)
value(m, yhat, X, y, ::Nothing, ::Val{true}, ::Val{true}) = m(yhat, X, y)

## helper

function check_pools(ŷ, y)
    levels(y) == levels(ŷ[1]) ||
        error("Conflicting categorical pools found "*
              "in observations and predictions. ")
    return nothing
end


## INCLUDE SPECIFIC MEASURES AND TOOLS

include("meta_utilities.jl")
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
