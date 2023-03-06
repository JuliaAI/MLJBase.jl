const PROPER_SCORING_RULES = "[Gneiting and Raftery (2007), \"Strictly"*
    "Proper Scoring Rules, Prediction, and Estimation\""*
    "](https://doi.org/10.1198/016214506000001437)"
const DOC_FINITE =
    "`AbstractArray{<:Union{Finite,Missing}` (multiclass classification)"
const DOC_FINITE_BINARY =
    "`AbstractArray{<:Union{Finite{2},Missing}}` (binary classification)"
const DOC_ORDERED_FACTOR =
    "`AbstractArray{<:Union{OrderedFactor,Missing}}` (classification of ordered target)"
const DOC_ORDERED_FACTOR_BINARY =
    "`AbstractArray{<:Union{OrderedFactor{2},Missing}}` "*
    "(binary classification where choice of \"true\" effects the measure)"
const DOC_CONTINUOUS = "`AbstractArray{<:Union{Continuous,Missing}}` (regression)"
const DOC_COUNT = "`AbstractArray{<:Union{Count,Missing}}`"
const DOC_MULTI = "`AbtractArray{<:Union{Missing,T}` where `T` is `Continuous` "*
    "or `Count` (for respectively continuous or discrete Distribution.jl objects in "*
    "`ŷ`) or  `OrderedFactor` or `Multiclass` "*
    "(for `UnivariateFinite` distributions in `ŷ`)"

const DOC_INFINITE = "`AbstractArray{<:Union{Infinite,Missing}}`"
const INVARIANT_LABEL =
    "This metric is invariant to class reordering."
const VARIANT_LABEL =
    "This metric is *not* invariant to class re-ordering"

is_measure_type(::Any) = false

# Each of the following traits, with fallbacks defined in
# StatisticalTraits.jl, make sense for some or all measures:

const MEASURE_TRAITS = [
    :name,
    :instances,
    :human_name,
    :target_scitype,
    :supports_weights,
    :supports_class_weights,
    :prediction_type,
    :orientation,
    :reports_each_observation,
    :aggregation,
    :is_feature_dependent,
    :docstring,
    :distribution_type
]

# # FOR BUILT-IN MEASURES (subtyping Measure)

abstract type Measure <: MLJType end
abstract type Aggregated <: Measure end
abstract type Unaggregated <: Measure end

StatisticalTraits.reports_each_observation(::Type{<:Aggregated}) = false
StatisticalTraits.reports_each_observation(::Type{<:Unaggregated}) = true


# # FALLBACK CHECKS
extra_check(::Measure, args...) = nothing
function _check(measure::Measure, yhat, y)
    check_dimensions(yhat, y)
    extra_check(measure, yhat, y)
end
function _check(measure::Measure, yhat, y, w)
    check_dimensions(yhat, y)
    extra_check(measure, yhat, y, w)
end
function _check(measure::Measure, yhat, y, w::Arr)
    check_dimensions(yhat, y)
    check_dimensions(y, w)
    extra_check(measure, yhat, y, w)
end
function _check(measure::Measure, yhat::Arr{<:UnivariateFinite})
    check_dimensions(yhat, y)
    check_pools(yhat, y)
    extra_check(measure, yhat, y)
end

function _check(
    measure::Measure,
    yhat::Arr{<:UnivariateFinite},
    y,
    w::Arr
)
    check_dimensions(yhat, y)
    check_pools(yhat, y)
    extra_check(measure, yhat, y, w)
end

function _check(
    measure::Measure,
    yhat::Arr{<:UnivariateFinite},
    y,
    w::AbstractDict
)
    check_dimensions(yhat, y)
    check_pools(yhat, y)
    check_pools(yhat, w)
    extra_check(measure, yhat, y, w)
end

# # METHODS TO EVALUATE MEASURES

# See measures/README.md for details

# `robust_single` can accept `missing` observations/predictions but is never overloaded;
# `single` is overloaded but does not need to handle missings. This factoring allows us
# to avoid method ambiguities which are cumbersome to avoid with only one function.

robust_single(args...) = single(args...)
robust_single(m, ::Missing, ::Missing) = missing
robust_single(m, ::Missing, η) = missing
robust_single(m, η̂, ::Missing) = missing

const Label = Union{CategoricalValue, Number, AbstractString, Symbol, AbstractChar}

# closure for broadcasting:
robust_single(measure::Measure) = (ηhat, η) -> robust_single(measure, ηhat, η)

call(measure::Unaggregated, yhat, y) = broadcast(robust_single(measure), yhat, y)
function call(measure::Unaggregated, yhat, y, w::AbstractArray)
    unweighted = broadcast(robust_single(measure), yhat, y)
    return w .* unweighted
end
function call(measure::Unaggregated, yhat, y, weight_given_class::AbstractDict)
    unweighted = broadcast(robust_single(measure), yhat, y)
    w = @inbounds broadcast(η -> weight_given_class[η], y)
    return w .* unweighted
end

# ## Top level
function (measure::Measure)(args...)
    _check(measure, args...)
    call(measure, args...)
end

# # TRAITS

# user-bespoke measures will subtype `Measure` directly and the
# following will therefore not apply:
StatisticalTraits.supports_weights(::Type{<:Union{Aggregated, Unaggregated}}) = true

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

# info
function StatisticalTraits.info(M::Type{<:Measure})
    values = Tuple(@eval($trait($M)) for trait in MEASURE_TRAITS)
    return NamedTuple{Tuple(MEASURE_TRAITS)}(values)
end

StatisticalTraits.info(m::Measure) = StatisticalTraits.info(typeof(m))


# # AGGREGATION

(::Sum)(v) = sum(skipinvalid(v))
(::Sum)(v::LittleDict) = sum(values(v))

(::Mean)(v) = mean(skipinvalid(v))
(::Mean)(v::LittleDict) = mean(values(v))

(::RootMeanSquare)(v) = sqrt(mean(skipinvalid(v).^2))

aggregate(v, measure) = aggregation(measure)(v)

# aggregation is no-op on scalars:
const MeasureValue = Union{Real,Tuple{<:Real,<:Real}} # number or interval
aggregate(x::MeasureValue, measure) = x


# # UNIVERSAL CALLING SYNTAX

# yhat - predictions (point or probabilisitic)
# X - features
# y - target observations
# w - per-observation weights

function value(measure, yhat, X, y, w)
    vfdep     = Val(is_feature_dependent(measure))
    vsweights = Val(supports_weights(measure) ||
                    supports_class_weights(measure))
    return value(measure, yhat, X, y, w, vfdep, vsweights)
end

# # UNIVERSAL CALLING INTERFACE

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

# # helpers

_scale(x, w::Arr, i) = x*w[i]
_scale(x, ::Nothing, i::Any) = x

function check_pools(ŷ, y)
    levels(y) == levels(ŷ[1]) ||
        error("Conflicting categorical pools found "*
              "in observations and predictions. ")
    return nothing
end

function check_pools(ŷ, w::AbstractDict)
    Set(levels(ŷ[1])) == Set(keys(w)) ||
        error("Conflicting categorical pools found "*
              "in class weights and predictions. ")
    return nothing
end

# # INCLUDE SPECIFIC MEASURES AND TOOLS

include("meta_utilities.jl")
include("roc.jl")
include("confusion_matrix.jl")
include("continuous.jl")
include("finite.jl")
include("probabilistic.jl")
include("loss_functions_interface.jl")


# # DEFAULT MEASURES

default_measure(T, S) = nothing

# Deterministic + Continuous / Count ==> RMS
function default_measure(
    ::Type{<:Deterministic},
    ::Type{<:Union{Vec{<:Union{Missing,Continuous}},
    Vec{<:Union{Missing,Count}}}}
)
   return rms
end

# Deterministic + Finite ==> Misclassification rate
function default_measure(
    ::Type{<:Deterministic},
    ::Type{<:Vec{<:Union{Missing,Finite}}}
)
    return misclassification_rate
end

# Probabilistic + Finite ==> log loss
function default_measure(
    ::Type{<:Probabilistic},
    ::Type{<:Vec{<:Union{Missing,Finite}}}
)
    return log_loss
end

# Probabilistic + Continuous ==> Log loss
function default_measure(
    ::Type{<:Probabilistic},
    ::Type{<:Vec{<:Union{Missing,Continuous}}}
)
    return log_loss
end

# Probabilistic + Count ==> Log score
function default_measure(
    ::Type{<:Probabilistic},
    ::Type{<:Vec{<:Union{Missing, Count}}}
)
    return log_loss
end

function default_measure(
    ::Type{<:MMI.ProbabilisticDetector},
    ::Type{<:Vec{<:Union{Missing,OrderedFactor{2}}}}
)
    return area_under_curve
end

function default_measure(
    ::Type{<:MMI.DeterministicDetector},
    ::Type{<:Vec{<:Union{Missing,OrderedFactor{2}}}}
)
    return balanced_accuracy
end

# Fallbacks
default_measure(M::Type{<:Supervised}) = default_measure(M, target_scitype(M))
default_measure(::M) where M <: Supervised = default_measure(M)

default_measure(M::Type{<:Annotator}) = default_measure(M, target_scitype(M))
default_measure(::M) where M <: Annotator = default_measure(M)
