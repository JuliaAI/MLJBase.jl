const PROPER_SCORING_RULES = "[Gneiting and Raftery (2007), \"Strictly"*
    "Proper Scoring Rules, Prediction, and Estimation\""*
    "](https://doi.org/10.1198/016214506000001437)"
const DOC_FINITE =
    "`AbstractArray{<:Finite}` (multiclass classification)"
const DOC_FINITE_BINARY =
    "`AbstractArray{<:Finite{2}}` (binary classification)"
const DOC_ORDERED_FACTOR =
    "`AbstractArray{<:OrderedFactor}` (classification of ordered target)"
const DOC_ORDERED_FACTOR_BINARY =
    "`AbstractArray{<:OrderedFactor{2}}` "*
    "(binary classification where choice of \"true\" effects the measure)"
const DOC_CONTINUOUS = "`AbstractArray{Continuous}` (regression)"
const DOC_COUNT = "`AbstractArray{Count}`"
const DOC_INFINITE = "AbstractArray{<:Infinite}"
const INVARIANT_LABEL =
    "This metric is invariant to class reordering."
const VARIANT_LABEL =
    "This metric is *not* invariant to class re-ordering"

is_measure_type(::Any) = false

# Each of the following traits, with fallbacks defined in
# StatisticalTraits.jl, make sense for some or all measures:

const MEASURE_TRAITS = [:name,
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
                        :distribution_type]

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
function _check(measure::Measure,
                yhat::Arr{<:UnivariateFinite},
                y,
                w::Arr)
    check_dimensions(yhat, y)
    check_pools(yhat, y)
    extra_check(measure, yhat, y, w)
end
function _check(measure::Measure,
                yhat::Arr{<:UnivariateFinite},
                y,
                w::AbstractDict)
    check_dimensions(yhat, y)
    check_pools(yhat, y)
    check_pools(yhat, w)
    extra_check(measure, yhat, y, w)
end


# # METHODS TO EVALUATE MEASURES

# See measures/README.md for details

single(::Unaggregated, η̂::Missing, η)          = missing
single(::Unaggregated, η̂,          η::Missing) = missing

const Label = Union{CategoricalValue,Number,AbstractString,Symbol,AbstractChar}

# closure for broadcasting:
single(measure::Measure) = (ηhat, η) -> single(measure, ηhat, η)

call(measure::Unaggregated, yhat, y) = broadcast(single(measure), yhat, y)
function call(measure::Unaggregated, yhat, y, w::Arr)
    unweighted = broadcast(single(measure), yhat, y) # `single` closure below
    return w .* unweighted
end
function call(measure::Unaggregated, yhat, y, weight_given_class::AbstractDict)
    unweighted = broadcast(single(measure), yhat, y) # `single` closure below
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
StatisticalTraits.supports_weights(::Type{<:Union{Aggregated,Unaggregated}}) =
    true

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
show_compact(::Type{<:Measure}) = true
Base.show(io::IO, m::Measure) = show(io, MIME("text/plain"), m)

# info (see also src/init.jl):
function ScientificTypes.info(M, ::Val{:measure_type})
    values = Tuple(@eval($trait($M)) for trait in MEASURE_TRAITS)
    return NamedTuple{Tuple(MEASURE_TRAITS)}(values)
end
ScientificTypes.info(m, ::Val{:measure}) = info(typeof(m))


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
default_measure(::Type{<:Deterministic},
                ::Type{<:Union{Vec{<:Union{Missing,Continuous}},
                               Vec{<:Union{Missing,Count}}}}) = rms

# Deterministic + Finite ==> Misclassification rate
default_measure(::Type{<:Deterministic},
                ::Type{<:Vec{<:Union{Missing,Finite}}}) = misclassification_rate

# Probabilistic + Finite ==> log loss
default_measure(::Type{<:Probabilistic},
                ::Type{<:Vec{<:Union{Missing,Finite}}}) = log_loss

# Probablistic + Continuous ==> Log loss
default_measure(::Type{<:Probabilistic},
                ::Type{<:Vec{<:Union{Missing,Continuous}}}) = log_loss


# Probablistic + Count ==> Log score
default_measure(::Type{<:Probabilistic},
                ::Type{<:Vec{<:Union{Missing,Count}}}) = log_loss

# Fallbacks
default_measure(M::Type{<:Supervised}) = default_measure(M, target_scitype(M))
default_measure(::M) where M <: Supervised = default_measure(M)
