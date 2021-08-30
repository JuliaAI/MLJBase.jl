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

## TRAITS

is_measure_type(::Any) = false

# The following traits, with fallbacks defined in
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
show_compact(::Type{<:Measure}) = true
Base.show(io::IO, m::Measure) = show(io, MIME("text/plain"), m)

# info (see also src/init.jl):
function ScientificTypes.info(M, ::Val{:measure_type})
    values = Tuple(@eval($trait($M)) for trait in MEASURE_TRAITS)
    return NamedTuple{Tuple(MEASURE_TRAITS)}(values)
end
ScientificTypes.info(m, ::Val{:measure}) = info(typeof(m))


## SKIPPING MISSINGS

const ERR_NOTHING_LEFT_TO_AGGREGATE = ErrorException(
    "Trying to aggregrate an empty collection of measurements. Perhaps all "*
    "measuresments are `missing` or `NaN`. ")

@inline function _skipmissing(v)
    ret = skipmissing(v)
    isempty(ret) && throw(ERR_NOTHING_LEFT_TO_AGGREGATE)
    return ret
end

function _skipmissing(yhat, y)
    mask = .!(ismissing.(yhat) .| ismissing.(y))
    return yhat[mask], y[mask]
end

function _skipmissing(yhat, y, w)
    mask = .!(ismissing.(yhat) .| ismissing.(y))
    return yhat[mask], y[mask], w[mask]
end

function _skipmissing(yhat, y, w::Nothing)
    mask = .!(ismissing.(yhat) .| ismissing.(y))
    return yhat[mask], y[mask], nothing
end


## AGGREGATION

(::Sum)(v) = sum(_skipmissing(v))
(::Sum)(v::LittleDict) = sum(values(v))

(::Mean)(v) = mean(_skipmissing(v))
(::Mean)(v::LittleDict) = mean(values(v))

(::RootMeanSquare)(v) = sqrt(mean(_skipmissing(v).^2))

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
    vsweights = Val(supports_weights(measure) ||
                    supports_class_weights(measure))
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

## helpers

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
                ::Type{<:Union{Vec{<:Union{Missing,Continuous}},
                               Vec{<:Union{Missing,Count}}}}) = rms

# Deterministic + Finite ==> Misclassification rate
default_measure(::Type{<:Deterministic},
                ::Type{<:Vec{<:Union{Missing,Finite}}}) = misclassification_rate

# Probabilistic + Finite ==> Cross entropy
default_measure(::Type{<:Probabilistic},
                ::Type{<:Vec{<:Union{Missing,Finite}}}) = cross_entropy

# Probablistic + Continuous ==> Brier score
default_measure(::Type{<:Probabilistic},
                ::Type{<:Vec{<:Union{Missing,Continuous}}}) =
                    continuous_brier_score

# Probablistic + Count ==> Brier score
default_measure(::Type{<:Probabilistic},
                ::Type{<:Vec{<:Union{Missing,Count}}}) = count_brier_score

# Fallbacks
default_measure(M::Type{<:Supervised}) = default_measure(M, target_scitype(M))
default_measure(::M) where M <: Supervised = default_measure(M)
