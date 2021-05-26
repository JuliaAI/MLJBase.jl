# implementation of MLJ measure interface for LossFunctions.jl

naked(T::Type) = split(string(T), '.') |> last |> Symbol

const MARGIN_LOSSES =
    [:DWDMarginLoss, :ExpLoss, :L1HingeLoss, :L2HingeLoss, :L2MarginLoss,
     :LogitMarginLoss, :ModifiedHuberLoss, :PerceptronLoss, :SigmoidLoss,
     :SmoothedL1HingeLoss, :ZeroOneLoss]
const DISTANCE_LOSSES =
    [:HuberLoss, :L1EpsilonInsLoss, :L2EpsilonInsLoss, :LPDistLoss,
     :LogitDistLoss, :PeriodicLoss, :QuantileLoss]

# Supervised Loss -- measure traits
const LSF = LossFunctions
is_measure_type(::Type{<:SupervisedLoss})          = true
orientation(::Type{<:SupervisedLoss})              = :loss
reports_each_observation(::Type{<:SupervisedLoss}) = true
is_feature_dependent(::Type{<:SupervisedLoss})     = false

MMI.supports_weights(::Type{<:SupervisedLoss}) = true
MMI.docstring(M::Type{<:SupervisedLoss})       = name(M)
instances(M::Type{<:SupervisedLoss}) = [snakecase(string.(naked(M))), ]


## DISTANCE BASED LOSS FUNCTION

MMI.prediction_type(::Type{<:DistanceLoss}) = :deterministic
MMI.target_scitype(::Type{<:DistanceLoss}) = Union{Vec{Continuous},Vec{Count}}

function value(measure::DistanceLoss, yhat, X, y, ::Nothing,
                ::Val{false}, ::Val{true})
    return LSF.value(measure, yhat, y)
end

function value(measure::DistanceLoss, yhat, X, y, w,
                ::Val{false}, ::Val{true})
    return w .* value(measure, yhat, X, y, nothing) ./ (sum(w)/length(y))
end


## MARGIN BASED LOSS FUNCTIONS

MMI.prediction_type(::Type{<:MarginLoss}) = :probabilistic
MMI.target_scitype(::Type{<:MarginLoss})  = AbstractArray{<:Finite{2}}

# rescale [0, 1] -> [-1, 1]:
_scale(p) = 2p - 1

function value(measure::MarginLoss, yhat, X, y, ::Nothing,
                ::Val{false}, ::Val{true})
    check_pools(yhat, y)
    probs_of_observed = broadcast(pdf, yhat, y)
    return (LSF.value).(measure, _scale.(probs_of_observed), 1)
end

function value(measure::MarginLoss, yhat, X, y, w,
                ::Val{false}, ::Val{true})
    return w .* value(measure, yhat, X, y, nothing) ./ (sum(w)/length(y))
end


## KEYWORD CONSTRUCTORS

# distance:
DWDMarginLoss(; q=1.0)         = LossFunctions.DWDMarginLoss(q)
SmoothedL1HingeLoss(; γ=1.0) = LossFunctions.SmoothedL1HingeLoss(γ)

# margin:
HuberLoss(x...; d=1.0)        = LossFunctions.HuberLoss(d)
L1EpsilonInsLoss(; ϵ=1.0) = LossFunctions.L1EpsilonInsLoss(ϵ)
L2EpsilonInsLoss(; ϵ=1.0) = LossFunctions.L2EpsilonInsLoss(ϵ)
LPDistLoss(; P=2)         = LossFunctions.LPDistLoss(P)
QuantileLoss(; τ=0.7)     = LossFunctions.QuantileLoss(τ)


## ADJUSTMENTS

human_name(::Type{<:L1EpsilonInsLoss}) = "l1 ϵ-insensitive loss"
human_name(::Type{<:L2EpsilonInsLoss}) = "l2 ϵ-insensitive loss"
human_name(::Type{<:DWDMarginLoss}) = "distance weighted discrimination loss"

_signature(::Any) = ""
_signature(::Type{<:DWDMarginLoss}) = "`DWDMarginLoss(; q=1.0)`"
_signature(::Type{<:SmoothedL1HingeLoss}) = "`SmoothedL1HingeLoss(; γ=1.0)`"
_signature(::Type{<:L1EpsilonInsLoss}) = "`L1EpsilonInsLoss(; ϵ=1.0)`"
_signature(::Type{<:L2EpsilonInsLoss}) = "`L2EpsilonInsLoss(; ϵ=1.0)`"
_signature(::Type{<:LPDistLoss}) = "`LPDistLoss(; P=2)`"
_signature(::Type{<:QuantileLoss}) = "`QuantileLoss(; τ=0.7)`"


## ALIASES AND DOCSTRINGS

for M_ex in DISTANCE_LOSSES
    eval(quote
         sig = _signature($M_ex)
         isempty(sig) || (sig = "Constructor signature: "*sig)
         @create_aliases $M_ex
         @create_docs($M_ex,
            body="See above for original LossFunctions.jl documentation. ",
            footer=sig)
         end)
end

for M_ex in MARGIN_LOSSES
    eval(quote
         sig = _signature($M_ex)
         isempty(sig) || (sig = "Constructor signature: "*sig)
         @create_aliases $M_ex
         @create_docs($M_ex,
            body="See above for original LossFunctions.jl documentation. ",
            scitype=DOC_FINITE_BINARY,
            footer= sig)
         end)
end

## CALLING BEHAVIOUR

for M in vcat(subtypes(DistanceLoss), subtypes(MarginLoss))
    (m::M)(yhat, y) = MLJBase.value(m, yhat, nothing, y, nothing)
    (m::M)(yhat, y, w) = MLJBase.value(m, yhat, nothing, y, w)
end
