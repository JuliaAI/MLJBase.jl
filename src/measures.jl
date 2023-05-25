# # DEFAULT MEASURES

default_measure(T, S) = _default_measure(T, nonmissingtype(S))

_default_measure(T, S) = nothing

# Deterministic + Continuous / Count ==> RMS
function _default_measure(
    ::Type{<:Deterministic},
    ::Type{<:Union{AbstractVector{<:Continuous}, AbstractVector{<:Count}}},
)
   return rms
end

# Deterministic + Finite ==> Misclassification rate
function _default_measure(
    ::Type{<:Deterministic},
    ::Type{<:AbstractVector{<:Finite}},
)
    return misclassification_rate
end

# Probabilistic + Finite / Count ==> log loss
function _default_measure(
    ::Type{<:Probabilistic},
    ::Type{<:Union{AbstractVector{<:Finite},AbstractVector{<:Count}}},
)
    return log_loss
end

# Probabilistic + Continuous ==> Log loss
function _default_measure(
    ::Type{<:Probabilistic},
    ::Type{<:AbstractVector{<:Continuous}},
)
    return log_loss
end

function _default_measure(
    ::Type{<:MMI.ProbabilisticDetector},
    ::Type{<:AbstractVector{<:OrderedFactor{2}}},
)
    return area_under_curve
end

function _default_measure(
    ::Type{<:MMI.DeterministicDetector},
    ::Type{<:AbstractVector{<:OrderedFactor{2}}},
)
    return balanced_accuracy
end

# Fallbacks
default_measure(M::Type{<:Supervised}) = default_measure(M, target_scitype(M))
default_measure(::M) where M <: Supervised = default_measure(M)

default_measure(M::Type{<:Annotator}) = _default_measure(M, target_scitype(M))
default_measure(::M) where M <: Annotator = default_measure(M)
