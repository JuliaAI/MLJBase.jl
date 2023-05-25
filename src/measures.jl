# # DEFAULT MEASURES

"""
    default_measure(model)

Return a measure that should work with `model`, or return `nothing` if none can be
reliably inferred.

"""
default_measure(m) = nothing
default_measure(m::Union{Supervised,Annotator}) =
    default_measure(m, nonmissingtype(guess_model_target_observation_scitype(m)))
default_measure(m, S) = nothing
default_measure(::Deterministic, ::Type{<:Union{Continuous,Count}}) = l2
default_measure(::Deterministic, ::Type{<:Finite}) = misclassification_rate
default_measure(::Probabilistic, ::Type{<:Union{Finite,Count}}) = log_loss
default_measure(::Probabilistic, ::Type{<:Continuous}) = log_loss
default_measure(::ProbabilisticDetector, ::Type{<:OrderedFactor{2}}) = area_under_curve
default_measure(::DeterministicDetector, ::Type{<:OrderedFactor{2}}) = balanced_accuracy
