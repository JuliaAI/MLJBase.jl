module DefaultMeasuresExt

using MLJBase
import MLJBase:default_measure, ProbabilisticDetector, DeterministicDetector
using StatisticalMeasures
using StatisticalMeasures.ScientificTypesBase

default_measure(::Deterministic, ::Type{<:Union{Continuous,Count}}) = l2
default_measure(::Deterministic, ::Type{<:Finite}) = misclassification_rate
default_measure(::Probabilistic, ::Type{<:Union{Finite,Count}}) = log_loss
default_measure(::Probabilistic, ::Type{<:Continuous}) = log_loss
default_measure(::ProbabilisticDetector, ::Type{<:OrderedFactor{2}}) = area_under_curve
default_measure(::DeterministicDetector, ::Type{<:OrderedFactor{2}}) = balanced_accuracy

end # module
