# implementation of MLJ measure interface for LossFunctions.jl

# Supervised Loss -- measure traits
const LSF = LossFunctions
is_measure_type(::Type{<:SupervisedLoss})          = true
orientation(::Type{<:SupervisedLoss})              = :loss
reports_each_observation(::Type{<:SupervisedLoss}) = true
is_feature_dependent(::Type{<:SupervisedLoss})     = false

MMI.supports_weights(::Type{<:SupervisedLoss}) = true
MMI.name(M::Type{<:SupervisedLoss})            = split(string(M), '.')[end]*"()"
MMI.docstring(M::Type{<:SupervisedLoss})       = name(M)

## DISTANCE BASED LOSS FUNCTION

MMI.prediction_type(::Type{<:DistanceLoss}) = :deterministic
MMI.target_scitype(::Type{<:DistanceLoss})  = AbstractArray{<:Continuous}

function value(measure::DistanceLoss, yhat, X, y, ::Nothing,
                ::Val{false}, ::Val{true})
    return measure(yhat, y)
end

function value(measure::DistanceLoss, yhat, X, y, w,
                ::Val{false}, ::Val{true})
    return w .* measure(yhat, y) ./ (sum(w)/length(y))
end

## MARGIN BASED LOSS FUNCTIONS

MMI.prediction_type(::Type{<:MarginLoss}) = :probabilistic
MMI.target_scitype(::Type{<:MarginLoss})  = AbstractArray{<:Binary}

# convert a Binary vector into vector of +1 or -1 values
# (for testing only):
pm1(y) = Int8(2) .* (Int8.(int(y))) .- Int8(3)

# rescale [0, 1] -> [-1, 1]
_scale(p) = 2p - 1

function value(measure::MarginLoss, yhat, X, y, ::Nothing,
                ::Val{false}, ::Val{true})
    check_pools(yhat, y)
    probs_of_observed = broadcast(pdf, yhat, y)
    return value.(measure, _scale.(probs_of_observed), 1)
end

function value(measure::MarginLoss, yhat, X, y, w,
                ::Val{false}, ::Val{true})
    return w .* value(measure, yhat, X, y, nothing) ./ (sum(w)/length(y))
end
