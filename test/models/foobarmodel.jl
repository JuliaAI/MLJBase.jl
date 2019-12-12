# a intercept-free ridge regressor for testing meta-alogorithms

export FooBarRegressor

import MLJBase
using LinearAlgebra
using ScientificTypes

mutable struct FooBarRegressor <: MLJBase.Deterministic
    lambda::Float64
end

function FooBarRegressor(; lambda=0.0)
    simpleridgemodel = FooBarRegressor(lambda)
    message = MLJBase.clean!(simpleridgemodel)
    isempty(message) || @warn message
    return simpleridgemodel
end

function MLJBase.clean!(model::FooBarRegressor)
    warning = ""
    if model.lambda < 0
        warning *= "Need lambda ≥ 0. Resetting lambda=0. "
        model.lambda = 0
    end
    return warning
end

function MLJBase.fitted_params(::FooBarRegressor, fitresult)
    return (coefficients=fitresult)
end

function MLJBase.fit(model::FooBarRegressor, verbosity::Int, X, y)
    x = MLJBase.matrix(X)
    fitresult = (x'x - model.lambda*I)\(x'y)
    cache = nothing
    report = NamedTuple()
    return fitresult, cache, report
end


function MLJBase.predict(model::FooBarRegressor, fitresult, Xnew)
    x = MLJBase.matrix(Xnew)
    return x*fitresult
end

# metadata:
MLJBase.load_path(::Type{<:FooBarRegressor}) = "MLJBase.FooBarRegressor"
MLJBase.package_name(::Type{<:FooBarRegressor}) = "MLJBase"
MLJBase.package_uuid(::Type{<:FooBarRegressor}) = ""
MLJBase.is_pure_julia(::Type{<:FooBarRegressor}) = true
MLJBase.input_scitype(::Type{<:FooBarRegressor}) = Table(Continuous)
MLJBase.target_scitype(::Type{<:FooBarRegressor}) = AbstractVector{Continuous}
