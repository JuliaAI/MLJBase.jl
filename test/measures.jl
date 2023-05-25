mutable struct DRegressor <: Deterministic end
MLJBase.target_scitype(::Type{<:DRegressor}) =
    AbstractVector{<:Continuous}

mutable struct D2Regressor <: Deterministic end
MLJBase.target_scitype(::Type{<:D2Regressor}) =
    AbstractVector{Continuous}

mutable struct DClassifier <: Deterministic end
MLJBase.target_scitype(::Type{<:DClassifier}) =
    AbstractVector{<:Finite}

mutable struct PClassifier <: Probabilistic end
MLJBase.target_scitype(::Type{<:PClassifier}) =
    AbstractVector{<:Finite}

mutable struct PRegressor <: Probabilistic end
MLJBase.target_scitype(::Type{<:PRegressor}) =
    AbstractVector{<:Continuous}

mutable struct PCountRegressor <: Probabilistic end
MLJBase.target_scitype(::Type{<:PCountRegressor}) =
    AbstractVector{<:Count}

@testset "default_measure" begin
    @test MLJBase.default_measure(DRegressor()) == rms
    @test MLJBase.default_measure(D2Regressor()) == rms
    @test MLJBase.default_measure(DClassifier()) == misclassification_rate
    @test MLJBase.default_measure(PClassifier()) == log_loss

    @test MLJBase.default_measure(DRegressor) == rms
    @test MLJBase.default_measure(D2Regressor) == rms
    @test MLJBase.default_measure(DClassifier) == misclassification_rate
    @test MLJBase.default_measure(PClassifier) == log_loss

    @test MLJBase.default_measure(PRegressor) == log_loss
    @test MLJBase.default_measure(PCountRegressor) == log_loss
end

true
