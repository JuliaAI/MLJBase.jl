mutable struct DRegressor <: Deterministic end
MLJBase.target_scitype(::Type{<:DRegressor}) =
    AbstractVector{<:Union{Missing,Continuous}}

mutable struct D2Regressor <: Deterministic end
MLJBase.target_scitype(::Type{<:D2Regressor}) =
    AbstractVector{<:Union{Missing,Continuous}}

mutable struct DClassifier <: Deterministic end
MLJBase.target_scitype(::Type{<:DClassifier}) =
    AbstractVector{<:Union{Missing,Finite}}

mutable struct DClassifierWeird <: Deterministic end
MLJBase.target_scitype(::Type{<:DClassifierWeird}) =
    AbstractVector{<:Textual}

mutable struct PClassifier <: Probabilistic end
MLJBase.target_scitype(::Type{<:PClassifier}) =
    AbstractVector{<:Union{Missing,Finite}}

mutable struct PRegressor <: Probabilistic end
MLJBase.target_scitype(::Type{<:PRegressor}) =
    AbstractVector{<:Union{Missing,Continuous}}

mutable struct PCountRegressor <: Probabilistic end
MLJBase.target_scitype(::Type{<:PCountRegressor}) =
    AbstractVector{<:Union{Missing,Count}}



@testset "default_measure" begin
    @test MLJBase.default_measure(DRegressor()) == l2
    @test MLJBase.default_measure(D2Regressor()) == l2
    @test MLJBase.default_measure(DClassifier()) == misclassification_rate
    @test MLJBase.default_measure(PClassifier()) == log_loss
    @test MLJBase.default_measure(PRegressor()) == log_loss
    @test MLJBase.default_measure(PCountRegressor()) == log_loss
    @test isnothing(MLJBase.default_measure(DClassifierWeird()))
    @test isnothing(MLJBase.default_measure("junk"))
end

true
