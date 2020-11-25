module TestMeasures

using MLJBase, Test
import Distributions
using CategoricalArrays
using Statistics
using LossFunctions
using StableRNGs
using OrderedCollections: LittleDict

@testset "aggregation" begin
    v = rand(5)
    @test aggregate(v, mae) ≈ mean(v)
    @test aggregate(v, TruePositive()) ≈ sum(v)
    @test aggregate(v, rms) ≈ sqrt(mean(v.^2))
    λ = rand()
    @test aggregate(λ, rms) === λ
    @test aggregate(aggregate(v, l2), l2) == aggregate(v, l2)
    m = LittleDict([0, 1, 2, 3, 4], v)
    @test aggregate(m, MTPR()) == mean(v)
end

@testset "metadata" begin
    measures()
    measures(m -> m.target_scitype <: AbstractVector{<:Finite} &&
             m.supports_weights)
    info(rms)
    @test true
end

@testset "coverage" begin
    # just checking that the traits work not that they're correct
    @test orientation(BrierScore()) == :score
    @test orientation(auc) == :score
    @test orientation(rms) == :loss

    @test reports_each_observation(auc) == false
    @test is_feature_dependent(auc) == false

    @test MLJBase.distribution_type(BrierScore) == MLJBase.UnivariateFinite
end

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

@testset "default_measure" begin
    @test MLJBase.default_measure(DRegressor()) == rms
    @test MLJBase.default_measure(D2Regressor()) == rms
    @test MLJBase.default_measure(DClassifier()) == misclassification_rate
    @test MLJBase.default_measure(PClassifier()) == cross_entropy

    @test MLJBase.default_measure(DRegressor) == rms
    @test MLJBase.default_measure(D2Regressor) == rms
    @test MLJBase.default_measure(DClassifier) == misclassification_rate
    @test MLJBase.default_measure(PClassifier) == cross_entropy
end

include("continuous.jl")
include("finite.jl")
include("loss_functions_interface.jl")
include("confusion_matrix.jl")

end
true
