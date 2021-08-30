module TestMeasures

using MLJBase, Test
import Distributions
using CategoricalArrays
using Statistics
import LossFunctions
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

@testset "show method for measures" begin
    io = IOBuffer()
    for meta in measures()
        m = eval(Meta.parse("$(meta.name)()"))
        show(io, MIME("text/plain"), m)
    end
end

@testset "missing and NaN values in aggregation" begin
    v =[1, 2, missing, 5, NaN]
    @test MLJBase.Sum()(v) == 8
    @test MLJBase.RootMeanSquare()(v) ≈ sqrt((1 + 4 + 25)/3)
    @test_throws MLJBase.ERR_NOTHING_LEFT_TO_AGGREGATE MLJBase.Mean()(Float32[])
end

@testset "skipinvalid" begin
    w = rand(5)
    @test MLJBase.skipinvalid([1, 2, missing, 3, NaN], [missing, 5, 6, 7, 8]) ==
        ([2, 3], [5, 7])
    @test(
        MLJBase.skipinvalid([1, 2, missing, 3, NaN],
                            [missing, 5, 6, 7, 8],
                            w) ==
        ([2, 3], [5, 7], w[[2,4]]))
    @test(
        MLJBase.skipinvalid([1, 2, missing, 3, NaN],
                            [missing, 5, 6, 7, 8],
                            nothing) ==
        ([2, 3], [5, 7], nothing))
end

end
true
