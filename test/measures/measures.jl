module TestMeasures

using MLJBase, Test
import Distributions
using CategoricalArrays
using Statistics
import LossFunctions
using StableRNGs
using OrderedCollections: LittleDict

rng  = StableRNGs.StableRNG(123)

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

    @test MLJBase.distribution_type(auc) == MLJBase.UnivariateFinite
end

@testset "MLJBase.value" begin
    yhat = randn(rng,5)
    X = (weight=randn(rng,5), x1 = randn(rng,5))
    y = randn(rng,5)
    w = randn(rng,5)

    @test MLJBase.value(mae, yhat, nothing, y, nothing) ≈ mae(yhat, y)
    @test MLJBase.value(mae, yhat, nothing, y, w) ≈ mae(yhat, y, w)

    spooky(yhat, y) = abs.(yhat - y) |> mean
    @test MLJBase.value(spooky, yhat, nothing, y, nothing) ≈ mae(yhat, y)

    cool(yhat, y, w) = abs.(yhat - y) .* w |> mean
    MLJBase.supports_weights(::Type{typeof(cool)}) = true
    @test MLJBase.value(cool, yhat, nothing, y, w) ≈ mae(yhat, y, w)

    funky(yhat, X, y) = X.weight .* abs.(yhat - y) |> mean
    MLJBase.is_feature_dependent(::Type{typeof(funky)}) = true
    @test MLJBase.value(funky, yhat, X, y, nothing) ≈ mae(yhat, y, X.weight)

    weird(yhat, X, y, w) = w .* X.weight .* abs.(yhat - y) |> mean
    MLJBase.is_feature_dependent(::Type{typeof(weird)}) = true
    MLJBase.supports_weights(::Type{typeof(weird)}) = true
    @test MLJBase.value(weird, yhat, X, y, w) ≈ mae(yhat, y, X.weight .* w)
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

include("confusion_matrix.jl")
include("roc.jl")
include("continuous.jl")
include("finite.jl")
include("probabilistic.jl")
include("loss_functions_interface.jl")

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

end
true
