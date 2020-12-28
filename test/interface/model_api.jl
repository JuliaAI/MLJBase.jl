module TestModelAPI

using Test
using MLJBase
import MLJModelInterface
using ..Models
using Distributions
using StableRNGs
using JLSO

rng = StableRNG(661)

@testset "predict_*" begin
    X = rand(rng, 5)
    yfinite = categorical(collect("abaaa"))
    ycont = float.(1:5)

    clf = ConstantClassifier()
    fitresult, _, _ = MLJBase.fit(clf, 1, X, yfinite)
    @test predict_mode(clf, fitresult, X)[1] == 'a'
    @test_throws ArgumentError predict_mean(clf, fitresult, X)
    @test_throws ArgumentError predict_median(clf, fitresult, X)

    rgs = ConstantRegressor()
    fitresult, _, _ = MLJBase.fit(rgs, 1, X, ycont)
    @test predict_mean(rgs, fitresult, X)[1] == 3
    @test predict_median(rgs, fitresult, X)[1] == 3
    @test_throws ArgumentError predict_mode(rgs, fitresult, X)
end

end
true
