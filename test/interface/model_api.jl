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

mutable struct UnivariateFiniteFitter <: MLJModelInterface.Probabilistic
    alpha::Float64
end
UnivariateFiniteFitter(;alpha=1.0) = UnivariateFiniteFitter(alpha)

@testset "models that fit a distribution" begin
    function MLJModelInterface.fit(model::UnivariateFiniteFitter,
                               verbosity, X, y)

        α = model.alpha
        N = length(y)
        _classes = classes(y)
        d = length(_classes)

        frequency_given_class = Distributions.countmap(y)
        prob_given_class =
            Dict(c => (frequency_given_class[c] + α)/(N + α*d) for c in _classes)

        fitresult = MLJBase.UnivariateFinite(prob_given_class)

        report = (params=Distributions.params(fitresult),)
        cache = nothing

        verbosity > 0 && @info "Fitted a $fitresult"

        return fitresult, cache, report
    end

    MLJModelInterface.predict(model::UnivariateFiniteFitter,
                              fitresult,
                              X) = fitresult


    MLJModelInterface.input_scitype(::Type{<:UnivariateFiniteFitter}) =
        Nothing
    MLJModelInterface.target_scitype(::Type{<:UnivariateFiniteFitter}) =
        AbstractVector{<:Finite}

    y = coerce(collect("aabbccaa"), Multiclass)
    X = nothing
    model = UnivariateFiniteFitter(alpha=0)
    mach = machine(model, X, y)
    fit!(mach, verbosity=0)

    ytest = y[1:3]
    yhat = predict(mach, nothing) # single UnivariateFinite distribution

    @test cross_entropy(fill(yhat, 3), ytest) ≈
        [-log(1/2), -log(1/2), -log(1/4)]

end

end
true
