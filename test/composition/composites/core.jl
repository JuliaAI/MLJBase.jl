module TestComposites

using Test
using MLJBase
using ..Models
using CategoricalArrays
import Random.seed!
seed!(1234)

@load KNNRegressor

N = 50
Xin = (a=rand(N), b=rand(N), c=rand(N))
yin = rand(N)

train, test = partition(eachindex(yin), 0.7);
Xtrain = MLJBase.selectrows(Xin, train)
ytrain = yin[train]

ridge_model = FooBarRegressor(lambda=0.1)
selector_model = FeatureSelector()

@testset "first test of hand-exported network" begin
    composite = SimpleDeterministicCompositeModel(model=ridge_model,
                                                  transformer=selector_model)

    fitresult, cache, report = MLJBase.fit(composite, 3, Xtrain, ytrain)

    # to check internals:
    ridge = MLJBase.machines(fitresult)[1]
    selector = MLJBase.machines(fitresult)[2]
    ridge_old = deepcopy(ridge)
    selector_old = deepcopy(selector)

    # this should trigger no retraining:
    fitresult, cache, report =
        @test_logs(
            (:info, r"^Not"),
            (:info, r"^Not"),
            MLJBase.update(composite, 2, fitresult, cache, Xtrain, ytrain))
    @test ridge.fitresult == ridge_old.fitresult
    @test selector.fitresult == selector_old.fitresult

    # this should trigger update of selector and training of ridge:
    selector_model.features = [:a, :b]
    fitresult, cache, report =
        @test_logs(
            (:info, r"^Updating"),
            (:info, r"^Training"),
            MLJBase.update(composite, 2, fitresult, cache, Xtrain, ytrain))
    @test ridge.fitresult != ridge_old.fitresult
    @test selector.fitresult != selector_old.fitresult
    ridge_old = deepcopy(ridge)
    selector_old = deepcopy(selector)

    # this should trigger updating of ridge only:
    ridge_model.lambda = 1.0
    fitresult, cache, report =
        @test_logs(
            (:info, r"^Not"),
            (:info, r"^Updating"),
            MLJBase.update(composite, 2, fitresult, cache, Xtrain, ytrain))
    @test ridge.fitresult != ridge_old.fitresult
    @test selector.fitresult == selector_old.fitresult

    predict(composite, fitresult, MLJBase.selectrows(Xin, test))

    Xs = source(Xtrain)
    ys = source(ytrain, kind=:target)

    mach = machine(composite, Xs, ys)
    yhat = predict(mach, Xs)
    fit!(yhat, verbosity=3)
    composite.transformer.features = [:b, :c]
    fit!(yhat, verbosity=3)
    fit!(yhat, rows=1:20, verbosity=3)
    yhat(MLJBase.selectrows(Xin, test))

end

mutable struct WrappedRidge <: DeterministicNetwork
    ridge
end

@testset "second test of hand-exported network" begin

    function MLJBase.fit(model::WrappedRidge, verbosity::Integer, X, y)
        Xs = source(X)
        ys = source(y, kind=:target)

        stand = Standardizer()
        standM = machine(stand, Xs)
        W = transform(standM, Xs)

        boxcox = UnivariateBoxCoxTransformer()
        boxcoxM = machine(boxcox, ys)
        z = transform(boxcoxM, ys)

        ridgeM = machine(model.ridge, W, z)
        zhat = predict(ridgeM, W)
        yhat = inverse_transform(boxcoxM, zhat)

        fit!(yhat)
        return fitresults(yhat)
    end

    MLJBase.input_scitype(::Type{<:WrappedRidge}) = Table(Continuous)
    MLJBase.target_scitype(::Type{<:WrappedRidge}) = AbstractVector{<:Continuous}

    ridge = FooBarRegressor(lambda=0.1)
    model_ = WrappedRidge(ridge)
    mach = machine(model_, Xin, yin)
    fit!(mach)
    yhat=predict(mach, Xin)
    ridge.lambda = 1.0
    fit!(mach)
    @test predict(mach, Xin) != yhat

end


end
true
