module TestCompositesCore

using Test
using MLJBase
using Tables
import MLJBase
using ..Models
using CategoricalArrays
using OrderedCollections
import Random.seed!
seed!(1234)

# @testset "anonymize!" begin
#     ss  = [source(1), source(2), source(3)]
#     a = MLJBase.anonymize!(ss)
#     @test all(s -> s.data === nothing, a.sources)
#     @test a.data == (1, 2, 3)
# end

@load KNNRegressor

N = 50
Xin = (a=rand(N), b=rand(N), c=rand(N));
yin = rand(N);

train, test = partition(eachindex(yin), 0.7);
Xtrain = MLJBase.selectrows(Xin, train);
ytrain = yin[train];

ridge_model = FooBarRegressor(lambda=0.1)
selector_model = FeatureSelector()

@testset "first test of hand-exported network" begin
    composite = SimpleDeterministicCompositeModel(model=ridge_model,
                                                  transformer=selector_model)

    fitresult, cache, rep = MLJBase.fit(composite, 0, Xtrain, ytrain);

    # test data anonymity:
    @test isempty(sources(fitresult[1])[1])

    # to check internals:
    ridge = MLJBase.machines(fitresult.predict)[1]
    selector = MLJBase.machines(fitresult.predict)[2]
    ridge_old = deepcopy(ridge)
    selector_old = deepcopy(selector)

    # this should trigger no retraining:
    fitresult, cache, rep =
        @test_logs(
            (:info, r"^Not"),
            (:info, r"^Not"),
            MLJBase.update(composite, 2, fitresult, cache, Xtrain, ytrain));
    @test ridge.fitresult == ridge_old.fitresult
    @test selector.fitresult == selector_old.fitresult

    # this should trigger update of selector and training of ridge:
    selector_model.features = [:a, :b]
    fitresult, cache, rep =
        @test_logs(
            (:info, r"^Updating"),
            (:info, r"^Training"),
            MLJBase.update(composite, 2, fitresult, cache, Xtrain, ytrain));
    @test ridge.fitresult != ridge_old.fitresult
    @test selector.fitresult != selector_old.fitresult
    ridge_old = deepcopy(ridge)
    selector_old = deepcopy(selector)

    # this should trigger updating of ridge only:
    ridge_model.lambda = 1.0
    fitresult, cache, rep =
        @test_logs(
            (:info, r"^Not"),
            (:info, r"^Updating"),
            MLJBase.update(composite, 2, fitresult, cache, Xtrain, ytrain));
    @test ridge.fitresult != ridge_old.fitresult
    @test selector.fitresult == selector_old.fitresult

    predict(composite, fitresult, MLJBase.selectrows(Xin, test));

    Xs = source(Xtrain)
    ys = source(ytrain)

    mach = machine(composite, Xs, ys)
    yhat = predict(mach, Xs)
    fit!(yhat, verbosity=3)
    composite.transformer.features = [:b, :c]
    fit!(yhat, verbosity=3)
    fit!(yhat, rows=1:20, verbosity=3)
    yhat(MLJBase.selectrows(Xin, test));

end

mutable struct WrappedRidge <: DeterministicComposite
    ridge
end

# julia bug? If I return the following test to a @testset block, then
# the test marked with ******* fails (bizarre!)
#@testset "second test of hand-exported network" begin
    function MLJBase.fit(model::WrappedRidge, verbosity::Integer, X, y)
        Xs = source(X)
        ys = source(y)

        stand = Standardizer()
        standM = machine(stand, Xs)
        W = transform(standM, Xs)

        boxcox = UnivariateBoxCoxTransformer()
        boxcoxM = machine(boxcox, ys)
        z = transform(boxcoxM, ys)

        ridgeM = machine(model.ridge, W, z)
        zhat = predict(ridgeM, W)
        yhat = inverse_transform(boxcoxM, zhat)

        mach = machine(Deterministic(), Xs, ys; predict=yhat)
        fit!(mach, verbosity=verbosity)
        return!(mach, model)
    end

    MLJBase.input_scitype(::Type{<:WrappedRidge}) =
        Table(Continuous)
    MLJBase.target_scitype(::Type{<:WrappedRidge}) =
        AbstractVector{<:Continuous}

    ridge = FooBarRegressor(lambda=0.1)
    model_ = WrappedRidge(ridge)
    mach = machine(model_, Xin, yin)
    id = objectid(mach)
    fit!(mach)
    @test  objectid(mach) == id  # *********
    yhat=predict(mach, Xin);
    ridge.lambda = 1.0
    fit!(mach)
    @test predict(mach, Xin) != yhat

    # test depreciated version:
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
    ridge = FooBarRegressor(lambda=0.1)
    model_ = WrappedRidge(ridge)
    mach = machine(model_, Xin, yin)
    @test_deprecated fit!(mach)
    @test yhat ≈ predict(mach, Xin);

#end

# A dummy clustering model:
mutable struct DummyClusterer <: Unsupervised
    n::Int
end
DummyClusterer(; n=3) = DummyClusterer(n)
function MLJBase.fit(model::DummyClusterer, verbosity::Int, X)
    Xmatrix = Tables.matrix(X)
    n = min(size(Xmatrix, 2), model.n)
    centres = Xmatrix[1:n, :]
    levels = categorical(1:n)
    report = (centres=centres,)
    fitresult = levels
    return fitresult, nothing, report
end
MLJBase.transform(model::DummyClusterer, fitresult, Xnew) =
    selectcols(Xnew, 1:length(fitresult))
MLJBase.predict(model::DummyClusterer, fitresult, Xnew) =
    [fill(fitresult[1], nrows(Xnew))...]

# A wrap of above model:
mutable struct WrappedDummyClusterer <: UnsupervisedComposite
    model
end
WrappedDummyClusterer(; model=DummyClusterer()) =
    WrappedDummyClusterer(model)

@testset "third test of hand-exported network" begin
    function MLJBase.fit(model::WrappedDummyClusterer, verbosity::Int, X)
        Xs = source(X)
        W = transform(machine(OneHotEncoder(), Xs), Xs)
        m = machine(model.model, W)
        yhat = predict(m, W)
        Wout = transform(m, W)
        mach = machine(Unsupervised(), Xs; predict=yhat, transform=Wout)
        fit!(mach)
        return!(mach, model)
    end
    X, _ = make_regression(10, 5);
    model = WrappedDummyClusterer(model=DummyClusterer(n=2))
    mach = machine(model, X) |> fit!
    model.model.n = 3
    fit!(mach)
    @test transform(mach, X) == selectcols(X, 1:3)
    r = report(mach)
    @test r.model.centres == MLJBase.matrix(X)[1:3,:]
    fp = fitted_params(mach)
    levs = fp.model.fitresult
    @test predict(mach, X) == fill(levs[1], 10)
end



end
true
