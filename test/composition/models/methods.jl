module TestCompositesCore

using Test
using MLJBase
using Tables
import MLJBase
using ..Models
using ..TestUtilities
using CategoricalArrays
using OrderedCollections
import Random.seed!
seed!(1234)

mutable struct Rubbish <: DeterministicComposite
    model_in_network
    model_not_in_network
    some_other_variable
end

knn = KNNRegressor()
model = Rubbish(knn, Standardizer(), 42)
X, y = make_regression(10, 2)

@testset "logic for composite model update - fallback()" begin
    Xs = source(X)
    ys = source(y)
    mach1 = machine(model.model_in_network, Xs, ys)
    yhat = predict(mach1, Xs)
    mach = machine(Deterministic(), Xs, ys; predict=yhat)
    _, cache, _ = return!(mach, model, 1)
    old_model = cache.old_model
    network_model_fields = cache.network_model_fields
    glb_node = MLJBase.glb(mach)
    @test !MLJBase.fallback(model, old_model, network_model_fields, glb_node)

    # don't fallback if mutating field for a network model:
    model.model_in_network.K = 24
    @test !MLJBase.fallback(model, old_model, network_model_fields, glb_node)

    # do fallback if replacing field for a network model:
    model.model_in_network = KNNRegressor()
    @test MLJBase.fallback(model, old_model, network_model_fields, glb_node)

    # return to original state:
    model.model_in_network = knn
    @test !MLJBase.fallback(model, old_model, network_model_fields, glb_node)

    # do fallback if a non-network field changes:
    model.model_not_in_network.features = [:x1,]
    @test MLJBase.fallback(model, old_model, network_model_fields, glb_node)

    # return to original state:
    model.model_not_in_network = Standardizer()
    @test !MLJBase.fallback(model, old_model, network_model_fields, glb_node)

    # do fallback if any non-model changes:
    model.some_other_variable = 123412
    @test MLJBase.fallback(model, old_model, network_model_fields, glb_node)

end

# deprecated version:
function MLJBase.fit(model::Rubbish, verbosity, X, y)
    Xs = source(X)
    ys = source(y)
    mach1 = machine(model.model_in_network, Xs, ys)
    yhat = predict(mach1, Xs)
    mach = machine(Deterministic(), Xs, ys; predict=yhat)
    fit!(mach, verbosity=verbosity)
    return mach()
end

@testset "deprecation warning" begin
    Xs = source(X)
    ys = source(y)
    mach1 = machine(model.model_in_network, Xs, ys)
    yhat = predict(mach1, Xs)
    mach = machine(Deterministic(), Xs, ys; predict=yhat)
    fit!(mach, verbosity=-1)
    @test_deprecated mach();

    mach = machine(model, X, y)
    @test_deprecated fit!(mach, verbosity=-1)
end

model = Rubbish(KNNRegressor(), Standardizer(), 42)

function MLJBase.fit(model::Rubbish, verbosity, X, y)
    Xs = source(X)
    ys = source(y)
    mach1 = machine(model.model_in_network, Xs, ys)
    yhat = predict(mach1, Xs)
    mach = machine(Deterministic(), Xs, ys; predict=yhat)
    return!(mach, model, verbosity)
end

mach = machine(model, X, y) |> fit! # `model` is instance of `Rubbish`

@testset "logic for composite model update - fit!" begin

    # immediately refit:
    @test_model_sequence(fit!(mach), [(:skip, model), ])

    # mutate a field for a network model:
    model.model_in_network.K = 24
    @test_model_sequence(fit!(mach),
                         [(:update, model), (:update, model.model_in_network)])

    # immediately refit:
    @test_model_sequence(fit!(mach), [(:skip, model), ])

    # replace a field for a network model:
    model.model_in_network = KNNRegressor()
    @test_model_sequence(fit!(mach),
                         [(:update, model), (:train, model.model_in_network)])

    # immediately refit:
    @test_model_sequence(fit!(mach), [(:skip, model), ])

    # mutate a field for a model not in network:
    model.model_not_in_network.features = [:x1,]
    @test_model_sequence(fit!(mach),
                         [(:update, model), (:train, model.model_in_network)])

    # immediately refit:
    @test_model_sequence(fit!(mach), [(:skip, model), ])

    # mutate some field that is not a model:
    model.some_other_variable = 123412
    @test_model_sequence(fit!(mach),
                         [(:update, model), (:train, model.model_in_network)])
end

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
        return!(mach, model, verbosity)
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
        return!(mach, model, verbosity)
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


## DATA FRONT-END IN AN EXPORTED LEARNING NETWORK

mutable struct Scale <: MLJBase.Static
    scaling::Float64
end

function MLJBase.transform(s::Scale, _, X)
    X isa AbstractVecOrMat && return X * s.scaling
    MLJBase.table(s.scaling * MLJBase.matrix(X), prototype=X)
end

function MLJBase.inverse_transform(s::Scale, _, X)
    X isa AbstractVecOrMat && return X / s.scaling
    MLJBase.table(MLJBase.matrix(X) / s.scaling, prototype=X)
end

mutable struct ElephantModel <: ProbabilisticComposite
    scaler
    clf
    cache::Bool
end

function MLJBase.fit(model::ElephantModel, verbosity, X, y)

    Xs = source(X)
    ys = source(y)

    scaler = model.scaler
    mach1 = machine(scaler, cache=model.cache)
    W = transform(mach1, Xs)

    # a classifier with reformat front-end:
    clf = model.clf
    mach2 = machine(clf, W, ys, cache=model.cache)
    yhat = predict(mach2, W)

    mach = machine(Probabilistic(), Xs, ys, predict=yhat)
    return!(mach, model, verbosity)
end

@testset "reformat/selectrows logic in composite model" begin

    X = (x1=ones(5), x2=ones(5))
    y = categorical(collect("abaaa"))
    model = ElephantModel(Scale(2.0),
                        ConstantClassifier(testing=true, bogus=1.0),
                        true)
    mach = machine(model, X, y, cache=false)

    @test_logs((:info, "reformatting X, y"),
               (:info, "resampling X, y"),
               fit!(mach, verbosity=0, rows=1:3)
               )
    @test mach.state == 1

    # new clf hyperparmater (same rows) means no reformatting or resampling:
    model.clf.bogus = 10
    @test_logs fit!(mach, verbosity=0, rows=1:3)
    @test mach.state == 2

    # however changing an upstream hyperparameter forces reformatting
    # and resampling:
    model.scaler.scaling = 3.1
    @test_logs((:info, "reformatting X, y"),
               (:info, "resampling X, y"),
               fit!(mach, verbosity=0, rows=1:3))

end

end # module
true
