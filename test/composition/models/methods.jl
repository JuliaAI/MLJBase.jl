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
model = Rubbish(knn, OneHotEncoder(), 42)
X, y = make_regression(10, 2)

@testset "logic for composite model update - fallback()" begin
    Xs = source(X)
    ys = source(y)
    mach0 = machine(Standardizer(), Xs)
    W = transform(mach0, Xs)
    mach1 = machine(model.model_in_network, W, ys)
    yhat = predict(mach1, W)
    mach = machine(Deterministic(), Xs, ys; predict=yhat)
    fitresult, cache, _ = return!(mach, model, 0)
    network_model_names = getfield(fitresult, :network_model_names)
    @test network_model_names == [:model_in_network, nothing]
    old_model = cache.old_model
    glb_node = MLJBase.glb(mach)
    @test !MLJBase.fallback(model, old_model, network_model_names, glb_node)

    # don't fallback if mutating field for a network model:
    model.model_in_network.K = 24
    @test !MLJBase.fallback(model, old_model, network_model_names, glb_node)

    # do fallback if replacing field for a network model:
    model.model_in_network = KNNRegressor()
    @test MLJBase.fallback(model, old_model, network_model_names, glb_node)

    # return to original state:
    model.model_in_network = knn
    @test !MLJBase.fallback(model, old_model, network_model_names, glb_node)

    # do fallback if a non-network field changes:
    model.model_not_in_network.features = [:x1,]
    @test MLJBase.fallback(model, old_model, network_model_names, glb_node)

    # return to original state:
    model.model_not_in_network = OneHotEncoder()
    @test !MLJBase.fallback(model, old_model, network_model_names, glb_node)

    # do fallback if any non-model changes:
    model.some_other_variable = 123412
    @test MLJBase.fallback(model, old_model, network_model_names, glb_node)

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

# `model` is instance of `Rubbish`
mach = fit!(machine(model, X, y), verbosity=0)

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

    # to check internals:
    ridge = MLJBase.machines(fitresult.predict)[1]
    selector = MLJBase.machines(fitresult.predict)[2]
    ridge_old_fitresult = deepcopy(ridge.fitresult)
    selector_old_fitresult = deepcopy(selector.fitresult)

    # this should trigger no retraining:
    fitresult, cache, rep =
        @test_logs(
            (:info, r"^Not"),
            (:info, r"^Not"),
            MLJBase.update(composite, 2, fitresult, cache, Xtrain, ytrain));
    @test ridge.fitresult == ridge_old_fitresult
    @test selector.fitresult == selector_old_fitresult

    # this should trigger update of selector and training of ridge:
    selector_model.features = [:a, :b]
    fitresult, cache, rep =
        @test_logs(
            (:info, r"^Updating"),
            (:info, r"^Training"),
            MLJBase.update(composite, 2, fitresult, cache, Xtrain, ytrain));
    @test ridge.fitresult != ridge_old_fitresult
    @test selector.fitresult != selector_old_fitresult
    ridge_old_fitresult = deepcopy(ridge.fitresult)
    selector_old_fitresult = deepcopy(selector.fitresult)

    # this should trigger updating of ridge only:
    ridge_model.lambda = 1.0
    fitresult, cache, rep =
        @test_logs(
            (:info, r"^Not"),
            (:info, r"^Updating"),
            MLJBase.update(composite, 2, fitresult, cache, Xtrain, ytrain));
    @test ridge.fitresult != ridge_old_fitresult
    @test selector.fitresult == selector_old_fitresult

    predict(composite, fitresult, MLJBase.selectrows(Xin, test));

    Xs = source(Xtrain)
    ys = source(ytrain)

    mach = machine(composite, Xs, ys)
    yhat = predict(mach, Xs)
    fit!(yhat, verbosity=0)
    composite.transformer.features = [:b, :c]
    fit!(yhat, verbosity=0)
    fit!(yhat, rows=1:20, verbosity=0)
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
fit!(mach, verbosity=0)
@test  objectid(mach) == id  # *********
yhat=predict(mach, Xin);
ridge.lambda = 1.0
fit!(mach, verbosity=0)
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
        foo = node(η -> first(η), yhat)
        mach = machine(Unsupervised(),
                       Xs;
                       predict=yhat,
                       transform=Wout,
                       report=(foo=foo,))
        return!(mach, model, verbosity)
    end
    X, _ = make_regression(10, 5);
    model = WrappedDummyClusterer(model=DummyClusterer(n=2))
    mach = fit!(machine(model, X), verbosity=0)
    model.model.n = 3
    fit!(mach, verbosity=0)
    @test transform(mach, X) == selectcols(X, 1:3)
    r = report(mach)
    @test r.model.centres == MLJBase.matrix(X)[1:3,:]
    @test r.foo == predict(mach, rows=:)[1]
    fp = fitted_params(mach)
    @test :model in keys(fp)
    levs = fp.model.fitresult
    @test predict(mach, X) == fill(levs[1], 10)
end


## NETWORK WITH MULTIPLE NODES REPORTING STATE/ REFIT

mutable struct TwoStages <: DeterministicComposite
    model1
    model2
    model3
end

function MLJBase.fit(m::TwoStages, verbosity, X, y)
    Xs = source(X)
    ys = source(y)
    mach1 = machine(m.model1, Xs, ys)
    mach2 = machine(m.model2, Xs, ys)
    ypred1 = MLJBase.predict(mach1, Xs)
    ypred2 = MLJBase.predict(mach2, Xs)
    Y = MLJBase.table(hcat(ypred1, ypred2))
    mach3 = machine(m.model3, Y, ys)
    ypred3 = MLJBase.predict(mach3, Y)
    μpred = node(x->mean(x), ypred3)
    σpred = node((x, μ)->mean((x.-μ).^2), ypred3, μpred)
    mach = machine(Deterministic(),
                   Xs,
                   ys;
                   predict=ypred3,
                   report=(μpred=μpred,
                           σpred=σpred))
    return!(mach, m, verbosity)
end

@testset "Test exported-network with multiple saved nodes and refit" begin
    X, y = make_regression(100, 3)
    model3 = FooBarRegressor(lambda=1)
    twostages = TwoStages(FooBarRegressor(lambda=0.1),
                          FooBarRegressor(lambda=10), model3)
    mach = machine(twostages, X, y)
    fit!(mach, verbosity=0)
    rep = report(mach)
    # All machines have been fitted once
    @test rep.machines[1].state ==
        rep.machines[2].state ==
        rep.machines[3].state == 1
    # Retrieve current values of interest
    μpred = rep.μpred
    σpred = rep.σpred
    # Change model3 and refit
    model3.lambda = 10
    fit!(mach, verbosity=0)
    rep = report(mach)
    # Machines 1,2 have been fitted once and machine 3 twice
    @test rep.machines[1].state == rep.machines[2].state == 1
    @test rep.machines[3].state == 2
    # The new values have been updated
    @test rep.μpred != μpred
    @test rep.σpred != σpred
end

## COMPOSITE WITH COMPONENT MODELS STORED IN NTUPLE

# `modelnames` is a tuple of `Symbol`s, one for each `model` in `models`:
mutable struct Averager{modelnames} <: DeterministicComposite
    models::NTuple{<:Any,Deterministic}
    weights::Vector{Float64}
    Averager(modelnames, models, weights) =
        new{modelnames}(models, weights)
end

# special kw constructor, allowing one to specify the property names
# to be attributed to each component model (see below):
function Averager(; weights=Float64[], named_models...)
        nt = NamedTuple(named_models)
    modelnames = keys(nt)
    models = values(nt)
    return Averager(modelnames, models, weights)
end

# for example:
averager = Averager(weights=[1, 1],
                    model1=KNNRegressor(K=3),
                    model2=RidgeRegressor())

# so we can do `averager.model1` and `averager.model2`:
Base.propertynames(::Averager{modelnames}) where modelnames =
        tuple(:weights, modelnames...)
function Base.getproperty(averager::Averager{modelnames},
                          name::Symbol) where modelnames
    name === :weights && return getfield(averager, :weights)
    models = getfield(averager, :models)
    for j in eachindex(modelnames)
        name === modelnames[j] && return models[j]
    end
    error("type Averager has no field $name")
end

# overload multiplication of a node by a matrix:
import Base.*
*(preds::Node, weights) = node(p->p*weights, preds)

# learning network wrapped in a fit method:
function MLJBase.fit(averager::Averager{modelnames},
                     verbosity,
                     X,
                     y) where modelnames

    Xs = source(X)
    ys = source(y)

    weights = averager.weights

    machines = [machine(getproperty(averager, name), Xs, ys) for
                name in modelnames]
    predictions = hcat([predict(mach, Xs) for mach in machines]...)
    yhat = (1/sum(weights))*(predictions*weights)

    mach = machine(Deterministic(), Xs, ys; predict=yhat)
    return!(mach, averager, verbosity)
end

@testset "composite with component models stored in ntuple" begin
    X, y = make_regression(10, 3);
    mach = machine(averager, X, y)
    fit!(mach, verbosity=0)
    fp = fitted_params(mach)
    @test keys(fp.model1) == (:tree, )
    @test keys(fp.model2) == (:coefficients, :intercept)
    r = report(mach)
    @test r.model1 == NamedTuple()
    @test r.model2 == NamedTuple()
    range(averager, :(model1.K), lower=2, upper=3)
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

@testset "operation nodes that are source nodes" begin

    mutable struct BananaComposite <: UnsupervisedComposite
        stand
    end
    BananaComposite(; stand=Standardizer()) = BananaComposite(stand)

    function MLJBase.fit(model::BananaComposite, verbosity, X)

        Xs = source(X)
        mach1 = machine(model.stand, Xs)
        X2 = transform(mach1, Xs)

        # node for the inverse_transform:

        network_mach = machine(Unsupervised(), Xs, transform=X2, inverse_transform=Xs)
        return!(network_mach, model, verbosity)

    end

    X = (x = Float64[1, 2, 3],)
    mach = machine(BananaComposite(), X)
    fit!(mach, verbosity=0, force=true)
    @test transform(mach, X).x ≈ Float64[-1, 0, 1]
    @test inverse_transform(mach, X) == X

end

end # module
true
