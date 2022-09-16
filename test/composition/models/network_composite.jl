module TestNetowrkComposite

using Test
using MLJBase
using ..Models
using ..TestUtilities
using StableRNGs
using Tables
using MLJModelInterface
using CategoricalArrays
using OrderedCollections

const MMI = MLJModelInterface

const rng = StableRNG(123)

X, _ = make_moons(10, rng=StableRNG(123))
Xnew = selectrows(X, 5:10)

# # DUMMY CLUSTERER THAT DOES NOT GENERALIZE TO NEW DATA

mutable struct OneShotClusterer <: Static
    nclusters
    rng
end

function MMI.predict(model::OneShotClusterer, ::Nothing, X)
    rng = copy(model.rng)
    Xmat = Tables.matrix(X)
    labels = map(i -> Char(64 + i), 1:model.nclusters)
    Xout =  categorical(rand(rng, labels, size(Xmat, 1)); levels=labels)
    report = (; labels)
    return Xout, report
end

MMI.reporting_operations(::OneShotClusterer) = (:predict,)


# # DUMMY STATIC TRANSFORMER THAT REPORTS

mutable struct ReportingScaler <: Static
    alpha::Float64
end

MLJBase.reporting_operations(::Type{<:ReportingScaler}) = (:transform, )

MLJBase.transform(model::ReportingScaler, _, X) = (
    model.alpha*Tables.matrix(X),
    (; nrows = size(MLJBase.matrix(X))[1]),
)

# # ANOTHER DUMMY STATIC TRANSFORMER THAT REPORTS

mutable struct ReportingClassSwapper <: Static
    shift::Int64
end

MLJBase.reporting_operations(::Type{<:ReportingClassSwapper}) = (:transform, )

MLJBase.transform(model::ReportingClassSwapper, _, y) = (
    MLJBase.int.(y) .+ model.shift,
    (nrows = length(y),)
)


# # TESTS

mutable struct WatermelonComposite <: UnsupervisedNetworkComposite
    scaler
    clusterer
    classifier1
    classifier2
    mix::Float64
    finalizer
end

function MLJBase.prefit(composite::WatermelonComposite, verbosity, X)

    Xs = source(X)

    len = node(Xs) do X
        size(Tables.matrix(X), 1)
    end

    W = transform(machine(:scaler), Xs)
    ytrain = predict(machine(:clusterer), Xs)

    mach1 = machine(:classifier1, W, ytrain)

    # two machines pointing to same model:
    mach2a = machine(:classifier2, W, ytrain)
    mach2b = machine(:classifier2, W, ytrain)

    y1 = predict(mach1, W) # probabilistic predictions
    y2a = predict(mach2a, W) # probabilistic predictions
    y2b = predict(mach2b, W) # probabilistic predictions

    training_loss = node(
        (y1, y2) -> brier_loss(y1, mode.(y2)) + brier_loss(y2, mode.(y1)) |> mean,
        y1,
        y2a,
    )

    λ = composite.mix
    ymix = λ*y1 + (1 - λ)*(0.2*y2a + 0.8*y2b)
    yhat = transform(machine(:finalizer), mode(ymix))

    return (; predict=yhat, report=(;training_loss, len))

end

composite = WatermelonComposite(
    ReportingScaler(3.0),
    OneShotClusterer(3, StableRNG(123)),
    DecisionTreeClassifier(),
    ConstantClassifier(),
    0.5,
    ReportingClassSwapper(0),
)

@testset "fitted parameters and reports" begin
    f, c, fitr = MLJBase.fit(composite, 0, X)

    # check fitted_params:
    fp = @test_logs fitted_params(composite, f)
    @test Set(keys(fp)) == Set([:classifier1, :classifier2])
    @test :tree_or_leaf in keys(fp.classifier1)
    constant_fps = fp.classifier2
    @test length(constant_fps) == 2
    @test all(constant_fps) do fp
        :target_distribution in keys(fp)
    end

    # check fit report (which omits key :finalizer):
    @test Set(keys(fitr)) ==
        Set([:scaler, :clusterer, :classifier1, :training_loss, :len])
    @test fitr.scaler == (nrows=10,)
    @test fitr.clusterer == (labels=['A', 'B', 'C'],)
    @test Set(keys(fitr.classifier1)) == Set([:classes_seen, :print_tree])
    @test fitr.training_loss isa Real
    @test fitr.len == 10

    o, predictr = predict(composite, f, selectrows(X, 1:5))

    # the above should have no effect on learned parameters:
    fp = fitted_params(composite, f)
    @test Set(keys(fp)) == Set([:classifier1, :classifier2])
    @test :tree_or_leaf in keys(fp.classifier1)
    constant_fps = fp.classifier2
    @test length(constant_fps) == 2
    @test all(constant_fps) do fp
        :target_distribution in keys(fp)
    end

    # check predict report (which excludes reports from "supplementary" nodes)
    @test Set(keys(predictr)) ==
        Set([:scaler, :clusterer, :classifier1, :finalizer])
    @test predictr.scaler == (nrows=5,)
    @test predictr.clusterer == (labels=['A', 'B', 'C'],)
    @test Set(keys(predictr.classifier1)) == Set([:classes_seen, :print_tree])
    @test predictr.finalizer == (nrows=5,)

    o, predictr = predict(composite, f, selectrows(X, 1:2))

    # after second predict, predict report should update:
    @test Set(keys(predictr)) ==
        Set([:scaler, :clusterer, :classifier1, :finalizer])
    @test predictr.scaler == (nrows=2,)    # <----------- different
    @test predictr.clusterer == (labels=['A', 'B', 'C'],)
    @test Set(keys(predictr.classifier1)) == Set([:classes_seen, :print_tree])
    @test predictr.finalizer == (nrows=2,) # <---------- different

    r = MMI.report(composite, Dict(:fit => fitr, :predict=> predictr))
    @test keys(r) == (:classifier1, :scaler, :clusterer, :training_loss, :len, :predict)
    @test r.predict == predictr
    @test r == merge(fitr, (predict=predictr,))
end

@testset "logic for composite model update - start_over() method" begin
    old_composite = deepcopy(composite)
    glb_node = MLJBase.prefit(composite, 0, X) |> MLJBase.Signature |> MLJBase.glb

    # don't start over if composite unchanged:
    @test !MLJBase.start_over(composite, old_composite, glb_node)

    # don't start over if a component is mutated:
    composite.scaler.alpha = 5.0
    @test !MLJBase.start_over(composite, old_composite, glb_node)

    # don't start over if a component is replaced:
    composite.classifier2 = KNNClassifier()
    @test !MLJBase.start_over(composite, old_composite, glb_node)

    # do start over if a non-model field is changed:
    composite.mix = 0.17
    @test MLJBase.start_over(composite, old_composite, glb_node)
end

N = 50
Xin = (a=rand(N), b=rand(N), c=rand(N));
yin = rand(N);

train, test = partition(eachindex(yin), 0.7);
Xtrain = MLJBase.selectrows(Xin, train);
ytrain = yin[train];

ridge_model = FooBarRegressor(lambda=0.1)
selector_model = FeatureSelector()

@testset "first integration test" begin
    composite = SimpleDeterministicNetworkCompositeModel(model=ridge_model,
                                                  transformer=selector_model)

    fitresult, cache, rep = MLJBase.fit(composite, 0, Xtrain, ytrain);

    # to check internals:
    d = MLJBase.machines_given_model(glb(fitresult))
    ridge = only(d[:model])
    selector = only(d[:transformer])
    ridge_old_fitresult = deepcopy(fitted_params(ridge))
    selector_old_fitresult = deepcopy(fitted_params(selector))

    # this should trigger no retraining:
    fitresult, cache, rep =
        @test_logs(
            (:info, r"^Not"),
            (:info, r"^Not"),
            MLJBase.update(composite, 2, fitresult, cache, Xtrain, ytrain));
    @test fitted_params(ridge) == ridge_old_fitresult
    @test fitted_params(selector) == selector_old_fitresult

    # this should trigger update of selector and training of ridge:
    selector_model.features = [:a, :b]
    fitresult, cache, rep =
        @test_logs(
            (:info, r"^Updating"),
            (:info, r"^Training"),
            MLJBase.update(composite, 2, fitresult, cache, Xtrain, ytrain));
    @test fitted_params(ridge) != ridge_old_fitresult
    @test fitted_params(selector) != selector_old_fitresult
    ridge_old_fitresult = deepcopy(fitted_params(ridge))
    selector_old_fitresult = deepcopy(fitted_params(selector))

    # this should trigger updating of ridge only:
    ridge_model.lambda = 1.0
    fitresult, cache, rep =
        @test_logs(
            (:info, r"^Not"),
            (:info, r"^Updating"),
            MLJBase.update(composite, 2, fitresult, cache, Xtrain, ytrain));
    @test fitted_params(ridge) != ridge_old_fitresult
    @test fitted_params(selector) == selector_old_fitresult

    # smoke tests
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

mutable struct WrappedRidge <: DeterministicNetworkComposite
    ridge
end

function MLJBase.prefit(model::WrappedRidge, verbosity::Integer, X, y)
    Xs = source(X)
    ys = source(y)

    stand = Standardizer()
    standM = machine(stand, Xs)
    W = transform(standM, Xs)

    boxcox = UnivariateBoxCoxTransformer()
    boxcoxM = machine(boxcox, ys)
    z = transform(boxcoxM, ys)

    ridgeM = machine(:ridge, W, z)
    zhat = predict(ridgeM, W)
    yhat = inverse_transform(boxcoxM, zhat)

   return  (predict=yhat,)
end

MLJBase.input_scitype(::Type{<:WrappedRidge}) =
    Table(Continuous)
MLJBase.target_scitype(::Type{<:WrappedRidge}) =
    AbstractVector{<:Continuous}

@testset "second integration test" begin
    ridge = FooBarRegressor(lambda=0.1)
    model_ = WrappedRidge(ridge)
    mach = machine(model_, Xin, yin)
    fit!(mach, verbosity=0)
    yhat=predict(mach, Xin);
    ridge.lambda = 1.0
    fit!(mach, verbosity=0)
    @test predict(mach, Xin) != yhat
end

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
mutable struct WrappedClusterer <: UnsupervisedNetworkComposite
    model
end
WrappedClusterer(; model=DummyClusterer()) =
    WrappedClusterer(model)

function MLJBase.prefit(model::WrappedClusterer, verbosity::Int, X)
    Xs = source(X)
    W = transform(machine(OneHotEncoder(), Xs), Xs)
    m = machine(:model, W)
    yhat = predict(m, W)
    Wout = transform(m, W)
    foo = node(η -> first(η), yhat)
    return (predict=yhat, transform=Wout, report=(foo=foo,))
end

@testset "third integration test" begin
    X, _ = make_regression(10, 5);
    model = WrappedClusterer(model=DummyClusterer(n=2))
    mach = fit!(machine(model, X), verbosity=0)
    model.model.n = 3
    @test_logs fit!(mach, verbosity=0)
    @test transform(mach, X) == selectcols(X, 1:3)
    r = report(mach)
    @test r.model.centres == MLJBase.matrix(X)[1:3,:]
    @test r.foo == predict(mach, rows=:)[1]
    fp = fitted_params(mach)
    @test :model in keys(fp)
    levs = fp.model.fitresult
    @test predict(mach, X) == fill(levs[1], 10)
end


## NETWORK WITH MULTIPLE REPORT NODES

mutable struct TwoStages <: DeterministicNetworkComposite
    model1
    model2
    model3
end

function MLJBase.prefit(m::TwoStages, verbosity, X, y)
    Xs = source(X)
    ys = source(y)
    mach1 = machine(:model1, Xs, ys)
    mach2 = machine(:model2, Xs, ys)
    ypred1 = MLJBase.predict(mach1, Xs)
    ypred2 = MLJBase.predict(mach2, Xs)
    Y = MLJBase.table(hcat(ypred1, ypred2))
    mach3 = machine(:model3, Y, ys)
    ypred3 = MLJBase.predict(mach3, Y)
    μpred = node(x->mean(x), ypred3)
    σpred = node((x, μ)->mean((x.-μ).^2), ypred3, μpred)
    return (predict=ypred3, report=(μpred=μpred, σpred=σpred))
end

@testset "multiple report nodes and retraining" begin
    X, y = make_regression(100, 3)
    model3 = FooBarRegressor(lambda=1)
    twostages = TwoStages(FooBarRegressor(lambda=0.1),
                          FooBarRegressor(lambda=10), model3)
    mach = machine(twostages, X, y)
    fit!(mach, verbosity=0)
    rep = report(mach)

    signature = mach.fitresult
    _glb = glb(signature)
    mach1 = only(machines(_glb, :model1))
    mach2 = only(machines(_glb, :model2))
    mach3 = only(machines(_glb, :model3))
    # All machines have been fitted once
    @test mach1.state == mach2.state == mach3.state

    # Retrieve current values of interest
    μpred = rep.μpred
    σpred = rep.σpred

    # Change model3 and refit
    model3.lambda = 10
    fit!(mach, verbosity=0)
    rep = report(mach)

    # Machines 1,2 have been fitted once and machine 3 twice
    @test mach1.state == mach2.state == 1
    @test mach3.state == 2
    # The new values have been updated
    @test rep.μpred != μpred
    @test rep.σpred != σpred
end


## COMPOSITE WITH COMPONENT MODELS STORED IN NTUPLE

# `modelnames` is a tuple of `Symbol`s, one for each `model` in `models`:
mutable struct Averager{modelnames} <: DeterministicNetworkComposite
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
function MLJBase.prefit(averager::Averager{modelnames},
                     verbosity,
                     X,
                     y) where modelnames

    Xs = source(X)
    ys = source(y)

    weights = averager.weights

    machines = [machine(name, Xs, ys) for
                name in modelnames]
    predictions = hcat([predict(mach, Xs) for mach in machines]...)
    yhat = (1/sum(weights))*(predictions*weights)

    return (; predict=yhat)
end

@testset "composite with component models stored in ntuple" begin
    X, y = make_regression(10, 3);
    mach = machine(averager, X, y)
    fit!(mach, verbosity=0)
    fp = fitted_params(mach)
    @test keys(fp.model1) == (:tree, )
    @test keys(fp.model2) == (:coefficients, :intercept)
    @test isnothing(report(mach))
    @test iterator(range(averager, :(model1.K), lower=1, upper=10), 10) == 1:10
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

mutable struct ElephantModel <: ProbabilisticNetworkComposite
    scaler
    clf
    cache::Bool
end

function MLJBase.prefit(model::ElephantModel, verbosity, X, y)

    Xs = source(X)
    ys = source(y)

    mach1 = machine(:scaler, cache=model.cache)
    W = transform(mach1, Xs)

    # a classifier with reformat front-end:
    mach2 = machine(:clf, W, ys, cache=model.cache)
    yhat = predict(mach2, W)

    return (; predict=yhat)
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

    mutable struct MontenegroComposite <: UnsupervisedNetworkComposite
        stand
    end
    MontenegroComposite(; stand=Standardizer()) = MontenegroComposite(stand)

    function MLJBase.prefit(model::MontenegroComposite, verbosity, X)

        Xs = source(X)
        mach1 = machine(:stand, Xs)
        X2 = transform(mach1, Xs)

       # node for the inverse_transform:

        return (transform=X2, inverse_transform=Xs)
    end

    X = (x = Float64[1, 2, 3],)
    mach = machine(MontenegroComposite(), X)
    fit!(mach, verbosity=0, force=true)
    @test transform(mach, X).x ≈ Float64[-1, 0, 1]
    @test inverse_transform(mach, X) == X

end


# # MACHINE INTEGRATION TESTS

X = (x1=rand(3), x2=rand(3), x3=rand(3))
y = float.([1, 2, 3])

mutable struct Bar <: DeterministicNetworkComposite
    scale::Float64
    rgs
    input_stand
    target_stand
end

function MLJBase.prefit(model::Bar, verbosity, X, y)
    Xs = source(X)
    ys = source(y)
    y1 = model.scale*ys
    mach1 = machine(:input_stand, Xs)
    X1 = transform(mach1, Xs)
    mach2 = machine(:target_stand, y1)
    z = transform(mach2, ys)
    mach3 = machine(:rgs, X1, z)
    zhat = predict(mach3, X1)
    yhat = inverse_transform(mach2, zhat)
    return (; predict=yhat)
end

@testset "user-friendly inspection of reports and fitted params" begin
    scale=0.97
    rgs = KNNRegressor()
    input_stand = Standardizer()
    target_stand = UnivariateStandardizer()
    model = Bar(scale, rgs, input_stand, target_stand)

    mach = machine(model, X, y)
    fit!(mach, verbosity=0)

    # mutating the models should not effect result:
    model.scale = 42.3
    model.rgs.K = 42
    model.input_stand.features=[:x1,]

    r = report(mach)
    @test only(keys(r)) == :input_stand
    @test Set(r.input_stand.features_fit) == Set([:x1, :x2, :x3])

    fp = fitted_params(mach)
    @test fp.rgs isa NamedTuple{(:tree,)}
    @test fp.input_stand.mean_and_std_given_feature[:x1] |> collect ≈
        [mean(X.x1), std(X.x1)]
    @test fp.target_stand.fitresult |> collect ≈
        [mean(0.97*y), std(0.97*y)]  # scale = 0.97 at fit! call
end

mutable struct Mixer <: DeterministicNetworkComposite
    model1
    model2
    misc::Int
end

function MLJBase.prefit(model::Mixer, verbosity, X, y)
    Xs = source(X)
    ys = source(y)

    mach1 = machine(:model1, Xs, ys)
    mach2 = machine(:model2, Xs, ys)

    yhat1 = predict(mach1, Xs)
    yhat2 = predict(mach2, Xs)

    yhat = 0.5*yhat1 + 0.5*yhat2

    return (; predict=yhat)
end

@testset "#549" begin
    model = Mixer(KNNRegressor(), KNNRegressor(), 42)
    mach = fit!(machine(model, make_regression(10, 3)...), verbosity=0)
    fp = fitted_params(mach)
    @test !(fp.model1 isa Vector)
    @test !(fp.model2 isa Vector)

end

end # module

true
