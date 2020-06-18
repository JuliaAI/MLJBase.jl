module TestCompositionModelsInspection

using Test
using MLJBase
using Tables
import MLJBase
using ..Models
using CategoricalArrays
using OrderedCollections
using Statistics

X = (x1=rand(3), x2=rand(3), x3=rand(3))
y = float.([1, 2, 3])

@load KNNRegressor

@testset "tuple_keyed_on_model_names" begin

    model1 = KNNRegressor(K=1)
    model2 = KNNRegressor(K=2)
    model3 = KNNRegressor(K=3)
    model4 = KNNRegressor(K=4)

    _models = deepcopy.((model4, model3, model1))

    a1 = machine(model1, X, y)
    a2 = machine(model1, X, y)
    b = machine(model2, X, y)
    c = machine(model3, X, y)
    d = machine(model4, X, y)

    fit!.([a1, a2, b, c, d])

    # mutating the models should not effect the result:
    model1.K=0
    model2.K=0
    model3.K=0
    model4.K=0

    item_given_machine =
        LittleDict(a1=>"a1", a2=>"a2", b=>"b", c=>"c", d=>"d")

    _names = (:knn4, :knn3, :knn1)

    nt = MLJBase.tuple_keyed_on_model_names(item_given_machine,
                                            _models,
                                            _names)

    @test nt == (knn4="d", knn3="c", knn1=["a1", "a2"])
end

mutable struct Bar <: DeterministicComposite
    scale::Float64
    rgs
    input_stand
    target_stand
end

function MLJBase.fit(model::Bar, verbosity, X, y)
    Xs = source(X)
    ys = source(y)
    y1 = model.scale*ys
    mach1 = machine(model.input_stand, Xs)
    X1 = transform(mach1, Xs)
    mach2 = machine(model.target_stand, y1)
    z = transform(mach2, ys)
    mach3 = machine(model.rgs, X1, z)
    zhat = predict(mach3, X1)
    yhat = inverse_transform(mach2, zhat)
    mach = machine(Deterministic(), Xs, ys; predict=yhat)
    fit!(mach)
    return mach()
end

scale=0.97
rgs = KNNRegressor()
input_stand = Standardizer()
target_stand = UnivariateStandardizer()
model = Bar(scale, rgs, input_stand, target_stand)

mach = machine(model, X, y)
fit!(mach)

@testset "models_and_names(::Machine{<:Composite})" begin
    _models, _names = MLJBase.models_and_names(mach)
    @test _models == (rgs, input_stand, target_stand)
    @test _names == (:rgs, :input_stand, :target_stand)
end

@testset "user-friendly inspection of reports and fitted params" begin

    # mutating the models should not effect result:
    model.scale = 42.3
    model.rgs.K = 42
    model.input_stand.features=[:x1,]

    r = report(mach)
    @test isempty(r.rgs)
    @test Set(r.input_stand.features_fit) == Set([:x1, :x2, :x3])
    @test isempty(r.target_stand)
    @test r.report_given_machine isa AbstractDict

    fp = fitted_params(mach)
    @test fp.rgs isa NamedTuple{(:tree,)}
    @test fp.input_stand.mean_and_std_given_feature[:x1] |> collect ≈
        [mean(X.x1), std(X.x1)]
    @test fp.target_stand.fitresult |> collect ≈
        [mean(0.97*y), std(0.97*y)]  # scale = 0.97 at fit! call
end

end

true
