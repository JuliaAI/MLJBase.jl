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
    return!(mach, model, verbosity)
end

scale=0.97
rgs = KNNRegressor()
input_stand = Standardizer()
target_stand = UnivariateStandardizer()
model = Bar(scale, rgs, input_stand, target_stand)

mach = machine(model, X, y)
fit!(mach, verbosity=0)

@testset "user-friendly inspection of reports and fitted params" begin

    # mutating the models should not effect result:
    model.scale = 42.3
    model.rgs.K = 42
    model.input_stand.features=[:x1,]

    r = report(mach)
    @test isnothing(r.rgs)
    @test Set(r.input_stand.features_fit) == Set([:x1, :x2, :x3])
    @test isnothing(r.target_stand)
    @test r.report_given_machine isa AbstractDict

    fp = fitted_params(mach)
    @test fp.rgs isa NamedTuple{(:tree,)}
    @test fp.input_stand.mean_and_std_given_feature[:x1] |> collect ≈
        [mean(X.x1), std(X.x1)]
    @test fp.target_stand.fitresult |> collect ≈
        [mean(0.97*y), std(0.97*y)]  # scale = 0.97 at fit! call
end


mutable struct Mixer <: DeterministicComposite
    model1
    model2
    misc::Int
end

@testset "#549" begin

    function MLJBase.fit(model::Mixer, verbosity, X, y)
        Xs = source(X)
        ys = source(y)

        mach1 = machine(model.model1, Xs, ys)
        mach2 = machine(model.model2, Xs, ys)

        yhat1 = predict(mach1, Xs)
        yhat2 = predict(mach2, Xs)

        yhat = 0.5*yhat1 + 0.5*yhat2

        learning_mach = machine(Deterministic(), Xs, ys, predict=yhat)
        return!(learning_mach, model, verbosity)
    end

    model = Mixer(KNNRegressor(), KNNRegressor(), 42)
    mach = fit!(machine(model, make_regression(10, 3)...), verbosity=0)
    fp = fitted_params(mach)
    @test !(fp.model1 isa Vector)
    @test !(fp.model2 isa Vector)

end

end

true
