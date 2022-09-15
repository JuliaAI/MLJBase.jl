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

@testset "fitted parameters and reports at level of model API" begin
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






end # module

true
