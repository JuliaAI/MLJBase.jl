module TestSignatures

using ..Models
using MLJBase
using StableRNGs
using Tables
using Test
using MLJModelInterface
using OrderedCollections

@testset "signatures - accessor functions" begin
    a = source(:a)
    b = source(:b)
    W = source(:W)
    yhat = source(:yhat)

    s = (
        transform=W,
        report=(a=a, b=b),
        fitted_params=(c=W,),
        predict=yhat,
        acceleration=CPUThreads(),
    ) |> MLJBase.Signature
    @test MLJBase.report_nodes(s) == (a=a, b=b)
    @test MLJBase.fitted_params_nodes(s) == (c=W,)
    @test MLJBase.operation_nodes(s) == (transform=W, predict=yhat)
    @test MLJBase.operations(s) == (:transform, :predict)
    @test MLJBase.acceleration(s) == CPUThreads()

    s = (
        transform=W,
        predict=yhat,
    ) |> MLJBase.Signature
    @test MLJBase.report_nodes(s) == NamedTuple()
    @test MLJBase.acceleration(s) == CPU1()
end


# # DUMMY CLUSTERER THAT DOES NOT GENERALIZE TO NEW DATA

mutable struct OneShotClusterer <: Static
    nclusters
    rng
end

# X is a n x p matrix
function MLJModelInterface.predict(model::OneShotClusterer, ::Nothing, X)
    rng = copy(model.rng)
    Xmat = Tables.matrix(X)
    labels = map(i -> Char(64 + i), 1:model.nclusters)
    Xout =  categorical(rand(rng, labels, size(Xmat, 1)); levels=labels)
    report = (; labels)
    return Xout, report
end

MLJModelInterface.reporting_operations(::OneShotClusterer) = (:predict,)

# Some complicated learning network:
Xs = source(first(make_blobs(10)))
mach0 = machine(:clusterer)
ytrain = predict(mach0, Xs)
mach1 = machine(:classifier1, Xs, ytrain)
# two machines pointing to same model:
mach2a = machine(:classifier2, Xs, ytrain)
mach2b = machine(:classifier2, Xs, ytrain)
y1 = predict(mach1, Xs) # probabilistic predictions
junk = node(fitted_params, mach1)
y2a = predict(mach2a, Xs) # probabilistic predictions
y2b = predict(mach2b, Xs) # probabilistic predictions
loss = node(
    (y1, y2) -> brier_loss(y1, mode.(y2)) + brier_loss(y2, mode.(y1)) |> mean,
    y1,
    y2a,
)
λ = 0.3
ymix = λ*y1 + (1 - λ)*(0.2*y2a + 0.8*y2b)
yhat = mode(ymix)
signature = (;
             predict=yhat,
             report=(; loss=loss),
             fitted_params=(; junk),
             ) |> MLJBase.Signature

glb1 = glb(signature)
glb2 = glb(yhat, loss, junk)

clusterer = OneShotClusterer(3, StableRNG(123))
composite = (
    clusterer = clusterer,
    classifier1 = KNNClassifier(),
    classifier2 = ConstantClassifier(),
)

@testset "signature methods: glb, report, age, output_and_report" begin
    fit!(glb1; composite, verbosity=0)
    fit!(glb2; composite, verbosity=0)

    @test glb1() == glb2()

    r = MLJBase.report(signature)
    # neither classifier has a contribution to the report:
    @test isnothing(report(mach1))
    @test isnothing(report(mach2a))
    @test isnothing(report(mach2b))
    @test r == (clusterer = report(mach0), loss=loss())

    fr = MLJBase.fitted_params(signature)
    @test keys(fr) == (:classifier1, :classifier2, :junk)

    @test sum(MLJBase.age.(machines(glb1))) == MLJBase.age(signature)

    output, r = MLJBase.output_and_report(signature, :predict, selectrows(Xs(), 1:2))
    @test output == yhat(selectrows(Xs(), 1:2))
    @test r == (clusterer = (labels = ['A', 'B', 'C'],),)
end

@testset "signature helper: tuple_keyed_on_model" begin
    d = OrderedDict(:model1 => [:mach1a, :mach1b], :model2 => [:mach2,])
    f(mach) = mach == :mach2 ? nothing : 42
    g(mach) = mach in [:mach1a, :mach1b] ? nothing : 24

    @test MLJBase.tuple_keyed_on_model(f, d) == (model1=[42, 42],)
    @test MLJBase.tuple_keyed_on_model(f, d; scalarize=false) == (model1=[42, 42],)
    @test MLJBase.tuple_keyed_on_model(f, d; drop_nothings=false) ==
        (model1=[42, 42], model2=nothing)
    @test MLJBase.tuple_keyed_on_model(f, d; drop_nothings=false, scalarize=false) ==
        (model1=[42, 42], model2=[nothing,])
    @test MLJBase.tuple_keyed_on_model(g, d) == (model2=24,)
    @test MLJBase.tuple_keyed_on_model(g, d; scalarize=false) == (model2=[24,],)
    @test MLJBase.tuple_keyed_on_model(g, d; drop_nothings=false) ==
        (model1=[nothing, nothing], model2=24)
    @test MLJBase.tuple_keyed_on_model(g, d; drop_nothings=false, scalarize=false) ==
        (model1=[nothing, nothing], model2=[24,])
end

@testset "signature helper: machines_given_model" begin
    d = MLJBase.machines_given_model(glb1)
    @test d[:clusterer] == Any[mach0,]
    @test d[:classifier1] == Any[mach1,]
    @test d[:classifier2] == Any[mach2a, mach2b]
    @test length(keys(d)) == 3
end

@testset "signature helper: call_and_copy" begin
    @test_throws MLJBase.ERR_CALL_AND_COPY MLJBase.call_and_copy(42) == 42
    x = Ref(3)
    n = source(x)
    frozen_x = MLJBase.call_and_copy(n)
    @test frozen_x[] == 3
    x[] = 5
    @test frozen_x[] == 3
    y = source(7)
    @test MLJBase.call_and_copy((a=source(20), b=y)) == (a=20, b=7)
end # module

end

true
