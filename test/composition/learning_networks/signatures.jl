module TestSignatures

using ..Models
using MLJBase
using StableRNGs
using Tables
using Test
using MLJModelInterface


#KNNClassifier = @load KNNClassifier
#DecisionTreeClassifer = @load DecisionTreeClassifier pkg=DecisionTree

@testset "signatures - accessor functions" begin
    a = source(:a)
    b = source(:b)
    W = source(:W)
    yhat = source(:yhat)
    s = (transform=W,
         report=(a=a, b=b),
         predict=yhat)
    @test MLJBase.report_nodes(s) == (a=a, b=b)
    @test MLJBase.operation_nodes(s) == (transform=W, predict=yhat)
    @test MLJBase.operations(s) == (:transform, :predict)
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


@testset "signature methods: glb, report, age" begin

    # Some complicated learning network:
    Xs = source(first(make_blobs(10)))
    mach0 = machine(:clusterer)
    ytrain = predict(mach0, Xs)
    mach1 = machine(:classifier1, Xs, ytrain)
    # two machines pointing to same model:
    mach2a = machine(:classifier2, Xs, ytrain)
    mach2b = machine(:classifier2, Xs, ytrain)
    y1 = predict(mach1, Xs) # probabilistic predictions
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
    signature = (; predict=yhat, report=(; loss=loss))

    glb1 = glb(signature)
    glb2 = glb(yhat, loss)

    clusterer = OneShotClusterer(3, StableRNG(123))
    composite = (
        clusterer = clusterer,
        classifier1 = KNNClassifier(),
        classifier2 = ConstantClassifier(),
    )

    fit!(glb1; composite, verbosity=0)
    fit!(glb2; composite, verbosity=0)

    @test glb1() == glb2()

    r = MLJBase.report(signature)
    @test r.classifier1 == [MLJBase.report_given_method(mach1),]
    @test r.classifier2 == [
        MLJBase.report_given_method(mach2a),
        MLJBase.report_given_method(mach2b),
    ]
    @test r.clusterer == [MLJBase.report_given_method(mach0),]
    @test r.loss == loss()

    @test sum(MLJBase.age.(machines(glb1))) == MLJBase.age(signature)
end

end # module

true
