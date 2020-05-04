module TestLearningNetworkMachines

using Test
using ..Models
using MLJBase
using Tables

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

@testset "wrapping a learning network in a machine" begin

    N = 20
    X = (a = rand(N), b = categorical(rand("FM", N)))

    # unsupervised:
    Xs = source(X)
    W = transform(machine(OneHotEncoder(), Xs), Xs)
    clust = DummyClusterer(n=2)
    m = machine(clust, W)
    yhat = predict(m, W)
    Wout = transform(m, W)
    N = 20
    X = (a = rand(N), b = categorical(rand("FM", N)))

    Xs = source(X)
    W = transform(machine(OneHotEncoder(), Xs), Xs)
    clust = DummyClusterer(n=2)
    m = machine(clust, W)
    yhat = predict(m, W)
    Wout = transform(m, W)

    signature = (predict=yhat, transform=Wout)
    @test MLJBase.args(signature) == (Xs, )
    @test MLJBase.model_supertype(signature) == Unsupervised

    mach = machine!(predict=yhat, transform=Wout)
    @test mach.args == (Xs, )
    @test isempty(mach.args[1])
    fit!(mach, force=true)

    report(mach)
    fitted_params(mach)

    # supervised
    y = rand("ab", N) |> categorical;
    ys = source(y)
    mm = machine(ConstantClassifier(), W, ys)
    yhat = predict(mm, W)
    @test_throws Exception machine!(predict=yhat)
    ys.kind = :target
    machine!(predict=yhat)
    ys.kind = :input
    Xs.kind = :target
    machine!(Probabilistic(), Xs, ys, predict=yhat)
    @test Xs.kind == :input
    @test ys.kind == :target
    @test_throws ArgumentError machine!(Probabilistic(), Xs, ys)
    m = machine!(Probabilistic(), predict=yhat)
    @test m.model isa Probabilistic

end



end

true
