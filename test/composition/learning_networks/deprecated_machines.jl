module TestLearningNetworkMachines

const depwarn=false

using Test
using ..Models
using ..TestUtilities
using MLJBase
using Tables
using StableRNGs
using Serialization
using StatisticalMeasures
rng = StableRNG(616161)

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


N = 20
X = (a = rand(N), b = categorical(rand("FM", N)))

@testset "signature helpers" begin
    @test MLJBase._call(NamedTuple()) == NamedTuple()
    a = source(:a)
    b = source(:b)
    W = source(:W)
    yhat = source(:yhat)
    s = (transform=W,
         report=(a=a, b=b),
         predict=yhat)
    @test MLJBase._report_part(s) == (a=a, b=b)
    @test MLJBase._operation_part(s) == (transform=W, predict=yhat)
    @test MLJBase._nodes(s) == (W, yhat, a, b)
    @test MLJBase._operations(s) == (:transform, :predict)
    R = MLJBase._call(MLJBase._report_part(s))
    @test R.a == :a
    @test R.b == :b
end

@testset "wrapping a learning network in a machine" begin

    # unsupervised:
    Xs = source(X)
    W = transform(machine(OneHotEncoder(), Xs), Xs)
    clust = DummyClusterer(n=2)
    m = machine(clust, W)
    yhat = predict(m, W)
    Wout = transform(m, W)
    rnode = source(:stuff)

    # test of `fitted_params(::NamedTuple)':
    fit!(Wout, verbosity=0)

    @test_throws(MLJBase.ERR_BAD_SIGNATURE,
                 machine(Unsupervised();
                         predict=yhat,
                         fitted_params=rnode,
                         depwarn)
                 )
    @test_throws(MLJBase.ERR_EXPECTED_NODE_IN_SIGNATURE,
                 machine(Unsupervised();
                         predict=42,
                         depwarn)
                 )
    @test_throws(MLJBase.ERR_EXPECTED_NODE_IN_SIGNATURE,
                 machine(Unsupervised(), Xs;
                         predict=yhat,
                         transform=Wout,
                         report=(some_stuff=42,),
                         depwarn)
                 )
    mach = machine(Unsupervised(), Xs;
                   predict=yhat,
                   transform=Wout,
                   report=(some_stuff=rnode,),
                   depwarn)
    @test mach.args == (Xs, )
    @test mach.args[1] == Xs
    fit!(mach, force=true, verbosity=0)
    Θ = mach.fitresult
    @test Θ.predict == yhat
    @test Θ.transform == Wout
    Θ.report.some_stuff == rnode
    @test report(mach).some_stuff == :stuff
    @test report(mach).machines == fitted_params(mach).machines

    # supervised
    y = rand("ab", N) |> categorical;
    ys = source(y)
    mm = machine(ConstantClassifier(), W, ys)
    yhat = predict(mm, W)
    e = @node auc(yhat, ys)

    @test_throws Exception machine(; predict=yhat, depwarn)
    mach = machine(Probabilistic(), Xs, ys;
                   predict=yhat,
                   report=(training_auc=e,),
                   depwarn)
    @test mach.model isa Probabilistic
    @test_throws ArgumentError machine(Probabilistic(), Xs, ys; depwarn)
    @test_throws ArgumentError machine(Probabilistic(), Xs, ys;
                                       report=(training_auc=e,),
                                       depwarn)

    # test extra report items coming from `training_auc=e` above
    fit!(mach, verbosity=0)
    err = auc(yhat(), y)
    @test report(mach).training_auc ≈ err

    # supervised - predict_mode
    @test predict_mode(mach, X) == mode.(predict(mach, X))
    predict_mode(mach, rows=1:2) == predict_mode(mach, rows=:)[1:2]

    # evaluate a learning machine
    evaluate!(mach, measure=LogLoss(), verbosity=0)

    # supervised - predict_median, predict_mean
    X1, y1 = make_regression(20)

    Xs = source(X1); ys = source(y1)
    mm = machine(ConstantRegressor(), Xs, ys)
    yhat = predict(mm, Xs)
    mach = fit!(machine(Probabilistic(), Xs, ys; predict=yhat, depwarn), verbosity=0)
    @test predict_mean(mach, X1) ≈ mean.(predict(mach, X1))
    @test predict_median(mach, X1) ≈ median.(predict(mach, X1))

end

mutable struct DummyComposite <: DeterministicComposite
    stand1
    stand2
end

@testset "issue 377" begin
    stand = Standardizer()
    model = DummyComposite(stand, stand)

    Xs = source()
    mach1 = machine(model.stand1, Xs)
    X1 = transform(mach1, Xs)
    mach2 = machine(model.stand2, X1)
    X2 = transform(mach2, X1)

    mach = machine(Unsupervised(), Xs; transform=X2, depwarn)
    @test_logs((:error, r"The hyper"),
               @test_throws(ArgumentError,
                            MLJBase.network_model_names(model, mach)))
end

end

true
