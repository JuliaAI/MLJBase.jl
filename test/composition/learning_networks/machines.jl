module TestLearningNetworkMachines

using Test
using ..Models
using ..TestUtilities
using MLJBase
using Tables
using StableRNGs
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

@testset "wrapping a learning network in a machine" begin

    # unsupervised:
    Xs = source(X)
    W = transform(machine(OneHotEncoder(), Xs), Xs)
    clust = DummyClusterer(n=2)
    m = machine(clust, W)
    yhat = predict(m, W)
    Wout = transform(m, W)

    # test of `fitted_params(::NamedTuple)':
    fit!(Wout, verbosity=0)
    s = (predict=yhat, transform=Wout, a=source(:a), b=source(:b))
    Θ = MLJBase.call_non_operations(s)
    @test Θ.a == :a
    @test Θ.b == :b

    mach = machine(Unsupervised(), Xs; predict=yhat, transform=Wout)
    @test mach.args == (Xs, )
    @test mach.args[1] == Xs
    fit!(mach, force=true, verbosity=0)

    @test report(mach).machines == fitted_params(mach).machines

    # supervised
    y = rand("ab", N) |> categorical;
    ys = source(y)
    mm = machine(ConstantClassifier(), W, ys)
    yhat = predict(mm, W)
    e = @node auc(yhat, ys)

    @test_throws Exception machine(predict=yhat)
    mach = machine(Probabilistic(), Xs, ys; predict=yhat, training_auc=e)
    @test mach.model isa Probabilistic
    @test_throws ArgumentError machine(Probabilistic(), Xs, ys)
    @test_throws ArgumentError machine(Probabilistic(), Xs, ys; training_auc=e)

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
    mach = fit!(machine(Probabilistic(), Xs, ys; predict=yhat), verbosity=0)
    @test predict_mean(mach, X1) ≈ mean.(predict(mach, X1))
    @test predict_median(mach, X1) ≈ median.(predict(mach, X1))

end

# build a learning network:
x1 = map(n -> mod(n,3), rand(rng, UInt8, 100)) |> categorical;
x2 = randn(rng, 100);
X = (x1=x1, x2=x2);
y = x2.^2;

Xs = source(X)
ys = source(y)
z = log(ys)
stand = UnivariateStandardizer()
standM = machine(stand, z)
u = transform(standM, z)
hot = OneHotEncoder()
hotM = machine(hot, Xs)
W = transform(hotM, Xs)
knn = KNNRegressor()
knnM = machine(knn, W, u)
oak = DecisionTreeRegressor()
oakM = machine(oak, W, u)
uhat = 0.5*(predict(knnM, W) + predict(oakM, W))
zhat = inverse_transform(standM, uhat)
yhat = exp(zhat)

@testset "replace method for learning network machines" begin

    fit!(yhat, verbosity=0)

    # test nested reporting:
    r = MLJBase.report(yhat)
    d = r.report_given_machine
    ms = machines(yhat)
    @test ms == r.machines |> reverse
    @test all(mach -> report(mach) == d[mach], ms)

    hot2 = deepcopy(hot)
    knn2 = deepcopy(knn)

    # duplicate a learning network machine:
    mach  = machine(Deterministic(), Xs, ys; predict=yhat)
    mach2 = replace(mach, hot=>hot2, knn=>knn2,
                    ys=>source(ys.data);
                    empty_unspecified_sources=true)
    ss = sources(glb(mach2))
    @test isempty(ss[1])
    mach2 = @test_logs((:warn, r"No replacement"),
                       replace(mach, hot=>hot2, knn=>knn2,
                               ys=>source(ys.data)))
    yhat2 = mach2.fitresult.predict
    fit!(mach, verbosity=0)
    fit!(mach2, verbosity=0)
    @test predict(mach, X) ≈ predict(mach2, X)

    @test mach2.args[1]() == Xs()
    @test mach2.args[2]() == ys()


    ## EXTRA TESTS FOR TRAINING SEQUENCE

    # pickout the newly created machines:
    standM2 = machines(yhat2, stand) |> first
    oakM2 = machines(yhat2, oak) |> first
    knnM2 = machines(yhat2, knn) |> first
    hotM2 = machines(yhat2, hot) |> first

    @test_mach_sequence(fit!(yhat2, force=true),
                        [(:train, standM2), (:train, hotM2),
                         (:train, knnM2), (:train, oakM2)],
                        [(:train, hotM2), (:train, standM2),
                         (:train, knnM2), (:train, oakM2)],
                        [(:train, standM2), (:train, hotM2),
                         (:train, oakM2), (:train, knnM2)],
                        [(:train, hotM2), (:train, standM2),
                         (:train, oakM2), (:train, knnM2)])

    @test length(MLJBase.machines(yhat)) == length(MLJBase.machines(yhat2))
    @test MLJBase.models(yhat) == MLJBase.models(yhat2)
    @test sources(yhat) == sources(yhat2)
    @test MLJBase.tree(yhat) == MLJBase.tree(yhat2)
    @test yhat() ≈ yhat2()

    # this change should trigger retraining of all machines except the
    # univariate standardizer:
    hot2.drop_last = true
    @test_mach_sequence(fit!(yhat2),
                        [(:skip, standM2), (:update, hotM2),
                         (:train, knnM2), (:train, oakM2)],
                        [(:update, hotM2), (:skip, standM2),
                         (:train, knnM2), (:train, oakM2)],
                        [(:skip, standM2), (:update, hotM2),
                         (:train, oakM2), (:train, knnM2)],
                        [(:update, hotM2), (:skip, standM2),
                         (:train, oakM2), (:train, knnM2)])
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

    mach = machine(Unsupervised(), Xs; transform=X2)
    @test_logs((:error, r"The hyper"),
               @test_throws(ArgumentError,
                            MLJBase.network_model_names(model, mach)))
end

end

true
