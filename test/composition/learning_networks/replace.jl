module TestReplace

using Test
using ..Models
using ..TestUtilities
using MLJBase
using Tables
using StableRNGs
using Serialization
rng = StableRNG(616161)

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
enode = @node mae(ys, yhat)

@testset(
    "replace method; acceleration: $(typeof(accel))",
    for accel in (CPU1(), CPUThreads())

    fit!(yhat, verbosity=0, acceleration=accel)

    # test nested reporting:
    r = MLJBase.report(yhat)
    d = r.report_given_machine
    ms = machines(yhat)
    @test ms == r.machines |> reverse
    @test all(mach -> report(mach) == d[mach], ms)

    hot2 = deepcopy(hot)
    knn2 = deepcopy(knn)

    # duplicate a learning network machine:
    mach  = machine(Deterministic(), Xs, ys;
                    predict=yhat,
                    report=(mae=enode,))
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
    @test report(mach).mae ≈ report(mach2).mae

    @test mach2.args[1]() == Xs()
    @test mach2.args[2]() == ys()


    ## EXTRA TESTS FOR TRAINING SEQUENCE

    # pickout the newly created machines:
    standM2 = machines(yhat2, stand) |> first
    oakM2 = machines(yhat2, oak) |> first
    knnM2 = machines(yhat2, knn) |> first
    hotM2 = machines(yhat2, hot) |> first

    @test_mach_sequence(fit!(yhat2, force=true, acceleration=accel),
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
    @test_mach_sequence(fit!(yhat2, acceleration=accel),
                        [(:skip, standM2), (:update, hotM2),
                         (:train, knnM2), (:train, oakM2)],
                        [(:update, hotM2), (:skip, standM2),
                         (:train, knnM2), (:train, oakM2)],
                        [(:skip, standM2), (:update, hotM2),
                         (:train, oakM2), (:train, knnM2)],
                        [(:update, hotM2), (:skip, standM2),
                         (:train, oakM2), (:train, knnM2)])
    end,
)

end # module
