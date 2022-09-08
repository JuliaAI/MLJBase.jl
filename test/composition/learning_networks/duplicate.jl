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

_header(accel) =

@testset "duplicate()  method; $(typeof(accel))" for accel in (CPU1(), CPUThreads())

    fit!(yhat, verbosity=0, acceleration=accel)

    # test nested reporting:
    r = MLJBase.report(yhat)
    d = r.report_given_machine
    ms = machines(yhat)
    @test ms == r.machines |> reverse
    @test all(mach -> report(mach) == d[mach], ms)

    hot2 = deepcopy(hot)
    knn2 = deepcopy(knn)

    # duplicate the network with `yhat` as glb:
    yhat_clone = @test_logs(
        (:warn, r"No replacement"),
        MLJBase.duplicate(
            yhat,
            hot=>hot2,
            knn=>knn2,
            ys=>source(42);
            copy_models_deeply=false,
        ),
    )

    # test models and sources duplicated correctly:
    models_clone = MLJBase.models(yhat_clone)
    @test models_clone[1] === stand
    @test models_clone[2] === knn2
    @test models_clone[3] === hot2
    sources_clone = sources(yhat_clone)
    @test sources_clone[1]() == X
    @test sources_clone[2]() === 42

    # test serializable option:
    fit!(yhat, verbosity=0)
    yhat_ser = MLJBase.duplicate(yhat; serializable=true)
    machines_ser = machines(yhat_ser)
    mach4 = machines_ser[4]
    @test mach4.state == -1
    @test all(isempty, sources(yhat_ser))

    # duplicate a signature:
    signature = (predict=yhat, report=(mae=enode,))
    signature_clone = @test_logs(
        (:warn, r"No replacement"),
        MLJBase.duplicate(
            signature,
            hot=>hot2,
            knn=>knn2,
            ys=>source(42);
            copy_models_deeply=false,
        )
    )
    glb_node = glb(signature_clone)
    models_clone = MLJBase.models(glb_node)
    @test models_clone[1] === stand
    @test models_clone[2] === knn2
    @test models_clone[3] === hot2
    sources_clone = sources(glb_node)
    @test sources_clone[1]() == X
    @test sources_clone[2]() === 42

    # duplicate a learning network machine:
    mach  = machine(Deterministic(), Xs, ys;
                    predict=yhat,
                    report=(mae=enode,))
    mach2 = MLJBase.duplicate(mach, hot=>hot2, knn=>knn2,
                    ys=>source(ys.data);
                    empty_unspecified_sources=true)
    ss = sources(glb(mach2))
    @test isempty(ss[1])
    mach2 = @test_logs((:warn, r"No replacement"),
                       MLJBase.duplicate(mach, hot=>hot2, knn=>knn2,
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
end


end # module

true
