#module TestResampling

using Distributed
import ComputationalResources: CPU1, CPUProcesses, CPUThreads
using .TestUtilities
using ProgressMeter

@everywhere begin
    using .Models
    import Random.seed!
    seed!(1234)
    const verb = 0
end

using Test
using MLJBase
import Distributions
import StatsBase
@static if VERSION >= v"1.3.0-DEV.573"
    using .Threads
end

@testset_accelerated "dispatch of resources and progress meter" accel begin

    X = (x = [1, ],)
    y = [2.0, ]

    @everywhere begin
        nfolds = 6
        nmeasures = 2
        func(mach, k) = (sleep(0.01*rand()); fill(1:k, nmeasures))
    end

    machines = Dict(1 => machine(ConstantRegressor(), X, y))

    channel = RemoteChannel(()->Channel{Bool}(nfolds) , 1)
    p = Progress(nfolds, dt=0)

    @sync begin

        # printing the progress bar
        t1 = @async while take!(channel)
            next!(p)
        end

        t2 = @async begin
            global result =
                MLJBase._evaluate!(func, machines, accel, nfolds, channel)
        end
    end

    @test result ==
        [1:1, 1:1, 1:2, 1:2, 1:3, 1:3, 1:4, 1:4, 1:5, 1:5, 1:6, 1:6]

    close(channel)

end


@test CV(nfolds=6) == CV(nfolds=6)
@test CV(nfolds=5) != CV(nfolds=6)
@test MLJBase.train_test_pairs(CV(), 1:10) !=
     MLJBase.train_test_pairs(CV(shuffle=true), 1:10)
@test MLJBase.train_test_pairs(Holdout(), 1:10) !=
     MLJBase.train_test_pairs(Holdout(shuffle=true), 1:10)

@testset "checking measure/model compatibility" begin
    model = ConstantRegressor()
    y = rand(4)
    override=false
    @test MLJBase._check_measure(:junk, :junk, :junk, :junk, true) == nothing
    @test_throws(ArgumentError,
                  MLJBase._check_measure(model, rms, y, predict, override))
    @test MLJBase._check_measure(model, rms, y, predict_mean, override) ==
        nothing
    @test MLJBase._check_measure(model, rms, y, predict_median, override) ==
        nothing
    y=categorical(collect("abc"))
    @test_throws(ArgumentError,
                 MLJBase._check_measure(model, rms, y,
                                        predict_median, override))
    model = ConstantClassifier()
    @test_throws(ArgumentError,
                 MLJBase._check_measure(model, misclassification_rate, y,
                                        predict, override))
    @test MLJBase._check_measure(model, misclassification_rate, y,
                            predict_mode, override) == nothing
    model = Models.DeterministicConstantClassifier()
    @test_throws ArgumentError MLJBase._check_measure(model, cross_entropy, y,
                            predict, override)
end

@testset_accelerated "folds specified" accel begin
    x1 = ones(10)
    x2 = ones(10)
    X  = (x1=x1, x2=x2)
    y  = [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0]

    my_rms(yhat, y) = sqrt(mean((yhat -y).^2))
    my_mae(yhat, y) = abs.(yhat - y)
    MLJBase.reports_each_observation(::typeof(my_mae)) = true

    resampling = [(3:10, 1:2),
                  ([1, 2, 5, 6, 7, 8, 9, 10], 3:4),
                  ([1, 2, 3, 4, 7, 8, 9, 10], 5:6),
                  ([1, 2, 3, 4, 5, 6, 9, 10], 7:8),
                  (1:8, 9:10)]

    model = Models.DeterministicConstantRegressor()
    mach  = machine(model, X, y)

    # check detection of incompatible measure (cross_entropy):
    @test_throws ArgumentError evaluate!(mach, resampling=resampling,
                                         measure=[cross_entropy, rmslp1],
                                         verbosity=verb,
                                         acceleration=accel)
    result = evaluate!(mach, resampling=resampling, verbosity=verb,
                       measure=[my_rms, my_mae, rmslp1], acceleration=accel)

    v = [1/2, 3/4, 1/2, 3/4, 1/2]

    @test result.per_fold[1] ≈ v
    @test result.per_fold[2] ≈ v
    @test result.per_fold[3][1] ≈ abs(log(2) - log(2.5))
    @test ismissing(result.per_observation[1])
    @test result.per_observation[2][1] ≈ [1/2, 1/2]
    @test result.per_observation[2][2] ≈ [3/4, 3/4]
    @test result.measurement[1] ≈ mean(v)
    @test result.measurement[2] ≈ mean(v)
end

@testset "repeated resampling" begin
    x1 = ones(20)
    x2 = ones(20)
    X = (x1=x1, x2=x2)
    y = rand(20)

    holdout = Holdout(fraction_train=0.75, rng=123)
    model = Models.DeterministicConstantRegressor()
    mach = machine(model, X, y)
    result = evaluate!(mach, resampling=holdout, verbosity=verb,
                       measure=[rms, rmslp1], repeats=6)
    per_fold = result.per_fold[1]
    @test unique(per_fold) |> length == 6
    @test abs(mean(per_fold) - std(y)) < 0.06 # very rough check

    cv = CV(nfolds=3, rng=123)
    model = Models.DeterministicConstantRegressor()
    mach = machine(model, X, y)
    result = evaluate!(mach, resampling=cv, verbosity=verb,
                       measure=[rms, rmslp1], repeats=6)
    per_fold = result.per_fold[1]
    @test unique(per_fold) |> length == 18
    @test abs(mean(per_fold) - std(y)) < 0.06 # very rough check
end

@testset_accelerated "holdout" accel begin
    x1 = ones(4)
    x2 = ones(4)
    X = (x1=x1, x2=x2)
    y = [1.0, 1.0, 2.0, 2.0]

    @test MLJBase.show_as_constructed(Holdout)
    holdout = Holdout(fraction_train=0.75)
    model = Models.DeterministicConstantRegressor()
    mach = machine(model, X, y)
    result = evaluate!(mach, resampling=holdout, verbosity=verb,
                       measure=[rms, rmslp1], acceleration=accel)
    result = evaluate!(mach, resampling=holdout, verbosity=verb,
                       acceleration=accel)
    result.measurement[1] ≈ 2/3

    # test direct evaluation of a model + data:
    result = evaluate(model, X, y, verbosity=1,
                      resampling=holdout, measure=rms)
    @test result.measurement[1] ≈ 2/3

    X = (x=rand(100),)
    y = rand(100)
    mach = machine(model, X, y)
    evaluate!(mach, verbosity=verb,
              resampling=Holdout(shuffle=true, rng=123), acceleration=accel)
    e1 = evaluate!(mach, verbosity=verb,
                   resampling=Holdout(shuffle=true),
                   acceleration=accel).measurement[1]
    @test e1 != evaluate!(mach, verbosity=verb,
                          resampling=Holdout(),
                          acceleration=accel).measurement[1]
end

@testset_accelerated "cv" accel begin
    x1 = ones(10)
    x2 = ones(10)
    X = (x1=x1, x2=x2)
    y = [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0]

    @test MLJBase.show_as_constructed(CV)
    cv=CV(nfolds=5)
    model = Models.DeterministicConstantRegressor()
    mach = machine(model, X, y)
    result = evaluate!(mach, resampling=cv, measure=[rms, rmslp1],
                       acceleration=accel, verbosity=verb)

    @test result.per_fold[1] ≈ [1/2, 3/4, 1/2, 3/4, 1/2]

    shuffled = evaluate!(mach, resampling=CV(shuffle=true), verbosity=verb,
                          acceleration=accel) # using rms default
    @test shuffled.measurement[1] != result.measurement[1]
end

@testset "stratified_cv" begin

    # check in explicit example:
    y = categorical(['c', 'a', 'b', 'a', 'c', 'x',
                 'c', 'a', 'a', 'b', 'b', 'b', 'b', 'b'])
    rows = [14, 13, 12, 11, 10, 9, 8, 7, 5, 4, 3, 2, 1]
    @test y[rows] == collect("bbbbbaaccabac")
    scv = StratifiedCV(nfolds=3)
    pairs = MLJBase.train_test_pairs(scv, rows, nothing, y)
    @test pairs == [([12, 11, 10, 8, 5, 4, 3, 2, 1], [14, 13, 9, 7]),
                    ([14, 13, 10, 9, 7, 4, 3, 2, 1], [12, 11, 8, 5]),
                    ([14, 13, 12, 11, 9, 8, 7, 5], [10, 4, 3, 2, 1])]
    scv_random = StratifiedCV(nfolds=3, shuffle=true)
    pairs_random = MLJBase.train_test_pairs(scv_random, rows, nothing, y)
    @test pairs != pairs_random

    # wrong target type throws error:
    @test_throws Exception MLJBase.train_test_pairs(scv, rows, nothing, get.(y))

    # too many folds throws error:
    @test_throws Exception MLJBase.train_test_pairs(StratifiedCV(nfolds=4),
                                                rows, nothing, y)

    # check class distribution is preserved in a larger randomized example:
    N = 30
    y = shuffle(vcat(fill(:a, N), fill(:b, 2N),
                        fill(:c, 3N), fill(:d, 4N))) |> categorical;
    d = Distributions.fit(MLJBase.UnivariateFinite, y)
    pairs = MLJBase.train_test_pairs(scv, 1:10N, nothing, y)
    folds = vcat(first.(pairs), last.(pairs))
    @test all([Distributions.fit(MLJBase.UnivariateFinite, y[fold]) ≈ d for fold in folds])
end

@testset_accelerated "sample weights in evaluation" accel begin
    # cv:
    x1 = ones(4)
    x2 = ones(4)
    X = (x1=x1, x2=x2)
    y = [1.0, 2.0, 3.0, 1.0]
    w = 1:4
    cv=CV(nfolds=2)
    model = Models.DeterministicConstantRegressor()
    mach = machine(model, X, y)
    e = evaluate!(mach, resampling=cv, measure=l1,
                  weights=w, verbosity=verb, acceleration=accel).measurement[1]

    @test e ≈ (1/3 + 13/14)/2
end

@testset_accelerated "resampler as machine" accel begin
    N = 50
    X = (x1=rand(N), x2=rand(N), x3=rand(N))
    y = X.x1 -2X.x2 + 0.05*rand(N)

    ridge_model = FooBarRegressor(lambda=20.0)
    holdout = Holdout(fraction_train=0.75)
    resampler = Resampler(resampling=holdout, model=ridge_model, measure=mae)
    resampling_machine = machine(resampler, X, y)
    @test_logs((:info, r"^Training"), fit!(resampling_machine))
    e1=evaluate(resampling_machine).measurement[1]
    mach = machine(ridge_model, X, y)
    @test e1 ≈  evaluate!(mach, resampling=holdout,
                          measure=mae, verbosity=verb,
                          acceleration=accel).measurement[1]
    ridge_model.lambda=1.0
    fit!(resampling_machine, verbosity=2)
    e2=evaluate(resampling_machine).measurement[1]
    @test e1 != e2
    resampler.weights = rand(N)
    fit!(resampling_machine, verbosity=verb)
    e3=evaluate(resampling_machine).measurement[1]
    @test e3 != e2

    @test MLJBase.package_name(Resampler) == "MLJBase"
    @test MLJBase.is_wrapper(Resampler)
    rnd = randn(5)
    @test evaluate(resampler, rnd) === rnd
end

struct DummyResamplingStrategy <: MLJBase.ResamplingStrategy end

@testset_accelerated "custom strategy using resampling depending on X, y" accel begin
    function MLJBase.train_test_pairs(resampling::DummyResamplingStrategy,
                              rows, X, y)
        train = filter(rows) do j
            y[j] == y[1]
        end
        test = setdiff(rows, train)
        return [(train, test),]
    end

    X = (x = rand(8), )
    y = categorical([:x, :y, :x, :x, :y, :x, :x, :y])
    @test MLJBase.train_test_pairs(DummyResamplingStrategy(), 2:6, X, y) ==
        [([3, 4, 6], [2, 5]),]

    e = evaluate(ConstantClassifier(), X, y,
                 measure=misclassification_rate,
                 resampling=DummyResamplingStrategy(),
                 operation=predict_mode,
                 acceleration=accel)
    @test e.measurement[1] ≈ 1.0
end

@testset_accelerated "sample weights in training and evaluation" accel begin
    yraw = ["Perry", "Antonia", "Perry", "Antonia", "Skater"]
    X = (x=rand(5),)
    y = categorical(yraw)
    w = [1, 10, 1, 10, 5]

    # without weights:
    mach = machine(ConstantClassifier(), X, y)
    e = evaluate!(mach, resampling=Holdout(fraction_train=0.6),
                  operation=predict_mode, measure=misclassification_rate,
                  acceleration=accel, verbosity=verb)
    @test e.measurement[1] ≈ 1.0

    # with weights in training and evaluation:
    mach = machine(ConstantClassifier(), X, y, w)
    e = evaluate!(mach, resampling=Holdout(fraction_train=0.6),
                  operation=predict_mode, measure=misclassification_rate,
                  acceleration=accel, verbosity=verb)
    @test e.measurement[1] ≈ 1/3

    # with weights in training but overriden in evaluation:
    e = evaluate!(mach, resampling=Holdout(fraction_train=0.6),
                  operation=predict_mode, measure=misclassification_rate,
                  weights = fill(1, 5), acceleration=accel, verbosity=verb)
    @test e.measurement[1] ≈ 1/2

    @test_throws(DimensionMismatch,
                 evaluate!(mach, resampling=Holdout(fraction_train=0.6),
                           operation=predict_mode,
                           measure=misclassification_rate,
                           weights = fill(1, 100), acceleration=accel,
                           verbosity=verb))

    @test_throws(ArgumentError,
                 evaluate!(mach, resampling=Holdout(fraction_train=0.6),
                           operation=predict_mode,
                           measure=misclassification_rate,
                           weights = fill('a', 5), acceleration=accel,
                           verbosity=verb))

    # resampling on a subset of all rows:
    model = @load KNNClassifier

    N = 200
    X = (x = rand(3N), );
    y = categorical(rand("abcd", 3N));
    w = rand(3N);
    rows = StatsBase.sample(1:3N, 2N, replace=false);
    Xsmall = selectrows(X, rows);
    ysmall = selectrows(y, rows);
    wsmall = selectrows(w, rows);

    mach1 = machine(model, Xsmall, ysmall, wsmall)
    e1 = evaluate!(mach1, resampling=CV(),
                   measure=misclassification_rate,
                   operation=predict_mode, acceleration=accel, verbosity=verb)

    mach2 = machine(model, X, y, w)
    e2 = evaluate!(mach2, resampling=CV(),
                   measure=misclassification_rate,
                   operation=predict_mode,
                   rows=rows, acceleration=accel, verbosity=verb)

    @test e1.per_fold ≈ e2.per_fold

    # resampler as machine with evaluation weights not specified:
    resampler = Resampler(model=model, resampling=CV();
                          measure=misclassification_rate,
                          operation=predict_mode)
    resampling_machine = machine(resampler, X, y, w)
    fit!(resampling_machine)
    e1 = evaluate(resampling_machine).measurement[1]
    mach = machine(model, X, y, w)
    e2 = evaluate!(mach, resampling=CV();
                   measure=misclassification_rate,
                   operation=predict_mode,
                   acceleration=accel, verbosity=verb).measurement[1]

    @test e1 ≈ e2

    # resampler as machine with evaluation weights specified:
    weval = rand(3N);
    resampler = Resampler(model=model, resampling=CV();
                          measure=misclassification_rate,
                          operation=predict_mode,
                          weights=weval, acceleration=accel)
    resampling_machine = machine(resampler, X, y, w)
    fit!(resampling_machine)
    e1   = evaluate(resampling_machine).measurement[1]
    mach = machine(model, X, y, w)
    e2   = evaluate!(mach, resampling=CV();
                     measure=misclassification_rate,
                     operation=predict_mode,
                     weights=weval,
                     acceleration=accel, verbosity=verb).measurement[1]

    @test e1 ≈ e2
end

#end
true
