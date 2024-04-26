using Distributed
import ComputationalResources: CPU1, CPUProcesses, CPUThreads
using .TestUtilities
using ProgressMeter
import Tables
@everywhere import StatisticalMeasures.StatisticalMeasuresBase as API
using StatisticalMeasures
import LearnAPI

@everywhere begin
    using .Models
    using StableRNGs
    rng = StableRNG(1513515)
    const verb = 0
end

using Test
using MLJBase
import Distributions
import StatsBase
@static if VERSION >= v"1.3.0-DEV.573"
    using .Threads
end

struct DummyInterval <: Interval end
dummy_interval=DummyInterval()

dummy_measure_det(yhat, y) = 42
API.@trait(
    typeof(dummy_measure_det),
    observation_scitype = MLJBase.Textual,
    kind_of_proxy = LearnAPI.LiteralTarget(),
)

dummy_measure_interval(yhat, y) = 42
API.@trait(
    typeof(dummy_measure_interval),
    observation_scitype = MLJBase.Textual,
    kind_of_proxy = LearnAPI.ConfidenceInterval(),
)

@testset "_actual_operations" begin
    clf = ConstantClassifier()
    rgs = ConstantRegressor()
    clf_det = DeterministicConstantClassifier()
    rgs_det = DeterministicConstantRegressor()
    measures = [LogLoss(), Accuracy(), BrierScore()] # mixed prob/determ
    measures_det = [Accuracy(), FScore()]
    operations = [predict, predict_mode, predict]

    # single measure gets replicated to match length of `measures`:
    @test MLJBase._actual_operations(predict_mean,
                                   [Accuracy(), FScore()],
                                    clf,
                                    1) ==
                                   [predict_mean, predict_mean]

    # handling of a measure with `nothing` `kind_of_proxy` (eg,
    # custom measure):
    my_mae(yhat, y) = abs.(yhat - y)
    @test(
        MLJBase._actual_operations(nothing, [my_mae, LPLoss()], rgs_det , 1) ==
        [predict, predict])

    @test MLJBase._actual_operations(predict, [LogLoss(),], clf, 1) ==
        [predict,]
    @test MLJBase._actual_operations(operations, measures, clf, 1) == operations
    @test_throws MLJBase.ERR_OPERATION_MEASURE_MISMATCH _ =
        MLJBase._actual_operations([predict, predict_mode], measures, clf, 1)
    @test_throws MLJBase.ERR_OPERATION_MEASURE_MISMATCH junk =
        MLJBase._actual_operations([predict,], measures, clf, 1)
    @test_throws MLJBase.ERR_INVALID_OPERATION _ =
        MLJBase._actual_operations(transform, [LogLoss(),], clf, 1)
    @test MLJBase._actual_operations(nothing, measures, clf, 1) == operations
    @test(
       @test_logs MLJBase._actual_operations(nothing, [Accuracy(),], clf, 1) ==
           [predict_mode])
    @test MLJBase._actual_operations(nothing, [l2,], rgs, 1) ==
        [predict_mean, ]
    @test_throws(
        MLJBase.err_incompatible_prediction_types(clf_det, LogLoss()),
        MLJBase._actual_operations(nothing, [LogLoss(),], clf_det, 1),
    )
    @test MLJBase._actual_operations(nothing, measures_det, clf_det, 1) ==
        [predict, predict]

    # measure/model differ in prediction type:
    @test_throws(
        MLJBase.err_ambiguous_operation(clf, dummy_measure_det),
        MLJBase._actual_operations(nothing, [dummy_measure_det, ], clf, 1),
    )

    # measure has :interval prediction type but model does not (2 cases):
    @test_throws(
        MLJBase.err_ambiguous_operation(clf, dummy_measure_interval),
        MLJBase._actual_operations(
            nothing,
            [dummy_measure_interval, ],
            clf,
            1,
        ),
    )
    @test_throws(
        MLJBase.err_ambiguous_operation(clf_det, dummy_measure_interval),
        MLJBase._actual_operations(nothing,
                                   [dummy_measure_interval, ], clf_det, 1))

    # both measure and model have :interval prediction type:
    @test MLJBase._actual_operations(nothing,
                               [dummy_measure_interval, ],
                               dummy_interval, 1) == [predict, ]

    # model has :interval prediction type but measure does not:
    @test_throws(
        MLJBase.err_ambiguous_operation(dummy_interval, LogLoss()),
        MLJBase._actual_operations(nothing,
                                   [LogLoss(), ], dummy_interval, 1))
end

@everywhere begin
    nfolds = 6
    nmeasures = 2
    func(mach, k) = (
        (sleep(MLJBase.PROG_METER_DT*rand(rng)); fill(1:k, nmeasures)),
        :fitted_params,
        :report,
    )
end
@testset_accelerated "dispatch of resources and progress meter" accel begin

    @info "Checking progress bars:"

    X = (x = [1, ],)
    y = [2.0, ]

    mach = machine(ConstantRegressor(), X, y)
    if accel isa CPUThreads
        result = MLJBase._evaluate!(
            func,
            mach,
            CPUThreads(Threads.nthreads()),
            nfolds,
            1
        )
    else
        result = MLJBase._evaluate!(func, mach, accel, nfolds, 1)
    end
    measurements = vcat(result[1]...)
    @test measurements ==
        [1:1, 1:1, 1:2, 1:2, 1:3, 1:3, 1:4, 1:4, 1:5, 1:5, 1:6, 1:6]
    @test collect(result[2]) == fill(:fitted_params, nfolds)
end

@test CV(nfolds=6) == CV(nfolds=6)
@test CV(nfolds=5) != CV(nfolds=6)
@test MLJBase.train_test_pairs(CV(), 1:10) !=
     MLJBase.train_test_pairs(CV(shuffle=true), 1:10)
@test MLJBase.train_test_pairs(Holdout(), 1:10) !=
     MLJBase.train_test_pairs(Holdout(shuffle=true), 1:10)

@testset "train test pairs" begin
    cv = CV(nfolds=5)

    pairs = MLJBase.train_test_pairs(cv, 1:24)
    @test pairs == [
        (6:24, 1:5),
        ([1:5..., 11:24...], 6:10),
        ([1:10..., 16:24...], 11:15),
        ([1:15..., 21:24...], 16:20),
        (1:20, 21:24)
    ]

    # Not enough data for the number of folds.
    @test_throws ArgumentError MLJBase.train_test_pairs(cv, 1:4)
end

@testset "checking measure/model compatibility" begin
    model = ConstantRegressor()
    y = rand(rng,4)

    # model prediction type is Probablistic but measure is Deterministic:
    @test_throws(
        MLJBase.ERR_MEASURES_PROBABILISTIC(rms, MLJBase.LOG_SUGGESTION2),
        MLJBase._check_measure(rms, predict, model, y),
    )

    @test MLJBase._check_measure(rms, predict_mean, model, y)

    @test MLJBase._check_measure(rms, predict_median, model, y)

    # has `y`  `Finite` elscitype but measure `rms` is for `Continuous`:
    y=categorical(collect("abc"))
    @test_throws(
        MLJBase.ERR_MEASURES_OBSERVATION_SCITYPE(
            rms,
            Union{Missing,Infinite},
            Multiclass{3},
        ),
        MLJBase._check_measure(rms, predict_median, model, y),
    )
    model = ConstantClassifier()
    # model prediction type is Probablistic but measure is Deterministic:
    @test_throws(
        MLJBase.ERR_MEASURES_PROBABILISTIC(mcr, MLJBase.LOG_SUGGESTION1),
        MLJBase._check_measure(mcr, predict, model, y),
    )

    @test MLJBase._check_measure(mcr, predict_mode, model, y)

    # `Determistic` model but `Probablistic` measure:
    model = DeterministicConstantClassifier()
    @test_throws(
        MLJBase.ERR_MEASURES_DETERMINISTIC(cross_entropy),
        MLJBase._check_measure(cross_entropy, predict, model, y),
    )

    # measure with wrong target_scitype:
    @test_throws(
        MLJBase.ERR_MEASURES_DETERMINISTIC(brier_score),
        MLJBase._check_measures(
            [brier_score, rms],
            [predict_mode, predict_mean],
            model, y,
        ),
    )

    model = ConstantClassifier()
    @test MLJBase._check_measures([brier_score, cross_entropy, accuracy],
                                  [predict, predict, predict_mode],
                                  model, coerce(y, Multiclass))
end

@testset "check weights" begin
    @test_throws(MLJBase.ERR_WEIGHTS_LENGTH,
                 MLJBase._check_weights([0.5, 0.5], 3))
    @test MLJBase._check_weights([0.5, 0.5], 2)
end

@testset "check class weights" begin
    w = Dict('a'=> 0.2, 'b'=>0.8)
    @test_throws(MLJBase.ERR_WEIGHTS_DICT,
                 MLJBase._check_class_weights([0.1, 0.4], ['a', 'b']))
    @test_throws(MLJBase.ERR_WEIGHTS_CLASSES,
                 MLJBase._check_class_weights(w, ['a', 'c']))
    @test MLJBase._check_class_weights(w, ['b', 'a'])
end

@everywhere begin
    user_rms(yhat, y) = mean((yhat -y).^2) |> sqrt
    # deliberately omitting `consumes_multiple_observations` trait:
    API.@trait typeof(user_rms) kind_of_proxy=LearnAPI.LiteralTarget()
end

@testset_accelerated "folds specified" accel begin
    x1 = ones(10)
    x2 = ones(10)
    X  = (x1=x1, x2=x2)
    y  = [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0]

    resampling = [(3:10, 1:2),
                  ([1, 2, 5, 6, 7, 8, 9, 10], 3:4),
                  ([1, 2, 3, 4, 7, 8, 9, 10], 5:6),
                  ([1, 2, 3, 4, 5, 6, 9, 10], 7:8),
                  (1:8, 9:10)]

    for cache in [true, false]

        model = DeterministicConstantRegressor()
        mach  = machine(model, X, y, cache=cache)

        # check detection of incompatible measure (cross_entropy):
        @test_throws(
            MLJBase.err_incompatible_prediction_types(model, cross_entropy),
            evaluate!(
                mach,
                resampling=resampling,
                measure=[cross_entropy, rmslp1],
                verbosity=verb,
                acceleration=accel,
            ),
        )
        result = evaluate!(mach, resampling=resampling, verbosity=verb,
                           measure=[user_rms, mae, rmslp1], acceleration=accel)

        v = [1/2, 3/4, 1/2, 3/4, 1/2]

        @test result.per_fold[1] ≈ v
        @test result.per_fold[2] ≈ v
        @test result.per_fold[3][1] ≈ abs(log(2) - log(2.5))
        @test result.per_observation[1] ≈ map(result.per_fold[1]) do μ
            fill(μ, 2)
        end
        @test result.per_observation[2][1] ≈ [1/2, 1/2]
        @test result.per_observation[2][2] ≈ [3/4, 3/4]
        @test result.measurement[1] ≈ mean(v)
        @test result.measurement[2] ≈ mean(v)

        # fitted_params and report per fold:
        @test map(fp->fp.fitresult, result.fitted_params_per_fold) ≈
            [1.5, 1.25, 1.5, 1.25, 1.5]
        @test all(isnothing, result.report_per_fold)
    end
end

@testset "folds specified - per_observation=false" begin
    accel = CPU1()
    cache = true
    x1 = ones(10)
    x2 = ones(10)
    X  = (x1=x1, x2=x2)
    y  = [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0]

    resampling = [(3:10, 1:2),
                  ([1, 2, 5, 6, 7, 8, 9, 10], 3:4),
                  ([1, 2, 3, 4, 7, 8, 9, 10], 5:6),
                  ([1, 2, 3, 4, 5, 6, 9, 10], 7:8),
                  (1:8, 9:10)]

    model = DeterministicConstantRegressor()
    mach  = machine(model, X, y, cache=cache)

    result = evaluate!(mach, resampling=resampling, verbosity=verb,
                       measure=[user_rms, mae, rmslp1], acceleration=accel,
                       per_observation=false)

    v = [1/2, 3/4, 1/2, 3/4, 1/2]

    @test result.per_fold[1] ≈ v
    @test result.per_fold[2] ≈ v
    @test result.per_fold[3][1] ≈ abs(log(2) - log(2.5))
    @test result.per_observation isa Vector{Missing}
    @test result.measurement[1] ≈ mean(v)
    @test result.measurement[2] ≈ mean(v)

    # fitted_params and report per fold:
    @test map(fp->fp.fitresult, result.fitted_params_per_fold) ≈
        [1.5, 1.25, 1.5, 1.25, 1.5]
    @test all(isnothing, result.report_per_fold)
end

@testset "repeated resampling" begin
    x1 = ones(20)
    x2 = ones(20)
    X = (x1=x1, x2=x2)
    y = rand(rng, 20)

    holdout = Holdout(fraction_train=0.75, rng=rng)
    model = Models.DeterministicConstantRegressor()

    for cache in [true, false]
        mach = machine(model, X, y, cache=cache)
        result = evaluate!(mach, resampling=holdout, verbosity=verb,
                       measure=[rms, rmslp1], repeats=6)
        per_fold = result.per_fold[1]
        @test unique(per_fold) |> length == 6
        @test abs(mean(per_fold) - std(y)) < 0.06 # very rough check

        cv = CV(nfolds=3, rng=rng)
        result = evaluate!(mach, resampling=cv, verbosity=verb,
                       measure=[rms, rmslp1], repeats=6)
        per_fold = result.per_fold[1]
        @test unique(per_fold) |> length == 18
        @test abs(mean(per_fold) - std(y)) < 0.06 # very rough check
    end
end

@testset_accelerated "holdout" accel begin
    x1 = ones(4)
    x2 = ones(4)
    X = (x1=x1, x2=x2)
    y = [1.0, 1.0, 2.0, 2.0]

    @test MLJBase.show_as_constructed(Holdout)
    holdout = Holdout(fraction_train=0.75)
    model = Models.DeterministicConstantRegressor()
    for cache in [true, false]
        mach = machine(model, X, y, cache=cache)
        # to see if a default measure is found:
        evaluate!(mach, resampling=holdout, verbosity=verb,
                  acceleration=accel)
        result = evaluate!(mach, resampling=holdout, verbosity=verb,
                           measure=[rms, rmslp1], acceleration=accel)
        @test result.measurement[1] ≈ 2/3

        # test direct evaluation of a model + data:
        result = evaluate(model, X, y, verbosity=0,
                          resampling=holdout, measure=rms, cache=cache)
        @test result.measurement[1] ≈ 2/3
    end

    X = (x=rand(rng,100),)
    y = rand(rng,100)

    for cache in [true, false]
        mach = machine(model, X, y, cache=cache)
        evaluate!(mach, verbosity=verb,
                  resampling=Holdout(shuffle=true, rng=rng), acceleration=accel)
        e1 = evaluate!(mach, verbosity=verb,
                       resampling=Holdout(shuffle=true),
                       acceleration=accel).measurement[1]
        @test e1 != evaluate!(mach, verbosity=verb,
                              resampling=Holdout(),
                              acceleration=accel).measurement[1]
    end
end

@testset_accelerated "Exception handling (see issue 235)" accel begin
    X, y = make_moons(50)
    model = ConstantClassifier()

    bad_loss(yhat, y) = throw(Exception())
    @test_throws Exception evaluate(model, X, y, measure=bad_loss, verbosity=0)
end

@testset_accelerated "cv" accel begin
    x1 = ones(10)
    x2 = ones(10)
    X = (x1=x1, x2=x2)
    y = [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0]

    @test MLJBase.show_as_constructed(CV)
    cv=CV(nfolds=5)
    for cache in [true, false]
        model = Models.DeterministicConstantRegressor()
        mach = machine(model, X, y, cache=cache)
        result = evaluate!(mach, resampling=cv, measure=[rms, rsq, rmslp1],
                           acceleration=accel, verbosity=verb)

        @test result.per_fold[1] ≈ [1/2, 3/4, 1/2, 3/4, 1/2]

        shuffled = evaluate!(mach, resampling=CV(shuffle=true), verbosity=verb,
                             acceleration=accel) # using rms default
        @test shuffled.measurement[1] != result.measurement[1]
    end
end

@testset "TimeSeriesCV" begin
    tscv = TimeSeriesCV(; nfolds=3)

    pairs = MLJBase.train_test_pairs(tscv, 1:10)
    @test pairs == [
        (1:4, [5, 6]),
        (1:6, [7, 8]),
        (1:8, [9, 10])
    ]

    pairs = MLJBase.train_test_pairs(tscv, 1:2:15)
    @test pairs == [
        ([1, 3], [5, 7])
        ([1, 3, 5, 7], [9, 11])
        ([1, 3, 5, 7, 9, 11], [13, 15])
    ]

    @test_logs(
        (:warn, "TimeSeriesCV is being applied to `rows` not in sequence. "),
        MLJBase.train_test_pairs(tscv, reverse(1:10))
    )

    # Not enough data for the number of folds.
    @test_throws ArgumentError MLJBase.train_test_pairs(TimeSeriesCV(10), 1:8)
end

@testset "stratified_cv" begin

    # check in explicit example:
    y = categorical(
        ['b', 'c', 'a', 'b', 'b', 'b', 'c', 'c', 'c', 'a', 'a', 'a']
    )
    scv = StratifiedCV(nfolds=3)
    rows = 1:12
    pairs = MLJBase.train_test_pairs(scv, rows, y)
    expected_pairs = [
        ([2, 4, 9, 11, 3, 5, 7, 12], [1, 6, 8, 10]),
        ([1, 6, 8, 10, 3, 5, 7, 12], [2, 4, 9, 11]),
        ([1, 6, 8, 10, 2, 4, 9, 11], [3, 5, 7, 12])
    ]
    # Explanation of expected_pairs: The row indices are processed one at
    # a time. The test fold that a row index is placed in is determined
    # by this lookup:
    #
    # b b b b c c c c a a a a
    # 1 2 3 1 2 3 1 2 3 1 2 3
    #
    # For example, the first row such that y[row] == 'c' is placed in the
    # second fold, and the second row such that y[row] == 'c' is placed in
    # the third fold.
    @test pairs == expected_pairs

    # test invariance to label renaming:
    z = replace(y, 'a' => 'b', 'b' => 'c', 'c' => 'a')
    pairs = MLJBase.train_test_pairs(scv, rows, z)
    @test pairs == expected_pairs

    # test the case where rows is a shuffled subset of y:
    y = categorical(['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c'])
    rows = 8:-1:2
    pairs = MLJBase.train_test_pairs(scv, rows, y)
    @test pairs == [
        ([5, 4, 6, 2], [8, 7, 3]),
        ([8, 7, 3, 6, 2], [5, 4]),
        ([8, 7, 3, 5, 4], [6, 2])
    ]

    # test shuffle:
    scv_random = StratifiedCV(nfolds=3, shuffle=true, rng=1)
    pairs_random = MLJBase.train_test_pairs(scv_random, rows, y)
    @test pairs != pairs_random

    # check class distribution is preserved in a larger randomized example:
    N = 30
    y = shuffle(vcat(fill('a', N), fill('b', 2N),
                        fill('c', 3N), fill('d', 4N))) |> categorical;
    d = Distributions.fit(MLJBase.UnivariateFinite, y)
    pairs = MLJBase.train_test_pairs(scv, 1:10N, nothing, y)
    folds = vcat(first.(pairs), last.(pairs))
    @test all([Distributions.fit(MLJBase.UnivariateFinite, y[fold]) ≈
               d for fold in folds])
end

@testset_accelerated "weights in evaluation" accel begin
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
    efold1 = mean([1*1, 1*0])
    efold2 = mean([3*3/2, 4*1/2])
    @test e ≈ mean([efold1, efold2])

    # if I don't specify weights in `evaluate!`, then uniform should
    # be used:
    e = evaluate!(mach, resampling=cv, measure=l1,
                  verbosity=verb, acceleration=accel).measurement[1]
    efold1 = mean([1*1, 1*0])
    efold2 = mean([1*3/2, 1*1/2])
    @test e ≈ mean([efold1, efold2])
end

@testset_accelerated "class weights in evaluation" accel begin
    X, y = make_blobs(rng=rng)
    cv=CV(nfolds = 2)
    fold1, fold2 = partition(eachindex(y), 0.5)
    m = MulticlassFScore()
    class_w = Dict(1=>1, 2=>2, 3=>3)

    model = Models.DeterministicConstantClassifier()
    mach = machine(model, X, y)

    # fscore by hand:
    fit!(mach, rows=fold1, verbosity=0)
    score1 = m(predict(mach, rows=fold2), y[fold2], class_w)
    fit!(mach, rows=fold2, verbosity=0)
    score2 = m(predict(mach, rows=fold1), y[fold1], class_w)
    score_by_hand = mean([score1, score2])

    # fscore by evaluate!:
    score = evaluate!(
        mach,
        resampling=cv,
        measure=m,
        class_weights=class_w,
        verbosity=verb,
        acceleration=accel,
    ).measurement[1]

    @test score ≈ score_by_hand

    # if class weights in `evaluate!` isn't specified:
    plain_score = evaluate!(
        mach,
        resampling=cv,
        measure=m,
        verbosity=verb,
        acceleration=accel,
    ).measurement[1]
    @test !(score ≈ plain_score)
end

@testset_accelerated "resampler as machine" accel begin
    N = 50
    X = (x1=rand(rng, N), x2=rand(rng, N), x3=rand(rng, N))
    y = X.x1 -2X.x2 + 0.05*rand(rng, N)
    ridge_model = FooBarRegressor(lambda=20.0)
    holdout = Holdout(fraction_train=0.75)
    resampler = Resampler(resampling=holdout, model=ridge_model, measure=mae,
                          acceleration=accel)
    resampling_machine = machine(resampler, X, y)
    @test_logs((:info, r"^Training"), fit!(resampling_machine))
    e1=evaluate(resampling_machine).measurement[1]
    mach = machine(ridge_model, X, y)
    @test e1 ≈  evaluate!(mach, resampling=holdout,
                          measure=mae, verbosity=verb,
                          acceleration=CPU1()).measurement[1]
    ridge_model.lambda=1.0
    fit!(resampling_machine, verbosity=verb)
    e2=evaluate(resampling_machine).measurement[1]
    @test e1 != e2
    resampler.weights = rand(rng,N)
    fit!(resampling_machine, verbosity=verb)
    e3=evaluate(resampling_machine).measurement[1]
    @test e3 != e2

    @test MLJBase.package_name(Resampler) == "MLJBase"
    @test MLJBase.is_wrapper(Resampler)

    # when only `model` changes, the folds shouldn't change, even in
    # shuffled case:
    cv = CV(rng=StableRNGs.StableRNG(123))
    resampler=Resampler(model=ridge_model,
                        resampling=cv,
                        repeats=3,
                        measure=mae, acceleration=accel)
    mach = machine(resampler, X, y)
    fit!(mach, verbosity=verb)
    ev1 = evaluate(mach)
    rows1 = ev1.train_test_rows
    resampler.model.lambda *= 0.5
    fit!(mach, verbosity=verb)
    ev2 = evaluate(mach)
    @test rows1 == ev2.train_test_rows
    resampler.model.lambda *= 2
    fit!(mach, verbosity=verb)
    @test ev1.measurement[1] ≈ evaluate(mach).measurement[1]

    # but if `resampling` or `repeats`, then new
    # folds should be generated:
    resampler.resampling = CV(rng=cv.rng, nfolds=2)
    fit!(mach, verbosity=verb)
    rows2 = evaluate(mach).train_test_rows
    @test length(rows2) == 2 * resampler.repeats
    @test rows2 != rows1
    resampler.repeats += 1
    fit!(mach, verbosity=verb)
    rows3 = evaluate(mach).train_test_rows
    @test length(rows3) == 2 * resampler.repeats
end

struct DummyResamplingStrategy <: MLJBase.ResamplingStrategy end

function MLJBase.train_test_pairs(resampling::DummyResamplingStrategy,
                                  rows, X, y)
    train = filter(rows) do j
        y[j] == y[1]
        end
    test = setdiff(rows, train)
    return [(train, test),]
end
@testset_accelerated "custom strategy depending on X, y" accel begin

    X = (x = rand(rng,8), )
    y = categorical(string.([:x, :y, :x, :x, :y, :x, :x, :y]))
    @test MLJBase.train_test_pairs(DummyResamplingStrategy(), 2:6, X, y) ==
        [([3, 4, 6], [2, 5]),]

    e = evaluate(ConstantClassifier(), X, y,
                 measure=misclassification_rate,
                 resampling=DummyResamplingStrategy(),
                 operation=predict_mode,
                 acceleration=accel,
                 verbosity=verb)
    @test e.measurement[1] ≈ 1.0
end

@testset_accelerated "sample weights in training and evaluation" accel begin
    yraw = ["Perry", "Antonia", "Perry", "Antonia", "Skater"]
    X = (x=rand(rng,5),)
    y = categorical(yraw)
    w = [1, 10, 1, 10, 5]

    for cache in [true, false]
        # without weights:
        mach = machine(ConstantClassifier(), X, y, cache=cache)
        e = evaluate!(mach, resampling=Holdout(fraction_train=0.6),
                      operation=predict_mode, measure=misclassification_rate,
                      acceleration=accel, verbosity=verb)
        @test e.measurement[1] ≈ 1.0

        # with weights in training and evaluation:
        mach = machine(ConstantClassifier(), X, y, w, cache=cache)
        e = evaluate!(mach, resampling=Holdout(fraction_train=0.6),
                      operation=predict_mode, measure=misclassification_rate,
                      acceleration=accel, verbosity=verb, weights=w)
        @test e.measurement[1] ≈ mean([10*0, 5*1])

        # with different weights in training and evaluation:
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
    end

    # resampling on a subset of all rows:
    model = KNNClassifier()

    N = 200
    X = (x = rand(rng,3N), );
    y = categorical(rand(rng,"abcd", 3N));
    w = rand(rng,3N);
    class_w = Dict(zip(levels(y), rand(length(levels(y)))));
    rows = StatsBase.sample(1:3N, 2N, replace=false);
    Xsmall = selectrows(X, rows);
    ysmall = selectrows(y, rows);
    wsmall = selectrows(w, rows);

    for cache in [true, false]
        mach1 = machine(model, Xsmall, ysmall, wsmall, cache=cache)
        e1 = evaluate!(mach1,
                       resampling=CV(),
                       measure=misclassification_rate,
                       weights=wsmall,
                       operation=predict_mode,
                       acceleration=accel,
                       verbosity=verb)

        mach2 = machine(model, X, y, w, cache=cache)
        e2 = evaluate!(mach2,
                       resampling=CV(),
                       measure=misclassification_rate,
                       weights=w,
                       operation=predict_mode,
                       rows=rows,
                       acceleration=accel,
                       verbosity=verb)

        @test e1.per_fold ≈ e2.per_fold
    end

    for cache in [true, false]

        # resampler as machine with evaluation weights not specified:
        resampler = Resampler(model=model, resampling=CV();
                              measure=misclassification_rate,
                              operation=predict_mode,
                              cache=cache)
        resampling_machine = machine(resampler, X, y, w, cache=false)
        fit!(resampling_machine, verbosity=verb)
        e1 = evaluate(resampling_machine).measurement[1]
        mach = machine(model, X, y, w, cache=cache)
        e2 = evaluate!(mach, resampling=CV();
                       measure=misclassification_rate,
                       operation=predict_mode,
                       acceleration=accel, verbosity=verb).measurement[1]

        @test e1 ≈ e2

        # resampler as machine with evaluation weights specified:
        weval = rand(rng,3N);
        resampler = Resampler(model=model, resampling=CV();
                              measure=misclassification_rate,
                              operation=predict_mode,
                              weights=weval, acceleration=accel,
                              cache=cache)
        resampling_machine = machine(resampler, X, y, w, cache=false)
        fit!(resampling_machine, verbosity=verb)
        e1   = evaluate(resampling_machine).measurement[1]
        mach = machine(model, X, y, w, cache=cache)
        e2   = evaluate!(mach, resampling=CV();
                         measure=misclassification_rate,
                         operation=predict_mode,
                         weights=weval,
                         acceleration=accel,
                         verbosity=verb).measurement[1]

        @test e1 ≈ e2

    end

    x = [1,2,3,4,5,6,7]
    X, y = Tables.table([x x x x x x]), coerce([1,2,1,3,1,2,2], Multiclass)
    model = Models.DeterministicConstantClassifier()
    class_w = Dict(zip(levels(y), rand(length(levels(y)))))

    for cache in [true, false]

        #resampler as a machine with class weights specified
        cweval = Dict(zip(levels(y), rand(length(levels(y)))));
        resampler = Resampler(model=model, resampling=CV(nfolds=2);
                              measure=MulticlassFScore(return_type=Vector),
                              class_weights=cweval, acceleration=accel)
        resampling_machine = machine(resampler, X, y, cache=false)
        fit!(resampling_machine, verbosity=verb)
        e1   = evaluate(resampling_machine).measurement[1]
        mach = machine(model, X, y, cache=cache)
        e2   = evaluate!(mach, resampling=CV(nfolds=2);
                         measure=MulticlassFScore(return_type=Vector),
                         class_weights=cweval,
                         acceleration=accel,
                         verbosity=verb).measurement[1]

        @test e1 ≈ e2

    end

    @testset "warnings about measures not supporting weights" begin
        model = ConstantClassifier()
        N = 100
        X, y = make_moons(N)
        class_weights = Dict(0=>0.4, 1=>0.6)
        @test_logs((:warn, r"Sample weights"),
                   evaluate(model, X, y,
                            resampling=Holdout(fraction_train=0.5),
                            measure=auc, weights=ones(N)))
        @test_logs((:warn, r"Class weights"),
                   evaluate(model, X, y,
                            resampling=Holdout(fraction_train=0.5),
                            measure=auc, class_weights=class_weights))
    end

end

@testset_accelerated "automatic operations - integration" accel begin
    clf = ConstantClassifier()
    X, y = make_moons(100)
    e1 = evaluate(clf, X, y, resampling=CV(),
                  measures=[LogLoss(), Accuracy()], verbosity=1)
    e2 = evaluate(clf, X, y, resampling=CV(),
                  operation=[predict, predict_mode],
                  measures=[LogLoss(), Accuracy()], verbosity=1)
    @test e1.measurement ≈ e2.measurement
    evaluate(clf, X, y, resampling=CV(),
             operation=predict,
             measures=[LogLoss(), BrierScore()], verbosity=0)
end

@testset "reported fields in documentation" begin
    # Using `evaluate` to obtain a `PerformanceEvaluation` object.
    clf = ConstantClassifier()
    X, y = make_moons(100)
    y = coerce(y, OrderedFactor)
    evaluations = evaluate(clf, X, y, resampling=CV())
    T = typeof(evaluations)
    @test T <: PerformanceEvaluation

    show_text = sprint(show, MIME"text/plain"(), evaluations)
    cols = ["measure", "operation", "measurement", "1.96*SE", "per_fold"]
    @test all(contains.(show_text, cols))
    print(show_text)
    docstring_text = string(@doc(PerformanceEvaluation))
    for fieldname in fieldnames(PerformanceEvaluation)
        @test contains(show_text, string(fieldname))
        # string(text::Markdown.MD) converts `-` list items to `*`.
        @test contains(docstring_text, " * `$fieldname`")
    end

    measures = [LogLoss(), Accuracy()]
    evaluations = evaluate(clf, X, y; measures, resampling=Holdout())
    show_text = sprint(show, MIME"text/plain"(), evaluations)
    @test !contains(show_text, "std")

    # issue #871: trying to calculate SE when inappropriate should not throw an error in
    # display.
    evaluations = evaluate(
        clf,
        X,
        y,
        operation=predict_mode,
        measure=ConfusionMatrix(),
        resampling=CV(),
    );
    printed_evaluations = sprint(show, "text/plain", evaluations)
    @test contains(printed_evaluations, "N/A")
end

@testset_accelerated "issue with Resampler #954" acceleration begin
    knn = KNNClassifier()
    cnst =DeterministicConstantClassifier()
    X, y = make_blobs(10)

    resampler = MLJBase.Resampler(
        ;model=knn,
        measure=accuracy,
        operation=nothing,
        acceleration,
    )
    mach = machine(resampler, X, y) |> fit!

    resampler.model = cnst
    fit!(mach)
end

@testset "compact evaluation objects" begin
    model = ConstantClassifier()
    X, y = make_blobs(10)
    e = evaluate(model, X, y)
    ec = evaluate(model, X, y, compact=true)
    @test e isa PerformanceEvaluation
    @test ec isa CompactPerformanceEvaluation
    @test startswith(sprint(show, MIME("text/plain"), e), "PerformanceEvaluation")
    @test startswith(sprint(show, MIME("text/plain"), ec), "CompactPerformanceEvaluation")
    @test e.measurement[1] == ec.measurement[1]

    # smoke tests:
    mach = machine(model, X, y)
    for e in [
        evaluate!(mach, measures=[brier_loss, accuracy]),
        evaluate!(mach, measures=[brier_loss, accuracy], compact=true),
        evaluate!(mach, resampling=CV(), measures=[brier_loss, accuracy]),
        evaluate!(mach, resampling=CV(), measures=[brier_loss, accuracy], compact=true),
        ]
        @test contains(sprint(show, MIME("text/plain"), e), "predict")
        @test contains(sprint(show, MIME("text/plain"), e), "[")
        @test contains(sprint(show, e), "PerformanceEvaluation(")
        @test !contains(sprint(show, e), "predict")
    end
end

true
