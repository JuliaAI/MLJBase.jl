rng = StableRNG(51803)

@testset "built-in classifier measures" begin
    y    = categorical(collect("asdfasdfaaassdd"))
    yhat = categorical(collect("asdfaadfaasssdf"))
    w = 1:15
    @test misclassification_rate(yhat, y) ≈ 0.2
    @test misclassification_rate(yhat, y, w) ≈ (6*1 + 11*1 + 15*1) / 15
    y = categorical(collect("abb"))
    L = [y[1], y[2]]
    d1 = UnivariateFinite(L, [0.1, 0.9]) # a
    d2 = UnivariateFinite(L, Float32[0.4, 0.6]) # b
    d3 = UnivariateFinite(L, [0.2, 0.8]) # b
    yhat = [d1, d2, d3]
    @test mean(cross_entropy(yhat, y)) ≈
        Float32(-(log(0.1) + log(0.6) + log(0.8))/3)
    yhat = UnivariateFinite(L, [0.1 0.9;
                                0.4 0.6;
                                0.2 0.8])
    @test isapprox(mean(cross_entropy(yhat, y)),
                   -(log(0.1) + log(0.6) + log(0.8))/3, atol=eps(Float32))
    # sklearn test
    # >>> from sklearn.metrics import log_loss
    # >>> log_loss(["spam", "ham", "ham", "spam","ham","ham"],
    #    [[.1, .9], [.9, .1], [.8, .2], [.35, .65], [0.2, 0.8], [0.3,0.7]])
    # 0.6130097025803921
    y2 = categorical(["spam", "ham", "ham", "spam", "ham", "ham"])
    L2 = classes(y2[1])
    probs = vcat([.1 .9], [.9 .1], [.8 .2], [.35 .65], [0.2 0.8], [0.3 0.7])
    yhat2 = UnivariateFinite(L2, probs)
    @test mean(cross_entropy(yhat2, y2)) ≈ 0.6130097025803921
    # BrierScore
    scores = BrierScore()(yhat, y)
    @test size(scores) == size(y)
    @test Float32.(scores) ≈ [-1.62, -0.32, -0.08]
    # sklearn test
    # >>> from sklearn.metrics import brier_score_loss
    # >>> brier_score_loss([1, 0, 0, 1, 0, 0], [.9, .1, .2, .65, 0.8, 0.7])
    # 0.21875 NOTE: opposite orientation
    @test -mean(BrierScore()(yhat2, y2)) / 2 ≈ 0.21875
    probs2 = [[.1, .9], [Float32(0.9), Float32(1) - Float32(0.9)], [.8, .2],
              [.35, .65], [0.2, 0.8], [0.3, 0.7]]
    yhat3 = [UnivariateFinite(L2, prob) for prob in probs2]
    @test -mean(BrierScore()(yhat3, y2) / 2) ≈ 0.21875

end

@testset "mcr, acc, bacc, mcc" begin
    y = categorical(['m', 'f', 'n', 'f', 'm', 'n', 'n', 'm', 'f'])
    ŷ = categorical(['f', 'f', 'm', 'f', 'n', 'm', 'n', 'm', 'f'])
    @test accuracy(ŷ, y) == 1-mcr(ŷ,y) ==
            accuracy(confmat(ŷ, y, warn=false))  == 1-mcr(confmat(ŷ, y, warn=false))
    w = randn(rng,length(y))
    @test accuracy(ŷ, y, w) == 1-mcr(ŷ,y,w)

    ## balanced accuracy
    y = categorical([3, 4, 1, 1, 1, 4, 1, 3, 3, 1, 2, 3, 1, 3, 3, 3, 2, 4, 3, 2, 1, 3,
       3, 1, 1, 1, 2, 4, 1, 4, 4, 4, 1, 1, 4, 4, 3, 1, 2, 2, 3, 4, 2, 1,
       2, 2, 3, 2, 2, 3, 1, 2, 3, 4, 1, 2, 4, 2, 1, 4, 3, 2, 3, 3, 3, 1,
       3, 1, 4, 3, 1, 2, 3, 1, 2, 2, 4, 4, 1, 3, 2, 1, 4, 3, 3, 1, 3, 1,
       2, 2, 2, 2, 2, 3, 2, 1, 1, 4, 2, 2])
    ŷ = categorical([2, 3, 2, 1, 2, 2, 3, 3, 2, 4, 2, 3, 2, 4, 3, 4, 4, 2, 1, 3, 3, 3,
       3, 3, 2, 4, 4, 3, 4, 4, 1, 2, 3, 2, 4, 1, 2, 3, 1, 4, 2, 2, 1, 2,
       3, 2, 2, 4, 3, 2, 2, 2, 1, 2, 2, 1, 3, 1, 4, 1, 2, 1, 2, 4, 3, 2,
       4, 3, 2, 4, 4, 2, 4, 3, 2, 3, 1, 2, 1, 2, 1, 2, 3, 1, 1, 3, 4, 2,
       4, 4, 2, 1, 3, 2, 2, 4, 1, 1, 4, 1])
    sk_bacc = 0.17493386243386244
    @test bacc(ŷ, y) ≈ sk_bacc
    w = [0.5, 1.4, 0.6, 1. , 0.1, 0.5, 1.2, 0.2, 1.8, 0.3, 0.6, 2.2, 0.1,
       1.4, 0.2, 0.4, 0.6, 2.1, 0.7, 0.2, 0.9, 0.4, 0.7, 0.3, 0.1, 1.7,
       0.2, 0.7, 1.2, 1. , 0.9, 0.4, 0.5, 0.5, 0.5, 1. , 0.3, 0.1, 0.2,
       0. , 2.2, 0.8, 0.9, 0.8, 1.3, 0.2, 0.4, 0.7, 1. , 0.7, 1.7, 0.7,
       1.1, 1.8, 0.1, 1.2, 1.8, 1. , 0.1, 0.5, 0.6, 0.7, 0.6, 1.2, 0.6,
       1.2, 0.5, 0.5, 0.8, 0.2, 0.6, 1. , 0.3, 1. , 0.2, 1.1, 1.1, 1.1,
       0.6, 1.4, 1.2, 0.3, 1.1, 0.2, 0.5, 1.6, 0.3, 1. , 0.3, 0.9, 0.9,
       0. , 0.6, 0.6, 0.4, 0.5, 0.4, 0.2, 0.9, 0.4]
    sk_bacc_w = 0.1581913163016446
    @test bacc(ŷ, y, w) ≈ sk_bacc_w

    sk_mcc = -0.09759509982785947
    @test mcc(ŷ, y) == matthews_correlation(ŷ, y) ≈ sk_mcc
    # invariance with respect to permutation ?
    cm = confmat(ŷ, y, perm=[3, 1, 2, 4])
    @test mcc(cm) ≈ sk_mcc
end

@testset "AUC" begin
    # this is random binary and random scores generated with numpy
    # then using roc_auc_score from sklearn to get the AUC
    # we check that we recover a comparable AUC and that it's invariant
    # to ordering.
    c = ["neg", "pos"]
    y = categorical(c[[0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
                     1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
                     1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0,
                     1, 0] .+ 1])
    probs = [
        0.90237535, 0.41276349, 0.94511611, 0.08390761, 0.55847392,
        0.26043136, 0.78565351, 0.20133953, 0.7404382 , 0.15307601,
        0.59596716, 0.8169512 , 0.88200483, 0.23321489, 0.94050483,
        0.27593662, 0.60702176, 0.36427036, 0.35481784, 0.06416543,
        0.45576954, 0.12354048, 0.79830435, 0.15799818, 0.20981099,
        0.43451663, 0.24020098, 0.11401055, 0.25785748, 0.86490263,
        0.75715379, 0.06550534, 0.12628999, 0.18878245, 0.1283757 ,
        0.76542903, 0.8780248 , 0.86891113, 0.24835709, 0.06528076,
        0.72061354, 0.89451634, 0.95634394, 0.07555979, 0.16345437,
        0.43498831, 0.37774708, 0.31608861, 0.41369339, 0.95691113]

    ŷ = UnivariateFinite(y[1:2], probs, augment=true)
    # ŷ = [UnivariateFinite(y[1:2], [1.0 - p, p]) for p in [
    #     0.90237535, 0.41276349, 0.94511611, 0.08390761, 0.55847392,
    #     0.26043136, 0.78565351, 0.20133953, 0.7404382 , 0.15307601,
    #     0.59596716, 0.8169512 , 0.88200483, 0.23321489, 0.94050483,
    #     0.27593662, 0.60702176, 0.36427036, 0.35481784, 0.06416543,
    #     0.45576954, 0.12354048, 0.79830435, 0.15799818, 0.20981099,
    #     0.43451663, 0.24020098, 0.11401055, 0.25785748, 0.86490263,
    #     0.75715379, 0.06550534, 0.12628999, 0.18878245, 0.1283757 ,
    #     0.76542903, 0.8780248 , 0.86891113, 0.24835709, 0.06528076,
    #     0.72061354, 0.89451634, 0.95634394, 0.07555979, 0.16345437,
    #     0.43498831, 0.37774708, 0.31608861, 0.41369339, 0.95691113]]
    @test isapprox(auc(ŷ, y), 0.455716, rtol=1e-4)
    ŷ_unwrapped = [ŷ...]
    @test isapprox(auc(ŷ_unwrapped, y), 0.455716, rtol=1e-4)

    # reversing the roles of positive and negative should return very
    # similar score
    y2 = deepcopy(y);
    levels!(y2, reverse(levels(y2)));
    @test y == y2
    @test levels(y) != levels(y2)
    ŷ2 = UnivariateFinite(y2[1:2], probs, augment=true) # same probs
    @test isapprox(auc(ŷ2, y2), auc(ŷ, y), rtol=1e-4)

end

@testset "confusion matrix {2}" begin
    # first class is 1 is assumed negative, second positive
    y = categorical([1, 2, 1, 2, 1, 1, 2])
    ŷ = categorical([1, 2, 2, 2, 2, 1, 2])
    cm = confmat(ŷ, y, warn=false)
    TN = sum(ŷ .== y .== 1) # pred and true = - (1)
    TP = sum(ŷ .== y .== 2) # pred and true = + (2)
    FP = sum(ŷ .!= y .== 1) # pred + (2) and true - (1)
    FN = sum(ŷ .!= y .== 2) # pred - (1) and true + (2)
    @test cm[1,1] == TN
    @test cm[2,2] == TP
    @test cm[1,2] == FN
    @test cm[2,1] == FP

    cm2 = confmat(ŷ, y; rev=true)
    @test cm2[1,1] == cm[2,2]
    @test cm2[1,2] == cm[2,1]
    @test cm2[2,2] == cm[1,1]
    @test cm2[2,1] == cm[1,2]

    @test accuracy(ŷ, y) == accuracy(cm) == sum(y .== ŷ) / length(y)

    @test @test_logs (:warn, "The classes are un-ordered,\nusing: negative='1' and positive='2'.\nTo suppress this warning, consider coercing to OrderedFactor.") recall(ŷ, y) == TP / (TP + FN)

    ŷ = coerce(ŷ, OrderedFactor)
    y = coerce(y, OrderedFactor)

    @test precision(ŷ, y)   == TP / (TP + FP)
    @test specificity(ŷ, y) == TN / (TN + FP)
    @test f1score(ŷ, y)     == 2.0 / (1.0 / recall(ŷ, y) + 1.0 / precision(ŷ, y))

    recall_rev = Recall(rev=true)
    @test recall_rev(ŷ, y) == TN / (TN + FP) # no warning because rev is specified
    precision_rev = Precision(rev=true)
    @test precision_rev(ŷ, y) == TN / (TN + FN)
    specificity_rev = Specificity(rev=true)
    @test specificity_rev(ŷ, y) == TP / (TP + FN)
    f1score_rev = FScore{1}(rev=true)
    @test f1score_rev(ŷ, y) == 2.0 / (1.0 / recall_rev(ŷ, y) + 1.0 / precision_rev(ŷ, y))
end

@testset "confusion matrix {n}" begin
    y = coerce([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], Multiclass)
    ŷ = coerce([0, 1, 0, 0, 0, 2, 1, 2, 0, 1, 1, 2], Multiclass)
    class_w = Dict(0=>0,2=>2,1=>1)
    cm = confmat(ŷ, y, warn=false)

    #               ┌─────────────────────────────────────────┐
    #               │              Ground Truth               │
    # ┌─────────────┼─────────────┬─────────────┬─────────────┤
    # │  Predicted  │      0      │      1      │      2      │
    # ├─────────────┼─────────────┼─────────────┼─────────────┤
    # │      0      │      2      │      1      │      2      │
    # ├─────────────┼─────────────┼─────────────┼─────────────┤
    # │      1      │      2      │      2      │      0      │
    # ├─────────────┼─────────────┼─────────────┼─────────────┤
    # │      2      │      0      │      1      │      2      │
    # └─────────────┴─────────────┴─────────────┴─────────────┘

    #`no_avg` and `LittleDict`
    @test collect(values(MulticlassPrecision(average=no_avg)(cm))) ≈
        collect(values(MulticlassPrecision(average=no_avg)(ŷ, y))) ≈
        [0.4; 0.5; 2/3]
    @test MulticlassPrecision(average=macro_avg)(cm) ≈
        MulticlassPrecision(average=macro_avg)(ŷ, y) ≈ mean([0.4; 0.5; 2/3])
    @test collect(keys(MulticlassPrecision(average=no_avg)(cm)))  ==
        collect(keys(MulticlassPrecision(average=no_avg)(ŷ, y))) ==
        ["0"; "1"; "2"]
    @test collect(values(MulticlassRecall(average=no_avg)(cm))) ≈
        collect(values(MulticlassRecall(average=no_avg)(ŷ, y))) ≈
        [0.5; 0.5; 0.5]
    @test collect(values(MulticlassFScore(average=no_avg)(cm))) ≈
        collect(values(MulticlassFScore(average=no_avg)(ŷ, y))) ≈
        [4/9; 0.5; 4/7]

    #`no_avg` and `LittleDict` with class weights
    @test collect(values(MulticlassPrecision(average=no_avg)(cm, class_w))) ≈
        collect(values(MulticlassPrecision(average=no_avg)(ŷ, y, class_w))) ≈
        [0.4; 0.5; 2/3] .* [0; 1; 2]
    @test collect(values(MulticlassRecall(average=no_avg)(cm, class_w))) ≈
        collect(values(MulticlassRecall(average=no_avg)(ŷ, y, class_w))) ≈
        [0.5; 0.5; 0.5] .* [0; 1; 2]
    @test collect(values(MulticlassFScore(average=no_avg)(cm, class_w))) ≈
        collect(values(MulticlassFScore(average=no_avg)(ŷ, y, class_w))) ≈
        [4/9; 0.5; 4/7] .* [0; 1; 2]

    #`macro_avg` and `LittleDict`
    macro_prec = MulticlassPrecision(average=macro_avg)
    macro_rec  = MulticlassRecall(average=macro_avg)

    @test macro_prec(cm)    ≈ macro_prec(ŷ, y)    ≈ mean([0.4, 0.5, 2/3])
    @test macro_rec(cm)     ≈ macro_rec(ŷ, y)     ≈ mean([0.5; 0.5; 0.5])
    @test macro_f1score(cm) ≈ macro_f1score(ŷ, y) ≈ mean([4/9; 0.5; 4/7])

    #`micro_avg` and `LittleDict`
    micro_prec = MulticlassPrecision(average=micro_avg)
    micro_rec  = MulticlassRecall(average=micro_avg)

    @test micro_prec(cm)    == micro_prec(ŷ, y)    == 0.5
    @test micro_rec(cm)     == micro_rec(ŷ, y)     == 0.5
    @test micro_f1score(cm) == micro_f1score(ŷ, y) == 0.5

    #`no_avg` and `AbstractVector` with class weights
    vec_precision = MulticlassPrecision(return_type=AbstractVector)
    vec_recall    = MulticlassRecall(return_type=AbstractVector)
    vec_f1score   = MulticlassFScore(return_type=AbstractVector)

    @test vec_precision(cm, class_w) ≈ vec_precision(ŷ, y, class_w) ≈
        mean([0.4; 0.5; 2/3] .* [0; 1; 2])
    @test vec_recall(cm, class_w)    ≈ vec_recall(ŷ, y, class_w)    ≈
        mean([0.5; 0.5; 0.5] .* [0; 1; 2])
    @test vec_f1score(cm, class_w)   ≈ vec_f1score(ŷ, y, class_w)   ≈
        mean([4/9; 0.5; 4/7] .* [0; 1; 2])

    #`macro_avg` and `AbstractVector`
    v_ma_prec = MulticlassPrecision(average=macro_avg,
                                    return_type=AbstractVector)
    v_ma_rec  = MulticlassRecall(average=macro_avg, return_type=AbstractVector)
    v_ma_f1   = MulticlassFScore(average=macro_avg, return_type=AbstractVector)

    @test v_ma_prec(cm) ≈ v_ma_prec(ŷ, y) ≈ mean([0.4, 0.5, 2/3])
    @test v_ma_rec(cm)  ≈ v_ma_rec(ŷ, y)  ≈ mean([0.5; 0.5; 0.5])
    @test v_ma_f1(cm)   ≈ v_ma_f1(ŷ, y)   ≈ mean([4/9; 0.5; 4/7])

    #`macro_avg` and `AbstractVector` with class weights
    @test v_ma_prec(cm, class_w) ≈ v_ma_prec(ŷ, y, class_w) ≈
        mean([0.4, 0.5, 2/3] .* [0, 1, 2])
    @test v_ma_rec(cm, class_w)  ≈ v_ma_rec(ŷ, y, class_w)  ≈
        mean([0.5; 0.5; 0.5] .* [0, 1, 2])
    @test v_ma_f1(cm, class_w)   ≈ v_ma_f1(ŷ, y, class_w)   ≈
        mean([4/9; 0.5; 4/7] .* [0, 1, 2])

    #`micro_avg` and `AbstractVector`
    v_mi_prec = MulticlassPrecision(average=micro_avg, return_type=AbstractVector)
    v_mi_rec  = MulticlassRecall(average=micro_avg, return_type=AbstractVector)
    v_mi_f1   = MulticlassFScore(average=micro_avg, return_type=AbstractVector)

    @test v_mi_prec(cm) == v_mi_prec(ŷ, y) == 0.5
    @test v_mi_rec(cm)  == v_mi_rec(ŷ, y)  == 0.5
    @test v_mi_f1(cm)   == v_mi_f1(ŷ, y)   == 0.5
end

@testset "Metadata binary" begin
    for m in (accuracy, recall, Precision(), f1score, specificity)
        e = info(m)
        m == accuracy    && (@test e.name == "accuracy")
        m == recall      && (@test e.name == "true_positive_rate")
        m isa Precision  && (@test e.name == "positive_predictive_value")
        m == f1score     && (@test e.name == "FScore{1}")
        m == specificity && (@test e.name == "true_negative_rate")
        @test e.target_scitype <: AbstractVector{<:Finite}
        @test e.prediction_type == :deterministic
        @test e.orientation == :score
        @test e.reports_each_observation == false
        @test e.is_feature_dependent == false
        if m == accuracy
            @test e.supports_weights
        else
            @test !e.supports_weights
        end
    end
    e = info(auc)
    @test e.name == "area_under_curve"
    @test e.target_scitype == AbstractVector{<:Finite{2}}
    @test e.prediction_type == :probabilistic
    @test e.reports_each_observation == false
    @test e.is_feature_dependent == false
    @test e.supports_weights == false
end

@testset "Metadata multiclass" begin
    for m in (MulticlassRecall(), MulticlassPrecision(),
              MulticlassFScore(), MulticlassTrueNegativeRate())
        e = info(m)
        m isa MulticlassRecall &&
            (@test e.name == "MulticlassTruePositiveRate")
        m isa MulticlassPrecision   &&
            (@test e.name == "MulticlassPrecision")
        m isa MulticlassFScore &&
            (@test e.name == "MulticlassFScore")
        m isa MulticlassTrueNegativeRate &&
            (@test e.name == "MulticlassTrueNegativeRate")
        @test e.target_scitype <: AbstractVector{<:Finite}
        @test e.prediction_type == :deterministic
        @test e.orientation == :score
        @test e.reports_each_observation == false
        @test e.is_feature_dependent == false
        @test e.supports_weights == false
        @test e.supports_class_weights == true
    end
end

@testset "More binary metrics" begin
    y = coerce(categorical([1, 2, 1, 2, 1, 1, 2, 1, 2, 2, 2, 1, 2,
                            2, 1, 2, 1, 1, 1, 2, 1, 2, 2, 1, 2, 1,
                            2, 2, 2]), OrderedFactor)
    ŷ = coerce(categorical([1, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2,
                            1, 1, 1, 2, 2, 1, 2, 1, 2, 2, 2, 1, 2,
                            1, 2, 2]), OrderedFactor)
    # check all constructors
    m = TruePositive()
    @test m(ŷ, y) == truepositive(ŷ, y)
    m = TruePositive(rev=true)
    @test m(ŷ, y) == truenegative(ŷ, y)
    m = TrueNegative()
    @test m(ŷ, y) == truenegative(ŷ, y)
    m = FalsePositive()
    @test m(ŷ, y) == falsepositive(ŷ, y)
    m = FalseNegative()
    @test m(ŷ, y) == falsenegative(ŷ, y)
    m = TruePositiveRate()
    @test m(ŷ, y) == tpr(ŷ, y) == truepositive_rate(ŷ, y)
    m = TrueNegativeRate()
    @test m(ŷ, y) == tnr(ŷ, y) == truenegative_rate(ŷ, y)
    m = FalsePositiveRate()
    @test m(ŷ, y) == fpr(ŷ, y) == falsepositive_rate(ŷ, y)
    m = FalseNegativeRate()
    @test m(ŷ, y) == fnr(ŷ, y) == falsenegative_rate(ŷ, y)
    m = FalseDiscoveryRate()
    @test m(ŷ, y) == fdr(ŷ, y) == falsediscovery_rate(ŷ, y)
    m = Precision()
    @test m(ŷ, y) == precision(ŷ, y)
    m = NPV()
    @test m(ŷ, y) == npv(ŷ, y)
    m = FScore{1}()
    @test m(ŷ, y) == f1score(ŷ, y)
    # check synonyms
    m = TPR()
    @test m(ŷ, y) == tpr(ŷ, y)
    m = TNR()
    @test m(ŷ, y) == tnr(ŷ, y)
    m = FPR()
    @test m(ŷ, y) == fpr(ŷ, y) == fallout(ŷ, y)
    m = FNR()
    @test m(ŷ, y) == fnr(ŷ, y) == miss_rate(ŷ, y)
    m = FDR()
    @test m(ŷ, y) == fdr(ŷ, y)
    m = PPV()
    @test m(ŷ, y) == precision(ŷ, y) == ppv(ŷ, y)
    m = Recall()
    @test m(ŷ, y) == tpr(ŷ, y) == recall(ŷ, y) == sensitivity(ŷ, y) == hit_rate(ŷ, y)
    m = Specificity()
    @test m(ŷ, y) == tnr(ŷ, y) == specificity(ŷ, y) == selectivity(ŷ, y)
    # 'higher order'
    m = BACC()
    @test m(ŷ, y) == bacc(ŷ, y) == (tpr(ŷ, y) + tnr(ŷ, y))/2

    ### External comparisons
    sk_prec = 0.6111111111111112 # m.precision_score(y, yhat, pos_label=2)
    @test precision(ŷ, y) ≈ sk_prec
    sk_rec = 0.6875
    @test recall(ŷ, y) == sk_rec # m.recall_score(y, yhat, pos_label=2)
    sk_f05 = 0.625
    f05 = FScore{0.5}()
    @test f05(ŷ, y) == sk_f05 # m.fbeta_score(y, yhat, 0.5, pos_label=2)

    # reversion mechanism
    sk_prec_rev = 0.5454545454545454
    prec_rev = Precision(rev=true)
    @test prec_rev(ŷ, y) ≈ sk_prec_rev
    sk_rec_rev = 0.46153846153846156
    rec_rev = Recall(rev=true)
    @test rec_rev(ŷ, y) ≈ sk_rec_rev
end

@testset "More multiclass metrics" begin
    y = coerce(categorical([1, 2, 0, 2, 1, 0, 0, 1, 2, 2, 2, 1, 2,
                            2, 1, 0, 1, 1, 1, 2, 1, 2, 2, 1, 2, 1,
                            2, 2, 2]), Multiclass)
    ŷ = coerce(categorical([2, 0, 2, 2, 2, 0, 1, 2, 1, 2, 0, 1, 2,
                            1, 1, 1, 2, 0, 1, 2, 1, 2, 2, 2, 1, 2,
                            1, 2, 2]), Multiclass)
    w = Dict(0=>1, 1=>2, 2=>3) #class_w
    # check all constructors
    m = MulticlassTruePositive()
    @test m(ŷ, y) == multiclass_truepositive(ŷ, y)
    m = MulticlassTrueNegative()
    @test m(ŷ, y) == multiclass_truenegative(ŷ, y)
    m = MulticlassFalsePositive()
    @test m(ŷ, y) == multiclass_falsepositive(ŷ, y)
    m = MulticlassFalseNegative()
    @test m(ŷ, y) == multiclass_falsenegative(ŷ, y)
    m = MulticlassTruePositiveRate()
    @test m(ŷ, y) == multiclass_tpr(ŷ, y) ==
        multiclass_truepositive_rate(ŷ, y)
    @test m(ŷ, y, w) == multiclass_tpr(ŷ, y, w) ==
        multiclass_truepositive_rate(ŷ, y, w)
    m = MulticlassTrueNegativeRate()
    @test m(ŷ, y) == multiclass_tnr(ŷ, y) ==
        multiclass_truenegative_rate(ŷ, y)
    @test m(ŷ, y, w) == multiclass_tnr(ŷ, y, w) ==
        multiclass_truenegative_rate(ŷ, y, w)
    m = MulticlassFalsePositiveRate()
    @test m(ŷ, y) == multiclass_fpr(ŷ, y) ==
        multiclass_falsepositive_rate(ŷ, y)
    @test m(ŷ, y, w) == multiclass_fpr(ŷ, y, w) ==
        multiclass_falsepositive_rate(ŷ, y, w)
    m = MulticlassFalseNegativeRate()
    @test m(ŷ, y) == multiclass_fnr(ŷ, y) ==
        multiclass_falsenegative_rate(ŷ, y)
    @test m(ŷ, y, w) == multiclass_fnr(ŷ, y, w) ==
        multiclass_falsenegative_rate(ŷ, y, w)
    m = MulticlassFalseDiscoveryRate()
    @test m(ŷ, y) == multiclass_fdr(ŷ, y) ==
        multiclass_falsediscovery_rate(ŷ, y)
    @test m(ŷ, y, w) == multiclass_fdr(ŷ, y, w) ==
        multiclass_falsediscovery_rate(ŷ, y, w)
    m = MulticlassPrecision()
    @test m(ŷ, y) == multiclass_precision(ŷ, y)
    @test m(ŷ, y, w) == multiclass_precision(ŷ, y, w)
    m = MulticlassNPV()
    @test m(ŷ, y) == multiclass_npv(ŷ, y)
    @test m(ŷ, y, w) == multiclass_npv(ŷ, y, w)
    m = MulticlassFScore()
    @test m(ŷ, y) == macro_f1score(ŷ, y)
    @test m(ŷ, y, w) == macro_f1score(ŷ, y, w)
    # check synonyms
    m = MTPR(return_type=AbstractVector)
    @test m(ŷ, y) == multiclass_tpr(ŷ, y)
    @test m(ŷ, y, w) == multiclass_tpr(ŷ, y, w)
    m = MTNR(return_type=AbstractVector)
    @test m(ŷ, y) == multiclass_tnr(ŷ, y)
    @test m(ŷ, y, w) == multiclass_tnr(ŷ, y, w)
    m = MFPR()
    @test m(ŷ, y) == multiclass_fpr(ŷ, y) == multiclass_fallout(ŷ, y)
    @test m(ŷ, y, w) == multiclass_fpr(ŷ, y, w) ==
        multiclass_fallout(ŷ, y, w)
    m = MFNR()
    @test m(ŷ, y) == multiclass_fnr(ŷ, y) ==
        multiclass_miss_rate(ŷ, y)
    @test m(ŷ, y, w) == multiclass_fnr(ŷ, y, w) ==
        multiclass_miss_rate(ŷ, y, w)
    m = MFDR()
    @test m(ŷ, y) == multiclass_fdr(ŷ, y)
    @test m(ŷ, y, w) == multiclass_fdr(ŷ, y, w)
    m = MPPV()
    @test m(ŷ, y) == MulticlassPrecision()(ŷ, y) ==
        multiclass_ppv(ŷ, y)
    @test m(ŷ, y, w) == MulticlassPrecision()(ŷ, y, w) ==
        multiclass_ppv(ŷ, y, w)
    m = MulticlassRecall()
    @test m(ŷ, y) == multiclass_tpr(ŷ, y)
    @test m(ŷ, y, w) == multiclass_tpr(ŷ, y, w)
    @test m(ŷ, y) == multiclass_sensitivity(ŷ, y) ==
        multiclass_hit_rate(ŷ, y)
    @test m(ŷ, y, w) == multiclass_sensitivity(ŷ, y, w) ==
        multiclass_hit_rate(ŷ, y, w)
    m = MulticlassSpecificity()
    @test m(ŷ, y) == multiclass_tnr(ŷ, y) == multiclass_specificity(ŷ, y) ==
        multiclass_selectivity(ŷ, y)
    @test m(ŷ, y, w) == multiclass_tnr(ŷ, y, w) ==
        multiclass_specificity(ŷ, y, w) == multiclass_selectivity(ŷ, y, w)
end


@testset "Additional multiclass functions" begin
    table = reshape(collect("aabbbccccddbabccbacccd"), 11, 2)
    table = coerce(table, Multiclass);
    yhat = table[:,1] # ['a', 'a', 'b', 'b', 'b', 'c', 'c', 'c', 'c', 'd', 'd']
    y    = table[:,2] # ['b', 'a', 'b', 'c', 'c', 'b', 'a', 'c', 'c', 'c', 'd']
    class_w = Dict('a'=>7, 'b'=>5, 'c'=>2, 'd'=> 0)

    # class | TP | FP | TP + FP | precision | FN | TP + FN | recall
    # ------|----|----|------------------------------------|-------
    # a     | 1  | 1  | 2       | 1/2       | 1  | 2       | 1/2
    # b     | 1  | 2  | 3       | 1/3       | 2  | 3       | 1/3
    # c     | 2  | 2  | 4       | 1/2       | 3  | 5       | 2/5
    # d     | 1  | 1  | 2       | 1/2       | 0  | 1       | 1

    # helper:
    inverse(x) = 1/x
    harmonic_mean(x...) = inverse(mean(inverse.(x)))

    # precision:
    p_macro = mean([1/2, 1/3, 1/2, 1/2])
    @test MulticlassPrecision()(yhat, y) ≈ p_macro
    p_macro_w = mean([7/2, 5/3, 2/2, 0/2])
    @test MulticlassPrecision()(yhat, y, class_w) ≈ p_macro_w
    @test p_macro_w ≈
        @test_logs((:warn, r"Using macro"),
                     MulticlassPrecision(average=micro_avg)(yhat, y, class_w))
    p_micro = (1 + 1 + 2 + 1)/(2 + 3 + 4 + 2)
    @test MulticlassPrecision(average=micro_avg)(yhat, y) ≈ p_micro

    # recall:
    r_macro = mean([1/2, 1/3, 2/5, 1])
    @test MulticlassRecall(average=macro_avg)(yhat, y) ≈ r_macro
    r_macro_w = mean([7/2, 5/3, 4/5, 0/1])
    @test MulticlassRecall(average=macro_avg)(yhat, y, class_w) ≈ r_macro_w
    @test r_macro_w ≈
        @test_logs((:warn, r"Using macro"),
                     MulticlassRecall(average=micro_avg)(yhat, y, class_w))
    r_micro = (1 + 1 + 2 + 1)/(2 + 3 + 5 + 1)
    @test MulticlassPrecision(average=micro_avg)(yhat, y) ≈ r_micro

    # fscore:
    harm_means = [harmonic_mean(1/2, 1/2),
                     harmonic_mean(1/3, 1/3),
                     harmonic_mean(1/2, 2/5),
                     harmonic_mean(1/2, 1)]
    f1_macro = mean(harm_means)
    @test MulticlassFScore(average=macro_avg)(yhat, y) ≈ f1_macro
    @test MulticlassFScore(average=no_avg, return_type=Vec)(yhat, y, class_w) ≈
        [7, 5, 2, 0] .* harm_means
    f1_macro_w = mean([7, 5, 2, 0] .* harm_means)
    @test MulticlassFScore(average=macro_avg)(yhat, y, class_w) ≈ f1_macro_w
    @test f1_macro_w ≈
        @test_logs((:warn, r"Using macro"),
                     MulticlassFScore(average=micro_avg)(yhat, y, class_w))
    f1_micro = harmonic_mean(p_micro, r_micro)
    @test MulticlassFScore(average=micro_avg)(yhat, y) ≈ f1_micro
end

@testset "ROC" begin
    y = [  0   0   0   1   0   1   1   0] |> vec |> categorical
    s = [0.0 0.1 0.1 0.1 0.2 0.2 0.5 0.5] |> vec
    ŷ = [UnivariateFinite(classes(y[1]), [1.0-p, p]) for p in s]

    fprs, tprs, ts = roc(ŷ, y)

    sk_fprs = [0. , 0.2, 0.4, 0.8, 1. ]
    sk_tprs = [0. , 0.33333333, 0.66666667, 1., 1.]

    @test fprs ≈ sk_fprs
    @test tprs ≈ sk_tprs
end

@testset "docstrings coverage" begin
    @test startswith(info(BrierScore()).docstring, "Brier proper scoring rule")
end
