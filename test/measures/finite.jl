module TestFiniteMeasures

using Test
using MLJBase
import Distributions
using CategoricalArrays
import Random.seed!
seed!(51803)

@testset "built-in classifier measures" begin
    y    = categorical(collect("asdfasdfaaassdd"))
    yhat = categorical(collect("asdfaadfaasssdf"))
    w = 1:15
    @test misclassification_rate(yhat, y) ≈ 0.2
    @test misclassification_rate(yhat, y, w) ≈ 4/15
    y = categorical(collect("abb"))
    L = [y[1], y[2]]
    d1 = UnivariateFinite(L, [0.1, 0.9])
    d2 = UnivariateFinite(L, [0.4, 0.6])
    d3 = UnivariateFinite(L, [0.2, 0.8])
    yhat = [d1, d2, d3]
    @test mean(cross_entropy(yhat, y)) ≈ -(log(0.1) + log(0.6) + log(0.8))/3
    scores = BrierScore()(yhat, y)
    @test scores ≈ [-1.62, -0.32, -0.08]
    wscores = BrierScore()(yhat, y, [1, 2, 7])
    @test wscores ≈ scores .* [0.3, 0.6, 2.1]
end

@testset "confusion matrix" begin
    y = categorical(['m', 'f', 'n', 'f', 'm', 'n', 'n', 'm', 'f'])
    ŷ = categorical(['f', 'f', 'm', 'f', 'n', 'm', 'n', 'm', 'f'])
    l = levels(y)
    cm = confmat(ŷ, y)
    e(i,j) = sum((ŷ .== l[i]) .& (y .== l[j]))
    for i in 1:3, j in 1:3
        @test cm[i,j] == e(i,j)
    end
end

@testset "confusion matrix {2}" begin
    # first class is 1 is assumed negative, second positive
    y = categorical([1, 2, 1, 2, 1, 1, 2])
    ŷ = categorical([1, 2, 2, 2, 2, 1, 2])
    cm = confmat(ŷ, y)
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

    @test truepositive(ŷ, y)  == TP
    @test truenegative(ŷ, y)  == TN
    @test falsepositive(ŷ, y) == FP
    @test falsenegative(ŷ, y) == FN

    @test truepositive(ŷ, y; rev=true) == TN

    @test recall(ŷ, y)      == TP / (TP + FN)
    @test precision(ŷ, y)   == TP / (TP + FP)
    @test specificity(ŷ, y) == TN / (TN + FP)
    @test f1score(ŷ, y)     == 2.0 / (1.0 / recall(ŷ, y) + 1.0 / precision(ŷ, y))

    recall_rev = Recall(rev=true)
    @test recall_rev(ŷ, y) == TN / (TN + FP)
    precision_rev = Precision(rev=true)
    @test precision_rev(ŷ, y) == TN / (TN + FN)
    specificity_rev = Specificity(rev=true)
    @test specificity_rev(ŷ, y) == TP / (TP + FN)
    f1score_rev = FScore{1}(rev=true)
    @test f1score_rev(ŷ, y) == 2.0 / (1.0 / recall_rev(ŷ, y) + 1.0 / precision_rev(ŷ, y))

    @test falsediscovery_rate(ŷ, y) == FP / (FP + TP)
    @test truepositive_rate(ŷ, y)   == TP / (TP + FN)
    @test truenegative_rate(ŷ, y)   == TN / (TN + FP)

    y = categorical(["n", "p", "n", "p", "n", "p"])
    ŷ = [UnivariateFinite(y[1:2], [p, 1.0 - p]) for p in [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]]
    @test auc(ŷ, y) == 0.0
    @test positive_label(y) == "p"
    @test positive_label(y; rev=true) == "n"
    auc_rev = AUC(rev=true)
    @test auc_rev(ŷ, y) == 1.0

    y = categorical(["n", "p", "n", "p", "n", "p"])
    @test positive_label(y) == "p"
    ŷ = [UnivariateFinite(y[1:2], [p, 1.0 - p]) for p in [0.1, 0.2, 0.3, 0.6, 0.7, 0.8]]
    @test auc(ŷ, y) ≈ 2/3

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html
    y = categorical([1, 1, 2, 2])
    ŷ = [UnivariateFinite(y[1:2], [p, 1.0 - p]) for p in [0.1, 0.4, 0.35, 0.8]]
    @test auc(ŷ, y) == 0.75

    # tested in sklearn and R
    y = categorical([0, 0, 0, 1, 0, 1, 0, 1, 1, 1])
    ŷ = [UnivariateFinite(y[1:2], [p, 1.0-p]) for p in [0.2, 0.1, 0.7, 0.8, 0.8, 0.2, 0.3, 0.5, 0.9, 0.8]]
    @test auc(ŷ, y) == 0.74
end

@testset "Metadata binary" begin
    for m in (accuracy, recall, Precision(), f1score, specificity)
        e = info(m)
        m == accuracy    && (@test e.name == "accuracy")
        m == recall      && (@test e.name == "recall")
        m isa Precision  && (@test e.name == "precision")
        m == f1score     && (@test e.name == "F1-score")
        m == specificity && (@test e.name == "specificity")
        @test e.target_scitype == AbstractVector{<:Finite}
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
    @test e.name == "auc"
    @test e.target_scitype == AbstractVector{<:Finite}
    @test e.prediction_type == :probabilistic
    @test e.reports_each_observation == false
    @test e.is_feature_dependent == false
    @test e.supports_weights == false
end

end
true
