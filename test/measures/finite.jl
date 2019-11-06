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

@testset "confusion matrix {2}" begin
    # first class is 1 is assumed positive
    y = categorical([1, 2, 1, 2, 1, 1, 2])
    ŷ = categorical([1, 2, 2, 2, 2, 1, 2])
    cm = confmat(ŷ, y)
    TP = sum(ŷ .== y .== 1) # pred and true = +
    TN = sum(ŷ .== y .== 2) # pred and true = -
    FP = sum(ŷ .!= y .== 2) # pred + and true -
    FN = sum(ŷ .!= y .== 1) # pred - and true +
    @test cm.mat[1,1] == TP
    @test cm.mat[2,2] == TN
    @test cm.mat[1,2] == FP
    @test cm.mat[2,1] == FN

    @test accuracy(ŷ, y) == accuracy(cm) == sum(y .== ŷ) / length(y)

    @test truepositive(ŷ, y) == TP
    @test truepositive(ŷ, y, first_class_positive=false) == TN
    @test truenegative(ŷ, y) == TN
    @test falsepositive(ŷ, y) == FP
    @test falsenegative(ŷ, y) == FN

    @test recall(ŷ, y) == TP / (TP + FN)
    @test precision(ŷ, y) == TP / (TP + FP)
    @test specificity(ŷ, y) == TN / (TN + FP)
    @test f1score(ŷ, y) == 2.0 / (1.0 / recall(ŷ, y) + 1.0 / precision(ŷ, y))

    @test falsediscovery_rate(ŷ, y) == FP / (FP + TP)
    @test truepositive_rate(ŷ, y) == TP / (TP + FN)
    @test truenegative_rate(ŷ, y) == TN / (TN + FP)

    y = categorical(["n", "p", "n", "p", "n", "p"])
    ŷ = [UnivariateFinite(y[1:2], [p, 1.0 - p]) for p in [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]]
    @test auc(ŷ, y) == 1.0

    y = categorical(["n", "p", "n", "p", "n", "p"])
    ŷ = [UnivariateFinite(y[1:2], [p, 1.0 - p]) for p in [0.1, 0.2, 0.3, 0.6, 0.7, 0.8]]
    @test auc(ŷ, y) ≈ 1/3

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html
    y = categorical([1, 1, 2, 2])
    ŷ = [UnivariateFinite(y[1:2], [p, 1.0 - p]) for p in [0.1, 0.4, 0.35, 0.8]]
    @test auc(ŷ, y; first_class_positive=false) == 0.75

    # tested in sklearn and R
    y = categorical([0, 0, 0, 1, 0, 1, 0, 1, 1, 1])
    ŷ = [UnivariateFinite(y[1:2], [p, 1.0-p]) for p in [0.2, 0.1, 0.7, 0.8, 0.8, 0.2, 0.3, 0.5, 0.9, 0.8]]
    @test auc(ŷ, y; first_class_positive=false) == 0.74
end

end
true
