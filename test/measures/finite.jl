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
    l = levels(y) # f, m, n
    cm = confmat(ŷ, y; warn=false)
    e(l,i,j) = sum((ŷ .== l[i]) .& (y .== l[j]))
    for i in 1:3, j in 1:3
        @test cm[i,j] == e(l,i,j)
    end
    perm = [3, 1, 2]
    l2 = l[perm]
    cm2 = confmat(ŷ, y; perm=perm) # no warning because permutation is given
    for i in 1:3, j in 1:3
        @test cm2[i,j] == e(l2,i,j)
    end
    @test_logs (:warn, "The classes are un-ordered,\nusing order: ['f', 'm', 'n'].\nTo suppress this warning, consider coercing to OrderedFactor.") confmat(ŷ, y)
    ŷc = coerce(ŷ, OrderedFactor)
    yc = coerce(y, OrderedFactor)
    @test confmat(ŷc, yc).mat == cm.mat

    y = categorical(['a','b','a','b'])
    ŷ = categorical(['b','b','a','a'])
    @test_logs (:warn, "The classes are un-ordered,\nusing: negative='a' and positive='b'.\nTo suppress this warning, consider coercing to OrderedFactor.") confmat(ŷ, y)
end

@testset "mcr-accuracy" begin
    y = categorical(['m', 'f', 'n', 'f', 'm', 'n', 'n', 'm', 'f'])
    ŷ = categorical(['f', 'f', 'm', 'f', 'n', 'm', 'n', 'm', 'f'])
    @test accuracy(ŷ, y) == 1-mcr(ŷ,y) ==
            accuracy(confmat(ŷ, y, warn=false))  == 1-mcr(confmat(ŷ, y, warn=false))
    w = randn(length(y))
    @test accuracy(ŷ, y, w) == 1-mcr(ŷ,y,w)
end

@testset "AUC" begin
    # this is random binary and random scores generated with numpy
    # then using roc_auc_score from sklearn to get the AUC
    # we check that we recover a comparable AUC and that it's invariant
    # to ordering.
    c = ["pos", "neg"]
    y = categorical(c[[0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
                     1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
                     1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0,
                     1, 0] .+ 1])
    ŷ = [UnivariateFinite(y[1:2], [p, 1.0 - p]) for p in [
        0.90237535, 0.41276349, 0.94511611, 0.08390761, 0.55847392,
        0.26043136, 0.78565351, 0.20133953, 0.7404382 , 0.15307601,
        0.59596716, 0.8169512 , 0.88200483, 0.23321489, 0.94050483,
        0.27593662, 0.60702176, 0.36427036, 0.35481784, 0.06416543,
        0.45576954, 0.12354048, 0.79830435, 0.15799818, 0.20981099,
        0.43451663, 0.24020098, 0.11401055, 0.25785748, 0.86490263,
        0.75715379, 0.06550534, 0.12628999, 0.18878245, 0.1283757 ,
        0.76542903, 0.8780248 , 0.86891113, 0.24835709, 0.06528076,
        0.72061354, 0.89451634, 0.95634394, 0.07555979, 0.16345437,
        0.43498831, 0.37774708, 0.31608861, 0.41369339, 0.95691113]]
    @test isapprox(auc(ŷ, y), 0.455716, rtol=1e-4)
    # flip the ordering and the scores, the AUC is invariant
    y2 = categorical(c[[1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1,
                      0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                      0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1,
                      0, 1] .+ 1])
    ŷ2 = [UnivariateFinite(y[1:2], [p, 1.0 - p]) for p in [
        0.09762465, 0.58723651, 0.05488389, 0.91609239, 0.44152608,
        0.73956864, 0.21434649, 0.79866047, 0.2595618 , 0.84692399,
        0.40403284, 0.1830488 , 0.11799517, 0.76678511, 0.05949517,
        0.72406338, 0.39297824, 0.63572964, 0.64518216, 0.93583457,
        0.54423046, 0.87645952, 0.20169565, 0.84200182, 0.79018901,
        0.56548337, 0.75979902, 0.88598945, 0.74214252, 0.13509737,
        0.24284621, 0.93449466, 0.87371001, 0.81121755, 0.8716243 ,
        0.23457097, 0.1219752 , 0.13108887, 0.75164291, 0.93471924,
        0.27938646, 0.10548366, 0.04365606, 0.92444021, 0.83654563,
        0.56501169, 0.62225292, 0.68391139, 0.58630661, 0.04308887]]
    @test auc(ŷ, y) ≈ auc(ŷ2, y2)
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
