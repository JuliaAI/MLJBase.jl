rng = StableRNG(51803)
using LinearAlgebra

const Vec = AbstractVector

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

    # The auc algorithm should be able to handle the case where two or more
    # samples in the prediction vector has the same UnivariateFinite distribution
    # We check this by comparing our auc with that gotten from roc_auc_score from sklearn.
    y = categorical(["class_1","class_1","class_0","class_0","class_1","class_1","class_0"])
    ŷ = UnivariateFinite(levels(y), [0.8,0.7,0.5,0.5,0.5,0.5,0.3], augment=true, pool=y)
    # We can see that ŷ[3] ≈ ŷ[4] ≈ ŷ[5] ≈ ŷ[6]
    @test isapprox(auc(ŷ, y), 0.8333333333333334, rtol=1e-16)
end

@testset "Log, Brier, Spherical - finite case" begin
    y = categorical(collect("abb"))
    L = [y[1], y[2]]
    d1 = UnivariateFinite(L, [0.1, 0.9]) # a
    d2 = UnivariateFinite(L, Float32[0.4, 0.6]) # b
    d3 = UnivariateFinite(L, [0.2, 0.8]) # b
    yhat = [d1, d2, d3]
    ym = vcat(y, [missing,])
    yhatm = vcat(yhat, [d3, ])

    @test mean(log_loss(yhat, y)) ≈
        Float32(-(log(0.1) + log(0.6) + log(0.8))/3)
    @test mean(skipmissing(log_loss(yhatm, ym))) ≈
        Float32(-(log(0.1) + log(0.6) + log(0.8))/3)
    yhat = UnivariateFinite(L, [0.1 0.9;
                                0.4 0.6;
                                0.2 0.8])
    @test isapprox(mean(log_loss(yhat, y)),
                   -(log(0.1) + log(0.6) + log(0.8))/3, atol=eps(Float32))

    @test log_score(yhat, y) ≈ -log_loss(yhat, y)

    # sklearn test
    # >>> from sklearn.metrics import log_loss
    # >>> log_loss(["spam", "ham", "ham", "spam","ham","ham"],
    #    [[.1, .9], [.9, .1], [.8, .2], [.35, .65], [0.2, 0.8], [0.3,0.7]])
    # 0.6130097025803921
    y2 = categorical(["spam", "ham", "ham", "spam", "ham", "ham"])
    L2 = classes(y2[1])
    probs = vcat([.1 .9], [.9 .1], [.8 .2], [.35 .65], [0.2 0.8], [0.3 0.7])
    yhat2 = UnivariateFinite(L2, probs)
    y2m = vcat(y2, [missing,])
    yhat2m = UnivariateFinite(L2, vcat(probs, [0.1 0.9]))
    @test mean(log_loss(yhat2, y2)) ≈ 0.6130097025803921
    @test mean(skipmissing(log_loss(yhat2, y2))) ≈ 0.6130097025803921

    ## Brier
    scores = BrierScore()(yhat, y)
    @test size(scores) == size(y)
    @test Float32.(scores) ≈ [-1.62, -0.32, -0.08]
    scoresm = BrierScore()(yhatm, ym)
    @test Float32.((scoresm)[1:3]) ≈ [-1.62, -0.32, -0.08]
    @test ismissing(scoresm[end])
    # test specialized broadcasting on brierloss
    @test BrierLoss()(yhat, y) == -BrierScore()(yhat, y) 
    # sklearn test
    # >>> from sklearn.metrics import brier_score_loss
    # >>> brier_score_loss([1, 0, 0, 1, 0, 0], [.9, .1, .2, .65, 0.8, 0.7])
    # 0.21875 NOTE: opposite orientation
    @test -mean(BrierScore()(yhat2, y2)) / 2 ≈ 0.21875
    probs2 = [[.1, .9], [Float32(0.9), Float32(1) - Float32(0.9)], [.8, .2],
              [.35, .65], [0.2, 0.8], [0.3, 0.7]]
    yhat3 = [UnivariateFinite(L2, prob) for prob in probs2]
    @test -mean(BrierScore()(yhat3, y2) / 2) ≈ 0.21875
    @test mean(BrierLoss()(yhat3, y2) / 2) ≈ -mean(BrierScore()(yhat3, y2) / 2)

    # Spherical
    s = SphericalScore() # SphericalScore(2)
    norms = [norm(probs[i,:]) for i in 1:size(probs, 1)]
    @test (pdf.(yhat2, y2) ./ norms) ≈  s(yhat2, y2)
    # non-performant version:
    yhat4 = [yhat2...]
    @test (pdf.(yhat2, y2) ./ norms) ≈  s(yhat4, y2)
end

@testset "LogScore, BrierScore, SphericalScore - infinite case" begin
    uniform = Distributions.Uniform(2, 5)
    betaprime = Distributions.BetaPrime()
    discrete_uniform = Distributions.DiscreteUniform(2, 5)
    w = [2, 3]

    # brier
    yhat = [missing, uniform]
    @test isapprox(brier_score(yhat, [1.0, 1.0]) |> last, -1/3)
    @test isapprox(brier_score(yhat, [NaN, 4.0]) |> last,  1/3)
    @test isapprox(brier_score(yhat, [1.0, 1.0], w) |> last, -1)
    yhat = [missing, uniform]
    # issue https://github.com/JuliaStats/Distributions.jl/issues/1392
    @test_broken isapprox(brier_score(yhat, [missing, 4.0], w), [1,])
    yhat = [discrete_uniform, discrete_uniform]
    @test isapprox(brier_score(yhat, [NaN, 1.0]), [-1/4, -1/4,])
    @test isapprox(brier_score(yhat, [4.0, 4.0]), [1/4, 1/4,])

    # spherical
    yhat = [uniform, uniform]
    @test isapprox(spherical_score(yhat, [1.0, 1.0]), [0, 0])
    @test isapprox(spherical_score(yhat, [NaN, 4.0]), [0, 1/sqrt(3),])
    # issue https://github.com/JuliaStats/Distributions.jl/issues/1392
    @test_broken isapprox(spherical_score(yhat, [missing, 4.0], w), [sqrt(3),])
    @test isapprox(spherical_score(yhat, [4.0, 4.0], w), [2/sqrt(3), sqrt(3),])
    yhat = [discrete_uniform, discrete_uniform]
    @test isapprox(spherical_score(yhat, [NaN, 1.0]), [0, 0])
    @test isapprox(spherical_score(yhat, [4.0, 4.0]), [1/2, 1/2])

    # log
    yhat = [uniform, uniform]
    @test isapprox(log_score(yhat, [4.0, 4.0]), [-log(3), -log(3),])
    @test isapprox(log_score(yhat, [4.0, 4.0], w), [-2*log(27)/3, -log(27)])
    yhat = [discrete_uniform, discrete_uniform]
    # issue https://github.com/JuliaStats/Distributions.jl/issues/1392
    @test_broken  isapprox(log_score(yhat, [missing, 4.0]), [-log(4),])

    log_score([missing, uniform], [4.0, 4.0])

    # errors
    @test_throws(MLJBase.err_l2_norm(brier_score),
                 brier_score([betaprime, betaprime], [1.0, 1.0]))
    s = SphericalScore(alpha=1)
    @test_throws MLJBase.ERR_UNSUPPORTED_ALPHA s(yhat, [1.0, 1.0])
end

true
