module TestUV

using Test
using MLJBase
using StableRNGs
using CategoricalArrays

UV = UnivariateFiniteVector
rng = StableRNG(111)
n   = 10
c   = 3

@testset "Mode/Pdf" begin
    # binary
    rng = StableRNG(668)
    s = rand(rng, n)
    u = UV(s)
    @test pdf(u, :negative) == 1 .- s
    @test pdf.(u, :negative) == 1 .- s
    @test mode(u) isa CategoricalArray
    expected = [ifelse(si > 0.5, u.support[2], u.support[1]) for si in s]
    @test all(mode(u) .== expected)
    @test mode.(u) isa CategoricalArray
    @test all(mode.(u) .== expected)

    # multiclass
    rng = StableRNG(554)
    P   = rand(rng, n, c)
    P ./= sum(P, dims=2)
    u   = UV(P)
    @test pdf(u, :class_1) == P[:,1]
    @test pdf.(u, :class_1) == P[:,1]
    expected = u.support[[findmax(r)[2] for r in eachrow(P)]]
    @test all(mode(u) .== expected)
    @test all(mode.(u) .== expected)
end

@testset "Metrics" begin
    # AUC
    s = [0.54, 0.25, 0.01, 0.16, 0.41, 0.  , 0.31, 0.8 , 0.32, 0.42, 0.48,
         0.2 , 0.02, 0.03, 0.09, 0.07, 0.07, 0.36, 0.35, 0.06]
    y = [1, -1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, 1]
    yc = categorical(y)
    skref = 0.6197916 # sklearn.metrics.roc_auc_score(y, a) == 0.619
    U = UV(MLJBase.classes(yc), s)
    u = [UnivariateFinite(MLJBase.classes(yc), [1-si, si]) for si in s]
    auc1 = auc(U, yc)
    auc2 = auc(u, yc)
    @test auc1 == auc2
    @test abs(skref - auc1) < 1e-2
    s = [0.37, 0.2 , 0.32, 0.6 , 0.27, 0.39, 0.48, 0.32, 0.73, 0.36, 0.54,
       0.09, 0.5 , 0.48, 0.4 , 0.17, 0.36, 0.62, 0.08, 0.48, 0.55, 0.3 ,
       0.79, 0.5 , 0.18, 0.25, 0.51, 0.25, 0.6 , 0.1 , 0.16, 0.59, 0.3 ,
       0.44, 0.27, 0.34, 0.54, 0.62, 0.26, 0.35, 0.22, 0.3 , 0.31, 0.48,
       0.23, 0.17, 0.23, 0.23, 0.43, 0.73]
    y = [-1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
        1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
        1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
        1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]
    skref = 0.448979
    auc1 = auc(UV([-1,1], s), categorical(y))
    @test abs(skref - auc1) < 1e-5
end

end

true
