module TestUFV

using Test
using MLJBase
using StableRNGs
using CategoricalArrays

UFV = UnivariateFiniteVector
rng = StableRNG(111)
n   = 10
c   = 3

@testset "Constr:binary" begin
    scores  = rand(rng, n)
    classes = ["class1", "class2"]

    u = UFV(scores, classes)
    @test length(u) == n
    @test size(u) == (n,)
    @test u.scores[1] == scores[1]
    @test u.classes[1] == classes[1]
    @test u.classes[2] == classes[2]

    # autoclasses
    u = UFV(scores)
    @test u.classes[1] == :negative
    @test u.classes[2] == :positive

    # errors
    scores = [-1,0,1]
    @test_throws DomainError UFV(scores)
end

@testset "Constr:multi" begin
    P   = rand(rng, n, c)
    @test_throws DomainError UFV(P)

    P ./= sum(P, dims=2)
    u   = UFV(P)

    @test length(u) == n
    @test length(u.classes) == c
    @test size(u) == (n,)
    @test u.classes[1] == :class_1
    @test u.classes[2] == :class_2
    @test u.classes[3] == :class_3
end

@testset "Get and Set" begin
    # binary
    s = rand(rng, n)
    u = UFV(s)

    @test u[1] isa UnivariateFinite
    @test all(MLJBase.classes(u[1]) .== u.classes)

    u[1] = 0.3
    @test pdf(u[1], u.classes[2]) == 0.3
    u[1:2] = [0.3,0.2]
    @test pdf(u[2], :positive) == 0.2

    @test_throws DomainError (u[1] = -0.2)
    @test_throws DomainError (u[1:2] = [-0.1,0.1])

    # multiclass
    P   = rand(rng, n, c)
    P ./= sum(P, dims=2)
    u   = UFV(P)

    @test u[1] isa UnivariateFinite
    @test all(MLJBase.classes(u[1]) .== u.classes)

    u[1] = [0.3, 0.5, 0.2]
    @test pdf(u[1], u.classes[1]) == 0.3
    @test pdf(u[1], u.classes[2]) == 0.5
    @test pdf(u[1], u.classes[2]) == 0.5
    u[1:2] = [0.3 0.4 0.3; 0.2 0.2 0.6]
    @test pdf(u[1], u.classes[2]) == 0.4
    @test pdf(u[2], u.classes[3]) == 0.6

    @test_throws DomainError (u[1] = [-0.2,0.3,0.9])
    @test_throws DomainError (u[1:2] = rand(2,3))
end

@testset "Mode/Pdf" begin
    # binary
    rng = StableRNG(668)
    s = rand(rng, n)
    u = UFV(s)
    @test pdf(u, :negative) == 1 .- s
    @test pdf.(u, :negative) == 1 .- s
    @test mode(u) isa CategoricalArray
    expected = [ifelse(si > 0.5, u.classes[2], u.classes[1]) for si in s]
    @test all(mode(u) .== expected)
    @test mode.(u) isa CategoricalArray
    @test all(mode.(u) .== expected)

    # multiclass
    rng = StableRNG(554)
    P   = rand(rng, n, c)
    P ./= sum(P, dims=2)
    u   = UFV(P)
    @test pdf(u, :class_1) == P[:,1]
    @test pdf.(u, :class_1) == P[:,1]
    expected = u.classes[[findmax(r)[2] for r in eachrow(P)]]
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
    ufv = UFV(s, classes(yc))
    uv = [UnivariateFinite(classes(yc), [1-si, si]) for si in s]
    auc1 = auc(ufv, yc)
    auc2 = auc(uv, yc)
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
    auc1 = auc(UFV(s, [-1,1]), categorical(y))
    @test abs(skref - auc1) < 1e-5
end

end

true
