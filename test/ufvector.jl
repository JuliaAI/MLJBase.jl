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

end

true
