module TestUnivariateFiniteArray

using Test
using MLJBase
using CategoricalArrays
import Distributions:pdf, support
import Distributions
using StableRNGs
import Random
rng = StableRNG(123)

rng = StableRNG(111)
n   = 10
c   = 3

@testset "constructing UnivariateFiniteArray objects" begin

    probs  = rand(rng, n)
    support = ["class1", "class2"]

    @test_throws DomainError UnivariateFinite(support, probs, pool=missing)
    u = UnivariateFinite(support, probs, pool=missing, augment=true)
    @test length(u) == n
    @test size(u) == (n,)
    @test pdf.(u, "class2") ≈ probs

    # autosupport:
    @test_throws DomainError UnivariateFinite(probs, pool=missing)
    u = UnivariateFinite(probs, pool=missing, augment=true)
    @test length(u) == n
    @test size(u) == (n,)
    @test pdf.(u, :class_2) ≈ probs
    probs = probs ./ sum(probs)
    u = UnivariateFinite(probs, pool=missing)
    @test u isa UnivariateFinite
    @test pdf(u, :class_1) == probs[1]
    probs = rand(10, 2)
    probs = probs ./ sum(probs, dims=2)
    u = UnivariateFinite(probs, pool=missing)
    @test length(u) == 10
    u.scitype == Multiclass{2}
    pdf.(u, :class_1) == probs[:, 1]
    u = UnivariateFinite(probs, pool=missing, augment=true)
    @test length(u) == 10
    u.scitype == Multiclass{3}
    pdf.(u, :class_2) == probs[:, 1]

    probs = [-1,0,1]
    @test_throws(DomainError,
                 UnivariateFinite(probs, pool=missing, augment=true))
end

@testset "get and set" begin
    # binary
    s = rand(rng, n)
    u = UnivariateFinite([:yes, :no], s, augment=true, pool=missing)

    @test u[1] isa UnivariateFinite
    v = u[3:4]
    @test v isa UnivariateFiniteArray
    @test v[1] ≈ u[3]

    # set:
    u[1] = UnivariateFinite([:yes, :no], [0.1, 0.9], pool=u[1])
    @test pdf(u[1], :yes) == 0.1
    @test pdf(u[1], :no) == 0.9

    # multiclass
    P   = rand(rng, n, c)
    P ./= sum(P, dims=2)
    u   = UnivariateFinite(P, pool=missing)

    @test u[1] isa UnivariateFinite
    v = u[3:4]
    @test v isa UnivariateFiniteArray
    @test v[1] ≈ u[3]

    # set:
    s = Distributions.support(u)
    scalar = UnivariateFinite(s, P[end,:])
    u[1] = scalar
    @test u[1] ≈ u[end]
    @test pdf.(u, s[1])[2:end] == P[2:end,1]
end

@testset "broadcasting pdf" begin
    n = 10
    P = rand(n);
    u = UnivariateFinite([:no, :yes], P, augment=true, pool=missing)
    @test pdf.(u,:yes) == P
    @test pdf.(u, [:yes, :no]) == hcat(P, 1 .- P)

    # check last also works for unwrapped arrays:
    @test pdf.([u...], [:yes, :no]) == hcat(P, 1 .- P)
end

@testset "broadcasting mode" begin
    # binary
    rng = StableRNG(668)
    probs = rand(rng, n)
    u = UnivariateFinite(probs, augment = true, pool=missing)
    support = Distributions.support(u)
    modes = mode.(u)
    @test modes isa CategoricalArray
    expected = [ifelse(p > 0.5, support[2], support[1]) for p in probs]
    @test all(modes .== expected)

    # multiclass
    rng = StableRNG(554)
    P   = rand(rng, n, c)
    P ./= sum(P, dims=2)
    u   = UnivariateFinite(P, pool=missing)
    support = Distributions.support(u)
    expected = support[[findmax(r)[2] for r in eachrow(P)]]
    @test all(mode.(u) .== expected)
end

end

true
