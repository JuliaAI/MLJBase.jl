module TestUnivariateFiniteArray

using Test
using MLJBase
using CategoricalArrays
import Distributions:pdf, support
import Distributions
using StableRNGs
import Random

rng = StableRNG(111)
n   = 10
c   = 3

@testset "constructing UnivariateFiniteArray objects" begin

    probs  = rand(rng, n)
    supp = ["class1", "class2"]

    @test_throws DomainError UnivariateFinite(supp, probs, pool=missing)
    u = UnivariateFinite(supp, probs, pool=missing, augment=true)
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

    v = categorical(1:3)
    @test_logs((:warn, r"Ignoring"),
               UnivariateFinite(v[1:2], rand(3), augment=true, pool=missing))
    @test_logs((:warn, r"Ignoring"),
               UnivariateFinite(v[1:2], rand(3), augment=true, ordered=true))

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

    # check unseen probablities are a zero *array*:
    v = categorical(1:4)
    probs = rand(3)
    u = UnivariateFinite(v[1:2], probs, augment=true)
    @test pdf.(u, v[3]) == zeros(3)

    # check if we can broadcast over vector of categorical *elements*:
    @test pdf.(u, v[1:2]) == hcat(1 .- probs, probs)
end

@testset "broadcasting mode" begin
    # binary
    rng = StableRNG(668)
    probs = rand(rng, n)
    u = UnivariateFinite(probs, augment = true, pool=missing)
    supp = Distributions.support(u)
    modes = mode.(u)
    @test modes isa CategoricalArray
    expected = [ifelse(p > 0.5, supp[2], supp[1]) for p in probs]
    @test all(modes .== expected)

    # multiclass
    rng = StableRNG(554)
    P   = rand(rng, n, c)
    P ./= sum(P, dims=2)
    u   = UnivariateFinite(P, pool=missing)
    expected = mode.([u...])
    @test all(mode.(u) .== expected)
end

@testset "cat for UnivariateFiniteArray" begin

    # ordered:
    v = categorical([:no, :yes, :maybe, :unseen])
    u1 = UnivariateFinite([v[1], v[2]], rand(5), augment=true)
    u2 = UnivariateFinite([v[3], v[2]], rand(6), augment=true)
    us = (u1, u2)
    u = cat(us..., dims=1)
    @test length(u) == length(u1) + length(u2)
    @test classes(u) == classes(u1)
    supp = Distributions.support(u)
    @test Set(supp) == Set([:no, :yes, :maybe])
    s1 = Distributions.support(u1)
    s2 = Distributions.support(u2)
    @test pdf.(u, s1)[1:length(u1),:] == pdf.(u1, s1)
    @test pdf.(u, s2)[length(u1)+1:length(u1)+length(u2),:] ==
        pdf.(u2, s2)
    @test pdf.(u, v[1])[length(u1)+1:length(u1)+length(u2)] ==
        zeros(length(u2))
    @test pdf.(u, v[3])[1:length(u1)] == zeros(length(u1))
    @test pdf.(u, v[4]) == zeros(length(u))
    
    # unordered:
    v = categorical([:no, :yes, :maybe, :unseen], ordered=true)
    u1 = UnivariateFinite([v[1], v[2]], rand(5), augment=true)
    u2 = UnivariateFinite([v[3], v[2]], rand(6), augment=true)
    us = (u1, u2)
    u = cat(us..., dims=1)
    @test length(u) == length(u1) + length(u2)
    @test classes(u) == classes(u1)
    supp = Distributions.support(u)
    @test Set(supp) == Set([:no, :yes, :maybe])
    s1 = Distributions.support(u1)
    s2 = Distributions.support(u2)
    @test pdf.(u, s1)[1:length(u1),:] == pdf.(u1, s1)
    @test pdf.(u, s2)[length(u1)+1:length(u1)+length(u2),:] ==
        pdf.(u2, s2)
    @test pdf.(u, v[1])[length(u1)+1:length(u1)+length(u2)] ==
        zeros(length(u2))
    @test pdf.(u, v[3])[1:length(u1)] == zeros(length(u1))
    @test pdf.(u, v[4]) == zeros(length(u))

    @test vcat(u1, u2) ≈ cat(u1, u2, dims=1)
    @test hcat(u1, u2) ≈ cat(u1, u2, dims=2)

    # errors
    v1 = categorical(1:2, ordered=true)
    v2 = categorical(v1, ordered=true)
    levels!(v2, levels(v2) |> reverse )
    probs = rand(3)
    u1 = UnivariateFinite(v1, probs, augment=true)
    u2 = UnivariateFinite(v2, probs, augment=true)
    @test_throws DomainError vcat(u1, u2)

    v1 = categorical(1:2)
    v2 = categorical(2:3)
    u1 = UnivariateFinite(v1, probs, augment=true)
    u2 = UnivariateFinite(v2, probs, augment=true)
    @test_throws DomainError vcat(u1, u2)

end

end

true
