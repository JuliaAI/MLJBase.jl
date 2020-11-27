module TestUnivariateFiniteArray

using Test
using MLJBase
using CategoricalArrays
import Distributions:pdf, logpdf, support
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
    @test pdf.(u, "class_2") ≈ probs
    probs = probs ./ sum(probs)
    u = UnivariateFinite(probs, pool=missing)
    @test u isa UnivariateFinite
    @test pdf(u, "class_1") == probs[1]
    probs = rand(10, 2)
    probs = probs ./ sum(probs, dims=2)
    u = UnivariateFinite(probs, pool=missing)
    @test length(u) == 10
    u.scitype == Multiclass{2}
    pdf.(u, "class_1") == probs[:, 1]
    u = UnivariateFinite(probs, pool=missing, augment=true)
    @test length(u) == 10
    u.scitype == Multiclass{3}
    pdf.(u, "class_2") == probs[:, 1]

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
    u = UnivariateFinite(["yes", "no"], s, augment=true, pool=missing)

    @test u[1] isa UnivariateFinite
    v = u[3:4]
    @test v isa UnivariateFiniteArray
    @test v[1] ≈ u[3]

    # set:
    u[1] = UnivariateFinite(["yes", "no"], [0.1, 0.9], pool=u[1])
    @test pdf(u[1], "yes") == 0.1
    @test pdf(u[1], "no") == 0.9

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

n = 10
P = rand(n);
all_classes = categorical(["no", "yes"], ordered=true)
u = UnivariateFinite(all_classes, P, augment=true) #uni_fin_arr

# next is not specific to `UnivariateFiniteArray` but is for any
# abstract array with eltype `UnivariateFinite`:
@testset "piratical pdf and logpdf" begin
    # test pdf(uni_fin_arr, labels) and
    # logpdf(uni_fin_arr, labels)
    @test pdf(u, ["yes", "no"]) == hcat(P, 1 .- P)
    @test isequal(logpdf(u, ["yes", "no"]), log.(hcat(P, 1 .- P)))
    @test pdf(u, reverse(all_classes)) == hcat(P, 1 .- P)
    @test isequal(logpdf(u, reverse(all_classes)), log.(hcat(P, 1 .- P)))
    
    # test pdf(::Array{UnivariateFinite, 1}, labels) and
    # logpdf(::Array{UnivariateFinite, labels)
    @test pdf([u...], ["yes", "no"]) == hcat(P, 1 .- P)
    @test isequal(logpdf([u...], ["yes", "no"]), log.(hcat(P, 1 .- P)))
    @test pdf([u...], all_classes) == hcat(1 .- P, P)
    @test isequal(logpdf([u...], all_classes), log.(hcat(1 .- P, P)))
end

@testset "broadcasting: pdf.(uni_fin_arr, scalar) and logpdf.(uni_fin_arr, scalar) " begin
    @test pdf.(u,"yes") == P
    @test isequal(logpdf.(u,"yes"), log.(P))
    @test pdf.(u,all_classes[2]) == P
    @test isequal(logpdf.(u,all_classes[2]), log.(P))

    # check unseen probablities are a zero *array*:
    v = categorical(1:4)
    probs = rand(3)
    u2 = UnivariateFinite(v[1:2], probs, augment=true)
    @test pdf.(u2, v[3]) == zeros(3)
    @test isequal(logpdf.(u2, v[3]), log.(zeros(3)))
end

@testset "broadcasting: pdf.(uni_fin_arr, array_same_shape) and logpdf.(uni_fin_arr, array_same_shape)" begin
    v = rand(classes(u), n)
    @test broadcast(pdf, u, v) == [pdf(u[i], v[i]) for i in 1:length(u)]
    @test isequal(broadcast(logpdf, u, v), 
        [logpdf(u[i], v[i]) for i in 1:length(u)])
    @test broadcast(pdf, u, get.(v)) ==
        [pdf(u[i], v[i]) for i in 1:length(u)]
    @test isequal(broadcast(logpdf, u, get.(v)), 
        [logpdf(u[i], v[i]) for i in 1:length(u)])
end

@testset "broadcasting: check indexing in `getter((cv, i), dtype)` see PR#375" begin
    c  = categorical([0,1,1])
    d = UnivariateFinite(c[1:1], [1 1 1]')
    v = categorical([0,1,1,1])
    @test broadcast(pdf, d, v[2:end]) == [0,0,0]
end

@testset "_getindex" begin
   @test MLJBase._getindex(collect(1:4), 2, Int64) == 2
   @test MLJBase._getindex(nothing, 2, Int64) == zero(Int64)
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
    v = categorical(["no", "yes", "maybe", "unseen"])
    u1 = UnivariateFinite([v[1], v[2]], rand(5), augment=true)
    u2 = UnivariateFinite([v[3], v[2]], rand(6), augment=true)
    us = (u1, u2)
    u = cat(us..., dims=1)
    @test length(u) == length(u1) + length(u2)
    @test classes(u) == classes(u1)
    supp = Distributions.support(u)
    @test Set(supp) == Set(["no", "yes", "maybe"])
    s1 = Distributions.support(u1)
    s2 = Distributions.support(u2)
    @test pdf(u, s1)[1:length(u1),:] == pdf(u1, s1)
    @test pdf(u, s2)[length(u1)+1:length(u1)+length(u2),:] ==
        pdf(u2, s2)
    @test pdf.(u, v[1])[length(u1)+1:length(u1)+length(u2)] ==
        zeros(length(u2))
    @test pdf.(u, v[3])[1:length(u1)] == zeros(length(u1))
    @test pdf.(u, v[4]) == zeros(length(u))

    # unordered:
    v = categorical(["no", "yes", "maybe", "unseen"], ordered=true)
    u1 = UnivariateFinite([v[1], v[2]], rand(5), augment=true)
    u2 = UnivariateFinite([v[3], v[2]], rand(6), augment=true)
    us = (u1, u2)
    u = cat(us..., dims=1)
    @test length(u) == length(u1) + length(u2)
    @test classes(u) == classes(u1)
    supp = Distributions.support(u)
    @test Set(supp) == Set(["no", "yes", "maybe"])
    s1 = Distributions.support(u1)
    s2 = Distributions.support(u2)
    @test pdf(u, s1)[1:length(u1),:] == pdf(u1, s1)
    @test pdf(u, s2)[length(u1)+1:length(u1)+length(u2),:] ==
        pdf(u2, s2)
    @test pdf.(u, v[1])[length(u1)+1:length(u1)+length(u2)] ==
        zeros(length(u2))
    @test pdf.(u, v[3])[1:length(u1)] == zeros(length(u1))
    @test pdf.(u, v[4]) == zeros(length(u))

    @test pdf([vcat(u1, u2)...], supp) ≈
        pdf(vcat([u1...], [u2...]), supp)
    h = hcat(u1, u1)
    h_nowrap = hcat([u1...], [u1...])
    @test size(h) == size(h_nowrap)
    # TODO: why does identity.(h) not work?
    @test h[3,2] ≈ h_nowrap[3,2]

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
