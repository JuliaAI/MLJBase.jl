module TestDistributions

using Test
using MLJBase
using CategoricalArrays
import Distributions:pdf, support
import Distributions
using StableRNGs
import Random
rng = StableRNG(123)

v = categorical(collect("asqfasqffqsaaaa"), ordered=true)
V = categorical(collect("asqfasqffqsaaaa"))
a, s, q, f = v[1], v[2], v[3], v[4]
A, S, Q, F = V[1], V[2], V[3], V[4]

@testset "classes" begin
    @test classes(v[1]) == levels(v)
    @test classes(v) == levels(v)
    levels!(v, reverse(levels(v)))
    @test classes(v[1]) == levels(v)
    @test classes(v) == levels(v)

    # no longer true:
    # @test classes(a) == levels(v)
end

@testset "UnivariateFinite constructor" begin

    # ordered (OrderedFactor)
    dict = Dict(s=>0.1, q=> 0.2, f=> 0.7)
    d    = UnivariateFinite(dict)
    @test classes(d) == [a, f, q, s]
    @test classes(d) == classes(s)
    @test levels(d) == levels(s)
    @test support(d) == [f, q, s]
    @test MLJBase.sample_scitype(d) == OrderedFactor{4}
    # levels!(v, reverse(levels(v)))
    # @test classes(d) == [s, q, f, a]
    # @test support(d) == [s, q, f]

    @test pdf(d, s) ≈ 0.1
    @test pdf(d, 's') ≈ 0.1
    @test pdf(d, q) ≈ 0.2
    @test pdf(d, 'q') ≈ 0.2
    @test pdf(d, f) ≈ 0.7
    @test pdf(d, 'f') ≈ 0.7
    @test pdf(d, a) ≈ 0.0
    @test mode(d) == f

    @test UnivariateFinite(support(d), [0.7, 0.2, 0.1]) ≈ d

    N = 50
    rng = StableRNG(125)
    samples = [rand(rng,d) for i in 1:50];
    rng = StableRNG(125)
    @test samples == [rand(rng, d) for i in 1:N]

    N = 10000
    samples = rand(rng, d, N);
    @test Set(samples) == Set(support(d))
    freq = Distributions.countmap(samples)
    @test isapprox(freq[f]/N, 0.7, atol=0.05)
    @test isapprox(freq[s]/N, 0.1, atol=0.05)
    @test isapprox(freq[q]/N, 0.2, atol=0.05)

    #
    # unordered (Multiclass):
    dict = Dict(S=>0.1, Q=> 0.2, F=> 0.7)
    d    = UnivariateFinite(dict)
    @test classes(d) == [a, f, q, s]
    @test classes(d) == classes(s)
    @test levels(d) == levels(s)
    @test support(d) == [f, q, s]
    @test MLJBase.sample_scitype(d) == Multiclass{4}
    # levels!(v, reverse(levels(v)))
    # @test classes(d) == [s, q, f, a]
    # @test support(d) == [s, q, f]

    @test pdf(d, S) ≈ 0.1
    @test pdf(d, 's') ≈ 0.1
    @test pdf(d, Q) ≈ 0.2
    @test pdf(d, 'q') ≈ 0.2
    @test pdf(d, F) ≈ 0.7
    @test pdf(d, 'f') ≈ 0.7
    @test pdf(d, A) ≈ 0.0
    @test mode(d) == F

    @test UnivariateFinite(support(d), [0.7, 0.2, 0.1]) ≈ d

    N = 50
    rng = StableRNG(661)
    samples = [rand(rng,d) for i in 1:50];
    rng = StableRNG(661)
    @test samples == [rand(rng, d) for i in 1:N]

    N = 10000
    samples = rand(rng, d, N);
    @test Set(samples) == Set(support(d))
    freq = Distributions.countmap(samples)
    @test isapprox(freq[F]/N, 0.7, atol=0.05)
    @test isapprox(freq[S]/N, 0.1, atol=0.05)
    @test isapprox(freq[Q]/N, 0.2, atol=0.05)

    # no support specified:
    @test_logs (:warn, r"No ") UnivariateFinite([0.7, 0.3])
    d = UnivariateFinite([0.7, 0.3], pool=missing)
    @test pdf(d, :class_1) == 0.7
end

@testset "constructor arguments not categorical values" begin
    @test_logs((:warn, r"No "),
               UnivariateFinite(Dict('f'=>0.7, 'q'=>0.2, 's'=>0.1)))
    @test_throws(MethodError,
                 UnivariateFinite(Dict('f'=>"junk", 'q'=>0.2, 's'=>0.1),
                                  pool=missing))
    d = UnivariateFinite(Dict('f'=>0.7, 'q'=>0.2, 's'=>0.1), pool=missing)
    @test pdf(d, 'f') ≈ 0.7
    @test pdf(d, 's') ≈ 0.1
    @test pdf(d, 'q') ≈ 0.2
    @test_logs((:warn, r"No "),
               UnivariateFinite(['f', 'q', 's'],  [0.7, 0.2, 0.1]))
    @test_throws(MethodError,
                 UnivariateFinite(['f', 'q', 's'],  ["junk", 0.2, 0.1],
                                  pool=missing))
    d = UnivariateFinite(['f', 'q', 's'],  [0.7, 0.2, 0.1], pool=missing)
    @test pdf(d, 'f') ≈ 0.7
    @test pdf(d, 's') ≈ 0.1
    @test pdf(d, 'q') ≈ 0.2

    d = UnivariateFinite(Dict('f'=>0.7, 'q'=>0.2, 's'=>0.1), pool=f)
    @test pdf(d, 'f') ≈ 0.7
    @test pdf(d, 's') ≈ 0.1
    @test pdf(d, 'q') ≈ 0.2

    d = UnivariateFinite(['f', 'q', 's'],  [0.7, 0.2, 0.1], pool=a)
    @test pdf(d, 'f') ≈ 0.7
    @test pdf(d, 's') ≈ 0.1
    @test pdf(d, 'q') ≈ 0.2

    d = UnivariateFinite(Dict('f'=>0.7, 'q'=>0.2, 's'=>0.1), pool=v)
    @test pdf(d, 'f') ≈ 0.7
    @test pdf(d, 's') ≈ 0.1
    @test pdf(d, 'q') ≈ 0.2

    d = UnivariateFinite(['f', 'q', 's'],  [0.7, 0.2, 0.1], pool=v)
    @test pdf(d, 'f') ≈ 0.7
    @test pdf(d, 's') ≈ 0.1
    @test pdf(d, 'q') ≈ 0.2
end

rng = StableRNG(111)
n   = 10
c   = 3

@testset "constructing of UnivariateFinite arrays" begin

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

@testset "broadcasting over UnivariateFiniteArrays" begin
    n = 10_000
    P = rand(n);
    u = UnivariateFinite([:no, :yes], P, augment=true, pool=missing)
    @test pdf.(u,:yes) == P
end

@testset "Univariate mode" begin
    v = categorical(1:101)
    p = rand(rng,101)
    s = sum(p) - p[42]
    p[42] = 0.5
    p = p/2/s
    p[42] = 0.5
    d = UnivariateFinite(v, p)
    @test mode(d) == 42
end

@testset "UnivariateFinite methods" begin
    y = categorical(["yes", "no", "yes", "yes", "maybe"])
    yes = y[1]
    no = y[2]
    maybe = y[end]
    prob_given_class = Dict(yes=>0.7, no=>0.3)
    d = UnivariateFinite(prob_given_class)
    @test pdf(d, yes) ≈ 0.7
    @test pdf(d, no) ≈ 0.3
    @test pdf(d, maybe) ≈ 0

    v = categorical(collect("abcd"))
    d = UnivariateFinite(v, [0.2, 0.3, 0.1, 0.4])
    sample = rand(rng,d, 10^4)
    freq_given_class = Distributions.countmap(sample)
    pairs = collect(freq_given_class)
    sort!(pairs, by=pair->pair[2], alg=QuickSort)
    sorted_classes = first.(pairs)
    @test sorted_classes == ['c', 'a', 'b', 'd']

    junk = categorical(['j',])
    j = junk[1]
    v = categorical(['a', 'b', 'a', 'b', 'c', 'b', 'a', 'a', 'f'])
    a = v[1]
    f = v[end]
    # remove f from sample:
    v = v[1 : end - 1]
    d = Distributions.fit(UnivariateFinite, v)
    @test pdf(d, a) ≈ 0.5
    @test pdf(d, 'a') ≈ 0.5
    @test pdf(d, 'b') ≈ 0.375
    @test pdf(d, 'c') ≈ 0.125
    @test pdf(d, 'f') == 0
    @test pdf(d, f) == 0
    @test_throws DomainError pdf(d, 'j')

    d2 = Distributions.fit(UnivariateFinite, v, nothing)
    @test d2 ≈ d

    # with weights:
    w = [2, 3, 2, 3, 5, 3, 2, 2]
    d = Distributions.fit(UnivariateFinite, v, w)
    @test pdf(d, a) ≈ 8/22
    @test pdf(d, 'a') ≈ 8/22
    @test pdf(d, 'b') ≈ 9/22
    @test pdf(d, 'c') ≈ 5/22
    @test pdf(d, 'f') == 0
    @test pdf(d, f) == 0
    @test_throws DomainError pdf(d, 'j')
end

@testset "approx for UnivariateFinite" begin
    y = categorical(["yes", "no", "maybe"])
    yes   = y[1]
    no    = y[2]
    maybe = y[3]
    @test(UnivariateFinite([yes, no, maybe], [0.1, 0.2, 0.7]) ≈
          UnivariateFinite([maybe, yes, no], [0.7, 0.1, 0.2]))
    @test(!(UnivariateFinite([yes, no, maybe], [0.1, 0.2, 0.7]) ≈
          UnivariateFinite([maybe, yes, no], [0.7, 0.2, 0.1])))
end

@testset "UnivariateFinite arithmetic" begin
    v = categorical(collect("abc"))
    a , b, c = v[1], v[2], v[3]

    d1  = UnivariateFinite([a, b], [0.2, 0.8])
    d2  = UnivariateFinite([b, c], [0.3, 0.7])
    dvec = [d1, d2]
    d = average(dvec)
    @test pdf(d, 'a') ≈ 0.1
    @test pdf(d, 'b') ≈ 0.55
    @test pdf(d, 'c') ≈ 0.35
    w = [4, 6]
    d = average(dvec, weights=w)
    @test pdf(d, 'a') ≈ 0.4*0.2
    @test pdf(d, 'b') ≈ 0.4*0.8 + 0.6*0.3
    @test pdf(d, 'c') ≈ 0.6*0.7
end

end # module

true
