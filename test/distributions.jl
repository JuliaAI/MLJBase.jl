module TestDistributions

using Test
using MLJBase
using CategoricalArrays
import Distributions:pdf, support
import Distributions
import Random.seed!
seed!(1234)

v = categorical(collect("asqfasqffqsaaaa"), ordered=true)
V = categorical(collect("asqfasqffqsaaaa"))
a, s, q, f = v[1], v[2], v[3], v[4]
A, S, Q, F = V[1], V[2], V[3], V[4]

@testset "classes" begin
    p = a.pool
    @test classes(p) == [a, f, q, s]
    levels!(v, reverse(levels(v)))
    @test classes(p) == reverse([a, f, q, s])
    levels!(v, reverse(levels(v)))

    p = A.pool
    @test classes(p) == [A, F, Q, S]
    levels!(V, reverse(levels(V)))
    @test classes(p) == reverse([A, F, Q, S])
    levels!(V, reverse(levels(V)))
end

@testset "UnivariateFinite constructor" begin
    dict = Dict(s=>0.1, q=> 0.2, f=> 0.7)
    d    = UnivariateFinite(dict)
    @test classes(d) == [a, f, q, s]
    @test support(d) == [f, q, s]
    # levels!(v, reverse(levels(v)))
    # @test classes(d) == [s, q, f, a]
    # @test support(d) == [s, q, f]

    @test pdf(d, s) ≈ 0.1
    @test pdf(d, 's') ≈ 0.1
    @test mode(d) == f
    @test Set(unique(rand(d, 100))) == Set(support(d))

    @test UnivariateFinite(support(d), [0.7, 0.2, 0.1]) ≈ d

    @test_throws ArgumentError UnivariateFinite(['f', 'q', 's'],  [0.7, 0.2, 0.1])
    @test_throws ArgumentError UnivariateFinite(Dict('f'=>0.7, 'q'=>0.2, 's'=>0.1))

    dict = Dict(S=>0.1, Q=> 0.2, F=> 0.7)
    d    = UnivariateFinite(dict)
    @test classes(d) == [A, F, Q, S]
    @test support(d) == [F, Q, S]
    # levels!(v, reverse(levels(v)))
    # @test classes(d) == [s, q, f, a]
    # @test support(d) == [s, q, f]

    @test pdf(d, S) ≈ 0.1
    @test pdf(d, 's') ≈ 0.1
    @test mode(d) == F
    @test Set(unique(rand(d, 100))) == Set(support(d))

    @test UnivariateFinite(support(d), [0.7, 0.2, 0.1]) ≈ d

    @test_throws ArgumentError UnivariateFinite(['F', 'Q', 'S'],  [0.7, 0.2, 0.1])
    @test_throws ArgumentError UnivariateFinite(Dict('F'=>0.7, 'Q'=>0.2, 'S'=>0.1))
end

@testset "Univariate mode" begin
    v = categorical(1:101)
    p = rand(101)
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
    sample = rand(d, 10^4)
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
    @test_throws ArgumentError pdf(d, 'j')

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
    @test_throws ArgumentError pdf(d, 'j')
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
