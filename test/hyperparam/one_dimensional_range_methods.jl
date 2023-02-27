module TestOneDimensionalRangeIterators

using Test
using MLJBase
using Random
import Distributions
using Statistics
using StableRNGs

rng = StableRNG(66600099)
stable_rng() = StableRNG(123)

const Dist = Distributions

mutable struct DummyModel <: Deterministic
    K::Int
    metric::Float64
    kernel::Char
end

dummy_model = DummyModel(4, 9.5, 'k')

mutable struct SuperModel <: Deterministic
    lambda::Float64
    model1::DummyModel
    model2::DummyModel
end

dummy1 = DummyModel(1, 9.5, 'k')
dummy2 = DummyModel(2, 9.5, 'k')
super_model = SuperModel(0.5, dummy1, dummy2)

z1 = range(dummy_model, :K, lower=1, upper=10)
z2 = range(dummy_model, :K, lower=10, origin=10^6, unit=10^5)
z3 = range(dummy_model, :K, upper=-10, origin=-10^6, unit=10^5)
z4 = range(super_model, :lambda, lower=1, upper=10)
z5 = range(dummy_model, :K, origin=10, unit=20)
p1 = range(dummy_model, :K, lower=1, upper=10, scale=:log10)
p2 = range(dummy_model, :kernel, values=['c', 'd'])
p3 = range(super_model, :lambda, lower=0.1, upper=1, scale=:log2)
p4 = range(dummy_model, :K, lower=1, upper=3, scale=x->2x)

[p4, p4]

# Starting from julia v"1.7.0-DEV.1233", the default RNG has changed
# create a function giving julia version dependent default RNG with seed.
@static if VERSION >= v"1.7.0-DEV.1230"
    _default_rng(seed) = (rng = TaskLocalRNG(); Random.seed!(rng, seed))
else
    _default_rng(seed) = MersenneTwister(seed)
end

@testset "scale transformations" begin
    @test transform(MLJBase.Scale, scale(:log), ℯ) == 1
    @test inverse_transform(MLJBase.Scale, scale(:log), 1) == float(ℯ)
end

@testset "inspecting scales of ranges" begin
    @test scale(p1) == :log10
    @test scale(p2) == :none
    @test scale(p3) == :log2
    @test scale(p4) == :custom
    @test scale(sin) === sin
end

@testset "iterators" begin

    @test iterator(p1, 5)  == [1, 2, 3, 6, 10]
    @test iterator(p2) == collect(p2.values)
    u = 2^(log2(0.1)/2)
    @test iterator(p3, 3) ≈ [0.1, u, 1]
    @test iterator(p4, 3) == [2, 4, 6]

    # semi-unbounded ranges:
    v = Int.(round.(exp.([(1-t)*log(10) + t*log(10+2e5)
                     for t in 0:(1/3):1]))) |> unique
    @test iterator(z2, 4) == v
    @test iterator(z3, 4) == reverse(-v)

    # doubly-unbounded ranges:
    @test iterator(z5, 4) ==
        iterator(range(Int, :foo, lower=-10, upper=30), 4)

    # truncated nominal range iteration:
    rr = range(Char, :foo, values=collect("abcdefg"))
    @test iterator(rr, 3) == ['a', 'b', 'c']

    # random:
    rng = StableRNG(66); @test iterator(rng, p1, 5) == [10, 2, 3, 6, 1]
    rng = StableRNG(22); @test iterator(rng, p1, 5) != [10, 2, 3, 6, 1]
    rng = StableRNG(33); @test iterator(rng, rr) == ['b', 'c', 'a', 'g',
                                    'f', 'd', 'e']
    rng = StableRNG(44); @test iterator(rng, rr) != ['b', 'c', 'a', 'g',
                                    'f', 'd', 'e']
    rng = StableRNG(88); @test iterator(rng, rr, 3) == ['b', 'c', 'a']
    rng = StableRNG(99); @test iterator(rng, rr, 3) != ['a', 'c', 'b']

    # with callable as scale:
    r = range(Int, :dummy, lower=1, upper=2, scale=x->10^x)
    expecting = map(x->round(Int,10^x), range(1, stop= 2, length=10))
    @test iterator(r, 10) == expecting
end

@testset "fitting distributions to NumericRange objects" begin

    # characterizations

    l = rand(rng)
    u = max(l, rand(rng)) + 1
    r = range(Int, :dummy, lower=l, upper=u)
    for D in [:Arcsine, :Uniform, :Biweight, :Cosine, :Epanechnikov,
              :SymTriangularDist, :Triweight]
        eval(quote
             d = Dist.fit(Dist.$D, $r)
             @test minimum(d) ≈ $l
             @test maximum(d) ≈ $u
             end
             )
    end

    o = randn(rng)
    s = rand(rng)
    r = range(Int, :dummy, lower=-Inf, upper=Inf, origin=o, unit=s)
    for D in [:Cauchy, :Gumbel, :Normal, :Laplace]
        eval(quote
             d = Dist.fit(Dist.$D, $r)
             @test Dist.location(d) ≈ $o
             @test Dist.scale(d) ≈ $s
             end
             )
    end

    o = rand(rng)
    s = o/(1 + rand(rng))
    r = range(Int, :dummy, lower=-Inf, upper=Inf, origin=o, unit=s)
    for D in [:Normal, :Gamma, :InverseGaussian, :LogNormal, :Logistic]
        eval(quote
             d = Dist.fit(Dist.$D, $r)
             @test mean(d) ≈ $o
             @test std(d) ≈ $s
             end
             )
    end

    r = range(Float64, :dummy, lower=-Inf, upper=Inf, unit=s, origin=o,)
    d = Dist.fit(Dist.Poisson, r)
    @test mean(d) ≈ s

    # truncation

    r = range(Int, :dummy, lower=l, upper=u)
    d = Dist.fit(Dist.Normal, r)
    @test minimum(d) == l
    @test maximum(d) == u

    # unsupported distributions

    @test_throws ArgumentError Dist.fit(Dist.Beta, r)

end

@testset "NumericSampler - distribution instance specified"  begin

    @testset  "integers" begin
        r = range(Int, :dummy, lower=11, upper=13)
        d = Dist.Uniform(1, 20)

        s = MLJBase.sampler(r, d)

        rng = StableRNG(0)
        dict = Dist.countmap(rand(rng, s, 1000))
        eleven, twelve, thirteen = map(x -> dict[x], 11:13)
        @test eleven == 252 && twelve == 514 && thirteen == 234

        rng = StableRNG(999);
        dict = Dist.countmap(rand(rng, s, 1000))
        eleven, twelve, thirteen = map(x -> dict[x], 11:13)
        @test eleven == 236 && twelve == 494 && thirteen == 270
    end

    @testset "right-unbounded floats" begin
        r = range(Float64, :dummy, lower=0.2, upper = Inf,
                  origin=5, unit=1) # origin and unit not relevant here
        s = MLJBase.sampler(r, Dist.Normal())

        rng = stable_rng()
        v = rand(rng, s, 1000)
        @test all(x >= 0.2 for x in v)
        @test abs(minimum(v)/0.2 - 1) <= 0.02

        rng = stable_rng()
        @test rand(rng, s, 1000) == v

        q = quantile(v, 0.0:0.1:1.0)
        v2 = filter(x -> x>=0.2, rand(stable_rng(), Dist.Normal(), 3000))[1:1000]
        q2 = quantile(v2, 0.0:0.1:1.0)
        @test all(x -> x≈1.0, q ./ q2)
    end

    @testset "sampler using callable scale" begin
        r = range(Int, :dummy, lower=1, upper=2, scale=x->10^x)
        s = sampler(r, Dist.Uniform)
        Random.seed!(123)
        v = rand(s, 10000)
        @test issubset(v, 10:100)
        rng = _default_rng(123)
        @test rand(rng, s, 10000) == v

        r = range(Float64, :dummy, lower=1, upper=2, scale=x->10^x)
        s = sampler(r, Dist.Uniform)
        Random.seed!(1)
        v = rand(s, 10000)
        @test abs(minimum(v) - 10) < 0.02
        @test abs(maximum(v) - 100) < 0.02
        rng = _default_rng(1)
        @test rand(rng, s, 10000) == v

    end

end

@testset "NumericSampler - distribution type specified"  begin

    r = range(Int, :k, lower=2, upper=6, origin=4.5, unit=1.2)
    s = MLJBase.sampler(r, Dist.Normal)
    v1 = rand(MersenneTwister(1), s, 50)
    d = Dist.truncated(Dist.Normal(r.origin, r.unit), r.lower, r.upper)
    v2 = map(x -> round(Int, x), rand(MersenneTwister(1), d, 50))
    @test v1 == v2

end

@testset "NominalSampler" begin

    r = range(Char, :(model.dummy), values=collect("cab"))

    @testset "probability vector specified" begin
        s = MLJBase.sampler(r, [0.1, 0.2, 0.7])
        rng = StableRNG(600)
        dict = Dist.countmap(rand(rng,s, 1000))
        c, a, b = map(x -> dict[x], collect("cab"))
        @test a == 201 && b == 714 && c == 85

        rng = StableRNG(89);
        dict = Dist.countmap(rand(rng, s, 1000))
        c, a, b = map(x -> dict[x], collect("cab"))
        @test a == 173 && b == 733 && c == 94
    end

    @testset "probability vector unspecified (uniform)" begin
        s = MLJBase.sampler(r)
        rng = StableRNG(55)
        dict = Dist.countmap(rand(rng,s, 1000))
        c, a, b = map(x -> dict[x], collect("cab"))
        @test a == 361 && b == 335 && c == 304

        rng = StableRNG(550)
        dict = Dist.countmap(rand(rng, s, 1000))
        c, a, b = map(x -> dict[x], collect("cab"))
        @test a == 332 && b == 356 && c == 312
    end

end

struct MySampler end
Base.rand(rng::AbstractRNG, ::MySampler) = rand(rng)


@testset "scale(s) for s a sampler" begin
    @test scale(MySampler()) == :none
    r = range(Char, :(model.dummy), values=collect("cab"))
    @test scale(MLJBase.sampler(r, [0.1, 0.2, 0.7])) == :none
    r1 = range(Int, :dummy, lower=1, upper=2, scale=x->10^x)
    @test scale(MLJBase.sampler(r1, Dist.Uniform)) == :custom
    r2 = range(Int, :k, lower=2, upper=6, origin=4.5, unit=1.2, scale=:log2)
    @test scale(MLJBase.sampler(r2, Dist.Normal)) == :log2
end

end

true
