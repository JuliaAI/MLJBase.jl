module TestOneDimensionalRangeIterators

using Test
using MLJBase
using Random
Random.seed!(123)

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
    rng = MersenneTwister(123)
    @test iterator(p1, 5, rng) == [10, 6, 1, 2, 3]
    @test iterator(rr, rng) == ['b', 'a', 'g', 'c', 'e', 'f', 'd']
    @test iterator(rr, 3, rng) == ['b', 'c', 'a']

end

end
true
