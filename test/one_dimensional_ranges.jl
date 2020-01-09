module TestOneDimensionalRanges

using Test
using MLJBase

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

@testset "range constructors, scale, iterator" begin
    @test_logs((:warn, r"`values`"),
               @test_throws ErrorException range(dummy_model, :K,
                                                 values=['c', 'd']))
    @test_throws ErrorException range(dummy_model, :K, lower=Inf,
                                      origin=1, unit=1)
    @test_throws ErrorException range(dummy_model, :K, upper=-Inf,
                                      origin=1, unit=1)

    @test_throws ErrorException range(dummy_model, :K, lower=1)
    @test_throws ErrorException range(dummy_model, :K, lower=1, upper=Inf)
    @test_throws ErrorException range(dummy_model, :K, upper=1)
    @test_throws ErrorException range(dummy_model, :K, upper=1, lower=-Inf)

    @test_throws ErrorException range(dummy_model, :K, lower=1, origin=2)
    @test_throws ErrorException range(dummy_model, :K, lower=1, upper=Inf,
                                      origin=2)
    @test_throws ErrorException range(dummy_model, :K, upper=1, origin=2)
    @test_throws ErrorException range(dummy_model, :K, upper=1, lower=-Inf,
                                      origin=2)

    @test_throws ErrorException range(dummy_model, :K,
                                      lower=3, unit=0, origin=4)
    @test_throws ErrorException range(dummy_model, :K,
                                      lower=3, unit=1, origin=2)

    @test_logs((:warn, r"`values` in"),
               range(dummy_model, :K, lower=1, upper=40, values=['c', 'd']))

    @test_throws ErrorException range(dummy_model, :kernel)

    z1 = range(dummy_model, :K, lower=1, upper=10)
    @test z1.origin == 6
    @test z1.unit == 4

    z2 = range(dummy_model, :K, lower=0, origin=2, unit=1)
    @test z2.origin == 2
    @test z2.unit == 1
    @test z2.upper  == Inf

    z3 = range(dummy_model, :K, upper=0, origin=-2, unit=1)
    @test z3.origin == -2
    @test z3.unit == 1
    @test z3.lower  == -Inf

    z4 = range(super_model, :lambda, lower=1, upper=10)
    @test z4.origin == 5.5
    @test z4.unit == 4.5

    p1 = range(dummy_model, :K, lower=1, upper=10, scale=:log10)
    p2 = range(dummy_model, :kernel, values=['c', 'd'])
    p3 = range(super_model, :lambda, lower=0.1, upper=1, scale=:log2)
    p4 = range(dummy_model, :K, lower=1, upper=3, scale=x->2x)
    @test scale(p1) == :log10
    @test scale(p2) == :none
    @test scale(p3) == :log2
    @test scale(p4) == :custom
    @test scale(sin) === sin
    @test transform(MLJBase.Scale, scale(:log), ℯ) == 1
    @test inverse_transform(MLJBase.Scale, scale(:log), 1) == float(ℯ)

    # test that you can replace model with type:
    @test z1 == range(Int, :K, lower=1, upper=10)
    @test z4 == range(Float64, :lambda, lower=1, upper=10)
    @test p2 == range(Char, :kernel, values=['c', 'd'])

    @test iterator(p1, 5)  == [1, 2, 3, 6, 10]
    @test iterator(p2) == collect(p2.values)
    u = 2^(log2(0.1)/2)
    @test iterator(p3, 3) ≈ [0.1, u, 1]
    @test iterator(p4, 3) == [2, 4, 6]
end

@testset "range constructors for nested parameters" begin
    p1 = range(dummy_model, :K, lower=1, upper=10, scale=:log10)
    q1 = range(super_model, :(model1.K) , lower=1, upper=10, scale=:log10)
    @test iterator(q1, 5) == iterator(p1, 5)
    q2 = range
end

end
true
