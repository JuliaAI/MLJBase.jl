using Random
using MLJBase
using Test

mutable struct Foo <: MLJType
    rng::AbstractRNG
    x::Int
    y::Int
end

mutable struct Bar <: MLJType
    rng::AbstractRNG
    x::Int
    y::Int
end

f1 = Foo(MersenneTwister(7), 1, 2)
f2 = Foo(MersenneTwister(8), 1, 2)
@test f1.rng != f2.rng
@test f1 == f2
f1.x = 10
@test f1 != f2
b = Bar(MersenneTwister(7), 1, 2)
@test f2 != b

@test is_same_except(f1, f2, :x)
f1.y = 20
@test f1 != f2
@test is_same_except(f1, f2, :x, :y)

# test for nested fields

mutable struct Super <: MLJType
    sub::Foo
    z::Int
end

f1 = Foo(MersenneTwister(7), 1, 2)
f2 = Foo(MersenneTwister(8), 1, 2)
s1 = Super(f1, 20)
s2 = Super(f2, 20)
@test s1 == s2
s2.sub.x = 10
@test f1 != f2


true
