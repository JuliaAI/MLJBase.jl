using Random
using MLJBase
using Test

mutable struct Foo <: MLJType
    rng::AbstractRNG
    x::Int
end

f1 = Foo(MersenneTwister(7), 1)
f2 = Foo(MersenneTwister(8), 1)
@test f1.rng != f2.rng
@test f1 == f2
f1.x = 2
@test f1 != f2

true
