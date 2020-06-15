module Utilities

using Test
using MLJBase
using StableRNGs
import Random
using ComputationalResources

@test MLJBase.finaltypes(Union{Missing,Int}) == [Union{Missing,Int64}]
@test MLJBase.finaltypes(Float64) == [Float64]

abstract type Foo end
struct Bar <: Foo end
struct Baz <: Foo end

@test MLJBase.finaltypes(Foo) == [Bar, Baz]

@testset "flat_values" begin
    t = (X = (x = 1, y = 2), Y = 3)
    @test flat_values(t) == (1, 2, 3)
end

mutable struct M
    a1
    a2
end
mutable struct A1
    a11
    a12
end
mutable struct A2
    a21
end
mutable struct A21
    a211
    a212
end

@testset "recursive getproperty, setproperty!" begin
    m = (a1 = (a11 = 10, a12 = 20), a2 = (a21 = (a211 = 30, a212 = 40),))

    @test MLJBase.recursive_getproperty(m, :(a1.a12)) == 20
    @test MLJBase.recursive_getproperty(m, :a1) == (a11 = 10, a12 = 20)
    @test MLJBase.recursive_getproperty(m, :(a2.a21.a212)) == 40

    m = M(A1(10, 20), A2(A21(30, 40)))
    MLJBase.recursive_setproperty!(m, :(a2.a21.a212), 42)
    @test MLJBase.recursive_getproperty(m, :(a1.a11)) == 10
    @test MLJBase.recursive_getproperty(m, :(a1.a12)) == 20
    @test MLJBase.recursive_getproperty(m, :(a2.a21.a211)) == 30
    @test MLJBase.recursive_getproperty(m, :(a2.a21.a212)) == 42
    @test MLJBase.recursive_getproperty(
        MLJBase.recursive_getproperty(m, :(a2.a21)), :a212) == 42
end

@testset "shuffle rows" begin
    rng = StableRNG(5996661)
    # check dims
    x = randn(rng, 5)
    y = randn(rng, 5, 5)
    z = randn(rng, 5, 5)
    @test MLJBase.check_dimensions(x, y) === nothing
    @test MLJBase.check_dimensions(z, x) === nothing
    @test MLJBase.check_dimensions(y, z) === nothing
    @test_throws DimensionMismatch MLJBase.check_dimensions(x, randn(4))

    x = 1:5 |> collect
    y = 1:5 |> collect
    rng = 555
    Random.seed!(rng)
    perm = Random.randperm(5)
    @test MLJBase.shuffle_rows(x, y; rng=rng) == (x[perm], y[perm])
    y = randn(5,5)
    @test MLJBase.shuffle_rows(x, y; rng=rng) == (x[perm], y[perm,:])
    @test MLJBase.shuffle_rows(z, y; rng=rng) == (z[perm,:], y[perm,:])
    @test MLJBase.shuffle_rows(x, x; rng=rng) == (x[perm], x[perm])
end

@testset "unwind" begin
    iterators = ([1, 2], ["a","b"], ["x", "y", "z"])
    @test unwind(iterators...) ==
        [1  "a"  "x";
         2  "a"  "x";
         1  "b"  "x";
         2  "b"  "x";
         1  "a"  "y";
         2  "a"  "y";
         1  "b"  "y";
         2  "b"  "y";
         1  "a"  "z";
         2  "a"  "z";
         1  "b"  "z";
         2  "b"  "z"]
end

@testset "comp resources" begin
    @test default_resource() == CPU1()
    default_resource(ComputationalResources.CPUProcesses())
    @test default_resource() == ComputationalResources.CPUProcesses()
    default_resource(CPU1())
end

@static if VERSION >= v"1.3.0-DEV.573"
    @testset "Chunks" begin
        nthreads = Threads.nthreads()
        #test for cases with exactly same work as threads
        @test length(MLJBase.chunks(1:nthreads, nthreads)) == nthreads
        #test for cases with more work than threads
        @test length(MLJBase.chunks(1:nthreads + rand(1:nthreads), nthreads)) == nthreads
        #test for cases with less work than threads
        @test length(MLJBase.chunks(1:nthreads-1, nthreads)) == nthreads - 1
    end
end

end # module
true
