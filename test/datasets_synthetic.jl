module TestDatasetsSynthetic

using Test
using MLJBase
using Random
using Statistics

@testset "make_blobs" begin
    # Standard behaviour
    n, p, centers = 110, 2, 3
    X, y = make_blobs(n, p; centers=centers)
    @test (n, p) == size(X)
    @test n == length(y)
    @test centers == length(unique(y))

    # Specific arguments
    Random.seed!(55151)
    n, p = 500, 3
    centers = randn(4, p)
    stds    = [1.0, 2.0, 3.0, 7.0]
    X, y = make_blobs(n, p; centers=centers, shuffle=false,
                            center_box=-5. => 5.,
                            cluster_std=stds, rng=534)
    @test size(X) == (n, p)
    @test isapprox(std((X[y .== 1, :])), 1.0, rtol=0.1) # roughly 1
    @test isapprox(std((X[y .== 4, :])), 7.0, rtol=0.1) # roughly 7

    # Errors
    @test_throws ArgumentError make_blobs(0, 0)
    @test_throws ArgumentError make_blobs(;center_box=5=>2)
    @test_throws ArgumentError make_blobs(n, p; centers=randn(4, p+1))
    @test_throws ArgumentError make_blobs(n, p; centers=3, cluster_std=[1,1])
    @test_throws ArgumentError make_blobs(n, p; centers=2, cluster_std=[0,1])
end

@testset "make_circles" begin
    n = 55
    X, y = make_circles(n)
    @test (n, 2) == size(X)
    @test n == length(y)
    @test 2 == length(unique(y))

    # specific arguments
    X, y = make_circles(150; shuffle=false, noise=0.01, factor=0.2, rng=55)
    rs = sqrt.(sum(X.^2, dims=2))
    @test all(0.15 .< rs[y.==0] .< 0.25)
    @test all(0.95 .< rs[y.==1] .< 1.05)

    # Errors
    @test_throws ArgumentError make_circles(-1)
    @test_throws ArgumentError make_circles(; noise=-1)
    @test_throws ArgumentError make_circles(; factor=5)
    @test_throws ArgumentError make_circles(; factor=0)
end

@testset "make_moons Tests" begin
    n = 55
    X, y = make_moons(n)
    @test (n, 2) == size(X)
    @test n == length(y)
    @test 2 == length(unique(y))

    # specific arguments
    X, y = make_moons(50; shuffle=false, noise=0.5, xshift=0.3, yshift=0.2,
                           rng=455)
    @test length(unique(y)) == 2

    # Errors
    @test_throws ArgumentError make_moons(-1)
    @test_throws ArgumentError make_moons(noise=-1)
end

end # module
true
