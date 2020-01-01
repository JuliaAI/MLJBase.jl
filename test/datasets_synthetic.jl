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

@testset "make_circles Tests" begin
    n = 55
    X, y = DatasetsSynthetic.make_circles(n)
    @test (n, 2) == size(X)
    @test n == length(y)
    @test 2 == length(unique(y))
end;

@testset "make_moons Tests" begin
    n = 55
    X, y = DatasetsSynthetic.make_moons(n)
    @test (n, 2) == size(X)
    @test n == length(y)
    @test 2 == length(unique(y))
end;

end # module
true
