module TestDatasetsSynthetic

using Test
using MLJBase
using Random
using Statistics
using CategoricalArrays

@testset "make_blobs" begin
    # Standard behaviour
    n, p, centers = 110, 2, 3
    X, y = make_blobs(n, p; centers=centers)
    @test (n, p) == size(MLJBase.matrix(X))
    @test n == length(y)
    @test centers == length(unique(y))
    @test y isa CategoricalVector

    # Specific arguments
    Random.seed!(55151)
    n, p = 500, 3
    centers = randn(4, p)
    stds    = [1.0, 2.0, 3.0, 7.0]
    X, y = make_blobs(n, p; centers=centers, shuffle=false,
                            center_box=-5. => 5.,
                            cluster_std=stds, rng=534,
                            as_table=false, eltype=Float32)
    @test size(X) == (n, p)
    @test eltype(X) == Float32
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
    @test (n, 2) == size(MLJBase.matrix(X))
    @test n == length(y)
    @test 2 == length(unique(y))
    @test y isa CategoricalVector

    # specific arguments
    X, y = make_circles(150; shuffle=false, noise=0.01, factor=0.2,
                             rng=55, as_table=false, eltype=Float32)
    @test eltype(X) == Float32
    rs = sqrt.(sum(X.^2, dims=2))
    @test all(0.15 .< rs[y.==0] .< 0.25)
    @test all(0.95 .< rs[y.==1] .< 1.05)

    # Errors
    @test_throws ArgumentError make_circles(-1)
    @test_throws ArgumentError make_circles(; noise=-1)
    @test_throws ArgumentError make_circles(; factor=5)
    @test_throws ArgumentError make_circles(; factor=0)
end

@testset "make_moons" begin
    n = 55
    X, y = make_moons(n)
    @test (n, 2) == size(MLJBase.matrix(X))
    @test n == length(y)
    @test 2 == length(unique(y))

    # specific arguments
    X, y = make_moons(50; shuffle=false, noise=0.5, xshift=0.3, yshift=0.2,
                           rng=455, as_table=false, eltype=Float32)
    @test length(unique(y)) == 2
    @test eltype(X) == Float32

    # Errors
    @test_throws ArgumentError make_moons(-1)
    @test_throws ArgumentError make_moons(noise=-1)
end

@testset "make_regression" begin
    n, p = 100, 5
    X, y = make_regression(n, p)
    Xm = MLJBase.matrix(X)
    @test size(Xm) == (n, p)
    @test length(y) == n

    # specific arguments end
    X, y = make_regression(150, 3; intercept=false, sparse=0.3, noise=0.5,
                                   outliers=0.5, as_table=false,
                                   eltype=Float32, rng=155)
    @test eltype(X) == Float32
    @test size(X) == (150, 3)
    @test length(y) == 150

    # binary
    X, y = make_regression(150, 3; binary=true)
    @test y isa CategoricalVector

    # errors
    @test_throws ArgumentError make_regression(-5, 2)
    @test_throws ArgumentError make_regression(2, -2)
    @test_throws ArgumentError make_regression(noise=-1)
    @test_throws ArgumentError make_regression(sparse=-1)
    @test_throws ArgumentError make_regression(outliers=-1)
end

end # module
true

# --------------------------------------------------------------------------
# The script below will eventually go in the tutorials, it generates points
# using standard keywords and plots them. It was also used to eyball the
# results and make sure they made sense
# --------------------------------------------------------------------------

# using PyPlot
#
# # BLOBS
#
# figure(figsize=(8, 6))
# X, y = make_blobs(100, 2, centers=3, as_table=false)
#
# plot(X[y.==1, 1], X[y.==1, 2], ls="none", marker="o")
# plot(X[y.==2, 1], X[y.==2, 2], ls="none", marker="o")
# plot(X[y.==3, 1], X[y.==3, 2], ls="none", marker="o")
#
# gcf()
#
# # CIRCLES
#
# figure(figsize=(8, 6))
# X, y = make_circles(200, as_table=false, factor=0.3)
#
# plot(X[y.==0, 1], X[y.==0, 2], ls="none", marker="o")
# plot(X[y.==1, 1], X[y.==1, 2], ls="none", marker="o")
#
# gcf()
#
# # MOONS
#
# figure(figsize=(8, 6))
# X, y = make_moons(200, as_table=false)
#
# plot(X[y.==0, 1], X[y.==0, 2], ls="none", marker="o")
# plot(X[y.==1, 1], X[y.==1, 2], ls="none", marker="o")
#
# gcf()
#
# # REGRESSION 1
#
# figure(figsize=(8, 6))
# X, y = make_regression(200, 1, as_table=false, noise=0.5, rng=550)
#
# plot(X[:, 1], y, ls="none", marker="o")
#
# gcf()
#
# # REGRESSION WITH OUTLIERS
#
# figure(figsize=(8, 6))
# X, y = make_regression(200, 1, as_table=false, noise=0.5, rng=550, outliers=0.1)
#
# plot(X[:, 1], y, ls="none", marker="o")
#
# gcf()
