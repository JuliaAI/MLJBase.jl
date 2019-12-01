module TestDatasetsSynthetic

using Test

@testset "make_bloobs Tests" begin

    n, p, centers = 110, 2, 3
    X, y = DatasetsSynthetic.make_blobs(n; p=p, centers=centers)
    @test (n, p) == size(X)
    @test n == length(y)
    @test centers == length(unique(y))

    n, p, centers = 90, 3, 4
    X, y = DatasetsSynthetic.make_blobs(n; p=p, centers=centers)
    @test (n, p) == size(X)
    @test n == length(y)
    @test centers == length(unique(y))

end;

@testset "make_circles Tests" begin
    n = 55
    X, y = DatasetsSynthetic.make_circles(n)
    @test (n,2) == size(X)
    @test n == length(y)
    @test 2 == length(unique(y))

end;

@testset "make_moons Tests" begin
    n = 55
    X, y = DatasetsSynthetic.make_moons(n)
    @test (n,2) == size(X)
    @test n == length(y)
    @test 2 == length(unique(y))
end;

end # module
true
