module TestDatasetsSynthetic

using Test
using MLJBase.DatasetsSynthetic

@testset "make_blobs Tests" begin
    
    @testset "make_blobs Tests: n=110, p=2, center=3" begin
    
        n, p, centers = 110, 2, 3
        X, y = DatasetsSynthetic.make_blobs(n; p=p, centers=centers)
        @test (n, p) == size(X)
        @test n == length(y)
        @test centers == length(unique(y))
        end;

    @testset "make_blobs Tests n=90, p=3, center=4" begin
        n, p, centers = 90, 3, 4
        X, y = DatasetsSynthetic.make_blobs(n; p=p, centers=centers)
        @test (n, p) == size(X)
        @test n == length(y)
        @test centers == length(unique(y))
    end;
end;

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
