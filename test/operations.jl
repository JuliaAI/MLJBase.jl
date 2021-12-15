module TestOperations

using Test
using MLJBase
using ..Models

@testset "error for operations on nodes" begin
    X = rand(4)
    m = fit!(machine(UnivariateStandardizer(), X), verbosity=0)
    @test_throws ArgumentError inverse_transform(m)
    X = source(rand(4))
    m = fit!(machine(UnivariateStandardizer(), X), verbosity=0)
    @test_throws ArgumentError inverse_transform(m)
end

end

true
