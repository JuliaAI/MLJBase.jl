module TestOperations

using Test
using MLJBase
using ..Models

@testset "error for operations on nodes" begin
    X = rand(4)
    m = machine(UnivariateStandardizer(), X) |> fit!
    @test_throws ArgumentError inverse_transform(m)
#    @test_deprecated transform(m)
    X = source(rand(4))
    m = machine(UnivariateStandardizer(), X) |> fit!
    @test_throws ArgumentError inverse_transform(m)
#    @test_deprecated transform(m)
end

end

true
