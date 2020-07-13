module TestOperations

using Test
using MLJBase
using ..Models

@testset "errors for deserialized machines" begin
    filename = joinpath(@__DIR__, "machine.jlso")
    m = machine(filename)
#    @test_deprecated @test_throws ArgumentError predict(m)
     @test_throws ArgumentError predict(m)
end

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
