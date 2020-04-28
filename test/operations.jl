module TestOperations

using Test
using MLJBase
using ..Models

@testset "errors for deserialized machines" begin
    filename = joinpath(@__DIR__, "machine.jlso")
    m = machine(filename)
    @test_throws ArgumentError predict(m)
end

@testset "error for operations on nodes" begin
    X = source()
    m = machine(OneHotEncoder(), X)
    @test_throws ArgumentError transform(m)
end

end

true
