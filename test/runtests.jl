include("preliminaries.jl")

@conditional_testset "composition_models" begin
    @test include("composition/models/methods.jl")
end
