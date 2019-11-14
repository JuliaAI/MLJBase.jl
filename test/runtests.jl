# It is suggested that test code for MLJ.jl include files be placed in
# a file of the same name under "test/" (and included below) and that
# this test code be wrapped in a module. Any new module name will do -
# eg, `module TestDatasets` for code testing `datasets.jl`.

using MLJBase, Test

@testset "scientific trait" begin
    @test include("scientific_trait.jl")
end

@testset "equality" begin
  @test include("equality.jl")
end

@testset "static transforms" begin
  @test include("static.jl")
end

@testset "utilities" begin
  @test include("utilities.jl")
end

@testset "parameters" begin
  @test include("parameters.jl")
end

@testset "distributions" begin
  @test include("distributions.jl")
end

@testset "data" begin
  @test include("data.jl")
end

@testset "info" begin
  @test include("info.jl")
end

@testset "datasets" begin
  @test include("datasets.jl")
end

@testset "tasks" begin
  @test include("tasks.jl")
end

@testset "measures" begin
  @test include("measures.jl")
end

@testset "interface for LossFunctions" begin
  @test include("loss_functions_interface.jl")
end

@testset "@mlj_model" begin
  @test include("mlj_model_macro.jl")
end

@testset "metadatautils" begin
  @test include("metadata_utilities.jl")
end
