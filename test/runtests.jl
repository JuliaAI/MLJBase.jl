using MLJBase, Test

@testset "model interface" begin
    @test include("MLJBase.jl")
end

@testset "scientific trait" begin
    @test include("scientific_trait.jl")
end

@testset "equality" begin
  @test include("equality.jl")
end

@testset "Static type" begin
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

@testset "info_dict" begin
  @test include("info_dict.jl")
end

@testset "datasets" begin
  @test include("datasets.jl")
end

@testset "tasks" begin
  @test include("tasks.jl")
end

@testset "measures" begin
  @test include("measures/measures.jl")
  @test include("measures/continuous.jl")
  @test include("measures/finite.jl")
  @test include("measures/loss_functions_interface.jl")
end

@testset "@mlj_model" begin
  @test include("mlj_model_macro.jl")
end

@testset "metadatautils" begin
  @test include("metadata_utilities.jl")
end

@testset "pipeline_static.jl" begin
  @test include("pipeline_static.jl")
end

# load Models module containing models for further testing:
print("Loading some models for testing...")
include("models.jl")
print("\r                                           \r")

@testset "machines" begin
   @test include("machines.jl")
end

@testset "networks" begin
   @test include("networks.jl")
end

@testset "composites" begin
   @test include("composites.jl")
end

@testset "pipelines" begin
   @test include("pipelines.jl")
end

VERSION ≥ v"1.3.0-" && @testset "arrows" begin
    @test include("arrows.jl")
end




