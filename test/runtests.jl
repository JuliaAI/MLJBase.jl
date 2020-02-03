using Test

# adds toy models in the environment + basic tests
include("models/models.jl")

@testset "utilities" begin
    @test include("utilities.jl")
end

@testset "distributions" begin
    include("distributions.jl")
end

@testset "interface" begin
    @test include("interface/interface.jl")
end

# ---------------------------------------------------------------
# XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX
# XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX
# ---------------------------------------------------------------

using Distributed
addprocs(2)

@everywhere begin
    using MLJBase, Test
    using Logging
    using ComputationalResources
end

function include_everywhere(filepath)
    include(filepath) # Load on Node 1 first, triggering any precompile
    if nprocs() > 1
        fullpath = joinpath(@__DIR__, filepath)
        @sync for p in workers()
            @async remotecall_wait(include, p, fullpath)
        end
    end
end

include("test_utilities.jl")

# load Models module containing models implementations for testing:
print("Loading some models for testing...")
include_everywhere("models.jl")
print("\r                                           \r")

@testset "computational resources" begin
    @test include("computational_resources.jl")
end

# @testset "model interface" begin
#     include("models.jl")
#     @test include("model_interface.jl")
# end

@testset "one_dimensional_ranges" begin
  @test include("one_dimensional_ranges.jl")
end

@testset "one_dimensional_range_methods" begin
  @test include("one_dimensional_range_methods.jl")
end

@testset "equality" begin
    @test include("equality.jl")
end

@testset "Static type" begin
    @test include("static.jl")
end

@testset "parameter inspection" begin
    @test include("parameter_inspection.jl")
end

# OK
@testset "distributions" begin
    @test include("distributions.jl")
end

# OK
@testset "info_dict" begin
    @test include("info_dict.jl")
end

# OK
@testset "data" begin
    @test include("data.jl")
end

# OK
@testset "datasets" begin
    @test include("datasets.jl")
    @test include("datasets_synthetic.jl")
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

@testset "resampling" begin
    @test include("resampling.jl")
end
