using Distributed
addprocs()

@info "nprocs() = $(nprocs())"
@static if VERSION >= v"1.3.0-DEV.573"
    import .Threads
    @info "nthreads() = $(Threads.nthreads())"
else
    @info "Running julia $(VERSION). Multithreading tests excluded. "
end

@everywhere begin
    using MLJModelInterface
    using MLJBase
    using Test
    using CategoricalArrays
    using Logging
    using ComputationalResources
    using StableRNGs
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

# load Models module containing model implementations for testing:
print("Loading some models for testing...")
include_everywhere("_models/models.jl")
print("\r                                           \r")

@testset "misc" begin
   @test include("utilities.jl")
   @test include("info_dict.jl")
   @test include("static.jl")
end

@testset "interface" begin
     @test include("interface/interface.jl")
end

@testset "univariate finite" begin
     @test include("univariate_finite/methods.jl")
     @test include("univariate_finite/arrays.jl")
end

@testset "measures" begin
    @test include("measures/measures.jl")
end

@testset "resampling" begin
    @test include("resampling.jl")
end

@testset "data" begin
    @test include("data/data.jl")
    @test include("data/datasets.jl")
    @test include("data/datasets_synthetic.jl")
end

@testset "sources" begin
    @test include("sources.jl")
end

@testset "machines" begin
    @test include("machines.jl")
end

@testset "composition" begin
    @test include("composition/learning_networks/nodes.jl")
    @test include("composition/learning_networks/inspection.jl")
    @test include("composition/learning_networks/machines.jl")
    VERSION ≥ v"1.3.0-" &&
        @test include("composition/learning_networks/arrows.jl")

    @test include("composition/models/methods.jl")
    @test include("composition/models/from_network.jl")
    @test include("composition/models/inspection.jl")
    @test include("composition/models/pipelines.jl")
    @test include("composition/models/_wrapped_function.jl")
    @test include("composition/models/static_transformers.jl")
#    @test include("composition/models/deprecated.jl")
end

@testset "operations.jl" begin
    @test include("operations.jl")
end

@testset "hyperparam" begin
    @test include("hyperparam/one_dimensional_ranges.jl")
    @test include("hyperparam/one_dimensional_range_methods.jl")
end

@testset "openml" begin
    @test include("openml.jl")
end
