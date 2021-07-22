using Distributed
addprocs()

ENV["MLJBASE"] = true

using MLJBase
if !MLJBase.TESTING
    error("To test MLJBase, the environment variable "*
          "`TEST_MLJBASE` must be set to `\"true\"`\n"*
          "You can do this in the REPL with `ENV[\"TEST_MLJBASE\"]=\"true\"")
end

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

@testset "resampling" begin
    @test include("resampling.jl")
end
