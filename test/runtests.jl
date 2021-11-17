using Distributed
addprocs()

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

import TypedTables
using Tables

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

# enable conditional testing of modules by providing test_args
# e.g. `Pkg.test("MLJBase", test_args=["misc"])`
RUN_ALL_TESTS = isempty(ARGS)
macro conditional_testset(name, expr)
    name = string(name)
    esc(quote 
        if RUN_ALL_TESTS || $name in ARGS
            @testset $name $expr
        end
    end)
end

@conditional_testset "misc" begin
    @test include("utilities.jl")
    @test include("static.jl")
end

@conditional_testset "interface" begin
    @test include("interface/interface.jl")
    @test include("interface/data_utils.jl")
end

@conditional_testset "univariate finite" begin
    @test include("univariate_finite/methods.jl")
    @test include("univariate_finite/arrays.jl")
end

@conditional_testset "measures" begin
    @test include("measures/measures.jl")
    @test include("measures/measure_search.jl")
end

@conditional_testset "resampling" begin
    @test include("resampling.jl")
end

@conditional_testset "data" begin
    @test include("data/data.jl")
    @test include("data/datasets.jl")
    @test include("data/datasets_synthetic.jl")
end

@conditional_testset "sources" begin
    @test include("sources.jl")
end

@conditional_testset "machines" begin
    @test include("machines.jl")
end

@conditional_testset "composition_learning_networks" begin
    @test include("composition/learning_networks/nodes.jl")
    @test include("composition/learning_networks/inspection.jl")
    @test include("composition/learning_networks/machines.jl")
    # VERSION ≥ v"1.3.0-" &&
    #     @test include("composition/learning_networks/arrows.jl")
end

@conditional_testset "composition_models" begin
    @test include("composition/models/methods.jl")
    @test include("composition/models/from_network.jl")
    @test include("composition/models/inspection.jl")
    @test include("composition/models/pipelines.jl")
    @test include("composition/models/pipelines2.jl")
    @test include("composition/models/target_transformed_model.jl")
    @test include("composition/models/stacking.jl")
    @test include("composition/models/_wrapped_function.jl")
    @test include("composition/models/static_transformers.jl")
end

@conditional_testset "operations" begin
    @test include("operations.jl")
end

@conditional_testset "hyperparam" begin
    @test include("hyperparam/one_dimensional_ranges.jl")
    @test include("hyperparam/one_dimensional_range_methods.jl")
end
