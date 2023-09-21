using MLJBase
if !MLJBase.TESTING
    error(
        "To test MLJBase, the environment variable "*
        "`TEST_MLJBASE` must be set to `\"true\"`\n"*
        "You can do this in the REPL with `ENV[\"TEST_MLJBASE\"]=\"true\"`"
    )
end

using Distributed
# Thanks to https://stackoverflow.com/a/70895939/5056635 for the exeflags tip.
addprocs(; exeflags="--project=$(Base.active_project())")

@info "nprocs() = $(nprocs())"
import .Threads
@info "nthreads() = $(Threads.nthreads())"

@everywhere begin
    using MLJModelInterface
    using MLJBase
    using Test
    using CategoricalArrays
    using Logging
    using ComputationalResources
    using StableRNGs
    using StatisticalMeasures
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

# To avoid printing `@conditional_testset (macro with 1 method)`
# when loading this file via `include("test/preliminaries.jl")`.
nothing
