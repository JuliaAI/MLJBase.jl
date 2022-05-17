# To speed up the development workflow, use `TestEnv`.
# For example:
# ```
# $ julia --project
#
# julia> ENV["TEST_MLJBASE"] = "true"
#
# julia> using TestEnv; TestEnv.activate()
#
# julia> include("test/preliminaries.jl")
# [...]
#
# julia> include("test/resampling.jl")
# [...]
# ```
include("preliminaries.jl")

@conditional_testset "misc" begin
    @test include("utilities.jl")
    @test include("static.jl")
end

@conditional_testset "interface" begin
    @test include("interface/interface.jl")
    @test include("interface/data_utils.jl")
end

@conditional_testset "measures" begin
    @test include("measures/measures.jl")
    @test include("measures/measure_search.jl")
    @test include("measures/doc_strings.jl")
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
end

@conditional_testset "composition_models" begin
    @test include("composition/models/methods.jl")
    @test include("composition/models/from_network.jl")
    @test include("composition/models/inspection.jl")
    @test include("composition/models/deprecated.jl")
    @test include("composition/models/pipelines.jl")
    @test include("composition/models/transformed_target_model.jl")
    @test include("composition/models/stacking.jl")
    @test include("composition/models/static_transformers.jl")
end

@conditional_testset "operations" begin
    @test include("operations.jl")
end

@conditional_testset "hyperparam" begin
    @test include("hyperparam/one_dimensional_ranges.jl")
    @test include("hyperparam/one_dimensional_range_methods.jl")
end
