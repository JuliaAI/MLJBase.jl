module Interface

using Test, Random, MLJBase,
      Tables, CategoricalArrays,
      OrderedCollections, DataFrames,
      TypedTables, MLJModelInterface, StableRNGs

using ..Models

rng = StableRNG(1551234)

include("data_utils.jl")

include("model_api.jl")

@test trait(rms) == :measure

end

true
