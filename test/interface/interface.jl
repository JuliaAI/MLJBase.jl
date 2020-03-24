module Interface

using Test, Random, MLJBase,
      Tables, CategoricalArrays,
      DataStructures, DataFrames,
      TypedTables, MLJModelInterface

using ..Models

include("data_utils.jl")

include("model_api.jl")

@test trait(rms) == :measure

end

true
