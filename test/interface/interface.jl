module Interface

using Test, Random, MLJBase,
      Tables, CategoricalArrays,
      OrderedCollections, DataFrames,
      TypedTables

using ..Models

include("data_utils.jl")

include("model_api.jl")
include("univariate_finite.jl")

end
