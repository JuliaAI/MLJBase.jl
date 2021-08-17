module Interface

using Test, Random, MLJBase,
    Tables, CategoricalArrays,
    OrderedCollections,
    TypedTables, MLJModelInterface,
    StableRNGs

using ..Models

rng = StableRNG(1551234)

include("data_utils.jl")

include("model_api.jl")

end

true
