# these two shortcuts are only used to complete MLJModelInterface to make
# things more readable, they are not used elsewhere than in the `src/interface`
# folder to avoid confusion.

const MMI = MLJModelInterface
const FI  = FullInterface

include("data_utils.jl")
include("model_api.jl")
include("univariate_finite.jl")

# STUB FOR @load (extended by MLJModels)
macro load end
