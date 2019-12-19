# If adding models from MLJModels for testing purposes, then do the
# following in the interface file (eg, DecisionTree.jl):

# - change `import ..DecisionTree` to `import DecisionTree`
# - remove wrapping as module

# load the models for testing:
module Models

using MLJBase

include("models/Constant.jl")
include("models/DecisionTree.jl")
include("models/NearestNeighbors.jl")
include("models/MultivariateStats.jl")
include("models/Transformers.jl")
include("models/foobarmodel.jl")
include("models/simple_composite_model.jl")

end

# to load a model with @load:
import MLJBase.@load
macro load(name_ex, kw_exp...)
    esc(quote
        $name_ex()
        end)
end
