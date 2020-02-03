# If adding models from MLJModels for testing purposes, then do the
# following in the interface file (eg, DecisionTree.jl):

# - change `import ..DecisionTree` to `import DecisionTree`
# - remove wrapping as module

# load the models for testing:
module Models

using MLJModelInterface

include("Constant.jl")
# include("DecisionTree.jl")
# include("NearestNeighbors.jl")
# include("MultivariateStats.jl")
# include("Transformers.jl")
# include("foobarmodel.jl")
# include("simple_composite_model.jl")

end

# to load a model with @load:
import MLJBase.@load
macro load(name_ex, kw_exp...)
    esc(quote
        $name_ex()
        end)
end
