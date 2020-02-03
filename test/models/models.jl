module Models

using MLJModelInterface

import MLJBase # needed for UnivariateFinite in ConstantClassifier

const MMI = MLJModelInterface

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
