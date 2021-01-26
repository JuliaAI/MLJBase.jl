module Models

using MLJModelInterface
import MLJBase # needed for UnivariateFinite in ConstantClassifier

const MMI = MLJModelInterface

include("Constant.jl")
include("DecisionTree.jl")
include("NearestNeighbors.jl")
include("MultivariateStats.jl")
include("Transformers.jl")
include("foobarmodel.jl")
include("simple_composite_model.jl")

end
