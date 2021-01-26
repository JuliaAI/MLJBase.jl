module Models

using MLJModelInterface
import MLJBase # needed for UnivariateFinite in ConstantClassifier

include("Constant.jl")
include("DecisionTree.jl")
include("NearestNeighbors.jl")
include("MultivariateStats.jl")
include("Transformers.jl")
include("foobarmodel.jl")
include("simple_composite_model.jl")

end
