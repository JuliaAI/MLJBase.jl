module TestDatasets

# using Revise
using Test
using MLJBase

task = load_boston()
load_ames()
load_reduced_ames()
load_iris()
load_crabs()
datanow()

end # module
true
