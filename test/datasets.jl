module TestDatasets

# using Revise
using Test
using MLJBase
using DataFrames


task = load_boston()
@test MLJBase.features(task) == [:Crim, :Zn, :Indus, :NOx, :Rm, :Age, :Dis,
                    :Rad, :Tax, :PTRatio, :Black, :LStat]
MLJBase.features(task)
load_ames()
load_iris()
datanow()

end # module
true
