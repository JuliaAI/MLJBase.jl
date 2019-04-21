module TestScitypes

# using Revise
using Test
using JuliaDB
using MLJBase
using CategoricalArrays
Unknown = MLJBase.Unknown

nd = ndsparse((document=[6, 1, 1, 2, 3, 4],
               word=[:house, :house, :sofa, :sofa, :chair, :house]), (values=["big", "small", 17, 34, 4, "small"],))
@test MLJBase.union_scitypes(nd) == Union{Count, Missing, MLJBase.Unknown}
@test MLJBase.column_scitypes_as_tuple(nd) == Tuple{Union{Missing,Count},Union{Missing,Unknown},Union{Missing,Count}}

db = JuliaDB.table((x=rand(5), y=rand(Int, 5), z=categorical(collect("asdfa"))))
@test MLJBase.union_scitypes(db) == Union{Continuous,Count,Multiclass{4}}           
@test MLJBase.column_scitypes_as_tuple(db) == Tuple{Continuous,Count,Multiclass{4}}

A = Any[2 4.5;
        4 4.5;
        6 4.5]

@test MLJBase.column_scitypes_as_tuple(A) == Tuple{Count,Continuous}

end
true
