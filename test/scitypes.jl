module TestScitypes

# using Revise
using Test
using JuliaDB
using MLJBase
using CategoricalArrays
Unknown = MLJBase.Unknown

cv = categorical([:x, :y])
c = cv[1]
uv = categorical([:x, :y], ordered=true)
u = uv[1]

scitype((4, 4.5, c, u, "X")) ==
    Tuple{Count,Continuous,Multiclass{2},
          FiniteOrderedFactor{2},MLJBase.Unknown}

nd = ndsparse((document=[6, 1, 1, 2, 3, 4],
               word=[:house, :house, :sofa, :sofa, :chair, :house]),
              (values=["big", "small", 17, 34, 4, "small"],))
@test MLJBase.scitypes(nd) == (chair=Union{Missing,Count},
                               house=Union{Missing,Unknown},
                               sofa=Union{Missing,Count})

db = JuliaDB.table((x=rand(5), y=rand(Int, 5),
                    z=categorical(collect("asdfa"))))
@test MLJBase.scitypes(db) == (x=Continuous,
                               y=Count,
                               z=Multiclass{4})

A = Any[2 4.5;
        4 4.5;
        6 4.5]

@test MLJBase.scitype_union(A) == Union{Count,Continuous}

end
true
