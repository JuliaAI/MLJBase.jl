module TestScitypes

# using Revise
using Test
# using JuliaDB
using MLJBase
using CategoricalArrays
Unknown = MLJBase.Unknown

cv = categorical([:x, :y])
c = cv[1]
uv = categorical([:x, :y], ordered=true)
u = uv[1]

scitype((4, 4.5, c, u, "X")) ==
    Tuple{Count,Continuous,Multiclass{2},
          OrderedFactor{2},MLJBase.Unknown}


# uncomment 6 lines to restore testing of scitpye on NDSparse:
# nd = ndsparse((document=[6, 1, 1, 2, 3, 4],
#                word=[:house, :house, :sofa, :sofa, :chair, :house]),
#               (values=["big", "small", 17, 34, 4, "small"],))
# @test MLJBase.scitypes(nd) == (chair=Union{Missing,Count},
#                                house=Union{Missing,Unknown},
#                                sofa=Union{Missing,Count})

db = (x=rand(5), y=rand(Int, 5),
                    z=categorical(collect("asdfa")))
@test MLJBase.scitypes(db) == (x=Continuous,
                               y=Count,
                               z=Multiclass{4})

@test_throws ArgumentError MLJBase.scitypes(categorical([:x, :y]))

A = Any[2 4.5;
        4 4.5;
        6 4.5]

@test MLJBase.scitype_union(A) == Union{Count,Continuous}
@test scitype_union(randn(1000000)) == Continuous
@test scitype_union(1) == Count
@test scitype_union([1]) == Count
@test scitype_union(Any[1]) == Count
@test scitype_union([1, 2.0, "3"]) == Union{Continuous, Count, Unknown}

end
true
