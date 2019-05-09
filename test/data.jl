module TestData

# using Revise
using Test
using MLJBase
using DataFrames
using TypedTables
using StatsBase
# using JuliaDB
using SparseArrays
using CategoricalArrays

using Random
import Random.seed!
seed!(1234)

import MLJBase: decoder, int, raw, classes


# ## RECONSTRUCT

# C = categorical(rand(UInt8,20,20))[1:2,1:3]
# @test length(levels(C)) > 6
# @test reconstruct(broadcast(identity, C)) == C

# C = categorical(string.(rand(UInt8,20,20)))[1:2,1:3]
# @test length(levels(C)) > 6
# @test reconstruct(broadcast(identity, C)) == C


## DECODER

N = 10
mix = shuffle(0:N - 1)

Xraw = broadcast(x->mod(x,N), rand(Int, 20, 100))
Yraw = string.(Xraw)

X = categorical(Xraw)
x = X[1]
Y = categorical(Yraw)
y = Y[1]
V = broadcast(identity, X)
W = broadcast(identity, Y)
Xo = levels!(deepcopy(X), mix)
xo = Xo[1]
Yo = levels!(deepcopy(Y), string.(mix))
yo = Yo[1]
Vo = broadcast(identity, Xo)
Wo = broadcast(identity, Yo)

# classes:
@test raw.(classes(xo)) == xo.pool.levels
@test raw.(classes(yo)) == yo.pool.levels

# getting all possible elements from one:
@test raw.(X) == Xraw
@test raw.(Y) == Yraw
@test raw.(classes(xo)) == levels(Xo)
@test raw.(classes(yo)) == levels(Yo)

# broadcasted encoding:
@test int(X) == int(V)
@test int(Y) == int(W)

# encoding and decoding are inverses:
d = decoder(xo)
@test d(int(Vo)) == Vo
e = decoder(yo)
@test e(int(Wo)) == Wo
A = sample(xo.pool.order, 100)
@test int(d(A)) == A
@test int(e(A)) == A


## MATRIX

B = rand(UInt8, (4, 5))
@test MLJBase.matrix(DataFrame(B)) == B


## ISTABLE

A = broadcast(x->Char(65+mod(x,5)), rand(Int, 10, 5))
X = CategoricalArrays.categorical(A)

df = DataFrame(A)
tt = Table(df)
# uncomment 4 lines to restore JuliaDB testing:
# db = JuliaDB.table(tt)
# nd = ndsparse((document=[6, 1, 1, 2, 3, 4],
#                word=[:house, :house, :sofa, :sofa, :chair, :house]), (values=["big", "small", 17, 34, 4, "small"],))

@test !MLJBase.istable(A)
@test !MLJBase.istable([1,2,3])
@test MLJBase.istable(df)
@test MLJBase.istable(tt)
# uncomment 2 lines to restore JuliaDB testing:
# @test MLJBase.istable(db)
# @test MLJBase.isndsparse(nd)

## TABLE INDEXING

df.z  = 1:10
tt = Table(df)
# uncomment 1 line to restore JuliaDB testing:
# db = JuliaDB.table(tt)

@test selectcols(df, 4:6) == selectcols(df[4:6], :)
@test selectcols(df, [:x1, :z]) == selectcols(df[[:x1, :z]], :)
@test selectcols(df, :x2) == df.x2
@test selectcols(df, 2) == df.x2
@test selectrows(df, 4:6) == selectrows(df[4:6, :], :)
@test selectrows(df, 1) == selectrows(df[1:1, :], :)
@test MLJBase.select(df, 2, :x2) == df[2,:x2]
s = schema(df)
@test s.names == tuple(names(df)...)
@test s.types == (Char, Char, Char, Char, Char, Int64)
@test nrows(df) == size(df, 1)

@test selectcols(tt, 4:6) == selectcols(Table(x4=tt.x4, x5=tt.x5, z=tt.z), :)
@test selectcols(tt, [:x1, :z]) == selectcols(Table(x1=tt.x1, z=tt.z), :)
@test selectcols(tt, :x2) == tt.x2
@test selectcols(tt, 2) == tt.x2
@test selectrows(tt, 4:6) == selectrows(tt[4:6], :)
s = schema(tt)
@test s.names == tuple(names(df)...)
@test s.types == (Char, Char, Char, Char, Char, Int64)
@test nrows(tt) == length(tt.x1)
@test MLJBase.select(tt, 2, :x2) == tt.x2[2]

v = rand(Int, 4)
@test selectrows(v, 2:3) == v[2:3]
@test nrows(v) == 4

v = categorical(collect("asdfasdf"))
@test selectrows(v, 2:3) == v[2:3]
@test nrows(v) == 8

df = DataFrame(v=v, w=v)
@test selectcols(df, :w) == v
tt = TypedTables.Table(df)
# uncomment 1 line to restore JuliaDB testing:
# db = JuliaDB.table(tt)
@test selectcols(tt, :w) == v
# uncomment 1 line to restore JuliaDB testing:
# @test selectcols(db, :w) == v

# uncomment 9 lines to restore JuliaDB testing:
# @test MLJBase.select(nd, 2, :house) isa Missing
# @test MLJBase.select(nd, 1, :house) == "small"
# @test all(MLJBase.select(nd, :, :house) .=== ["small", missing, missing, "small", missing, "big"])
# @test all(MLJBase.select(nd, [2,4], :house) .=== [missing, "small"])
# @test all(selectcols(nd, :house) .=== MLJBase.select(nd, :, :house))
# @test nrows(nd) == 6
# s = schema(nd)
# @test s.names == (:chair, :house, :sofa)
# @test s.types == (Union{Missing,Int64}, Union{Missing,String}, Union{Missing,Int64})



## MANIFESTING ARRAYS AS TABLES

A = hcat(v, v)
tab = MLJBase.table(A)
tab[1] == v
MLJBase.matrix(tab) == A

# uncomment 3 lines to restore JuliaDB testing:
# sparsearray = sparse([6, 1, 1, 2, 3, 4], [2, 2, 3, 3, 1, 2],
#                      ["big", "small", 17, 34, 4, "small"])
# @test MLJBase.matrix(nd) == sparsearray

end # module

true
