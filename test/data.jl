module TestData

# using Revise
using Test
using MLJBase
using DataFrames
using TypedTables
using JuliaDB

using SparseArrays
import CategoricalArrays

A = broadcast(x->Char(65+mod(x,5)), rand(Int, 10, 5))
X = CategoricalArrays.categorical(A)
Xsmall = X[2:5,3:4]


## DECODER

# using specified output type:
decoder = MLJBase.CategoricalDecoder(X, Float16)
@test inverse_transform(decoder, transform(decoder, Xsmall)) == Xsmall
@test minimum(unique(transform(decoder, X))) == 1.0

# special boolean case:
Xb = categorical([:y, :n, :n, :n, :y, :y])
decoder = MLJBase.CategoricalDecoder(Xb, Bool)
@test inverse_transform(decoder, transform(decoder, Xb[1:4])) == Xb[1:4]

decoder = MLJBase.CategoricalDecoder(X, Float16, true)
@test inverse_transform(decoder, transform(decoder, Xsmall)) == Xsmall
@test minimum(unique(transform(decoder, X))) == 0.0

# using original type:
decoder = MLJBase.CategoricalDecoder(X)
@test inverse_transform(decoder, transform(decoder, Xsmall)) == Xsmall

v = CategoricalArrays.categorical(collect("asdfasdfasdf"))
decoder = MLJBase.CategoricalDecoder(v[1:2], Float16)
@test levels(decoder) == ['a', 'd', 'f', 's']
@test levels_seen(decoder) == ['a', 's']

@test_throws Exception MLJBase.CategoricalDecoder(X, UInt64, true)



## MATRIX

@test MLJBase.matrix(DataFrame(A)) == A


## ISTABLE

df = DataFrame(A)
tt = Table(df)
db = JuliaDB.table(tt)
nd = ndsparse((document=[6, 1, 1, 2, 3, 4],
               word=[:house, :house, :sofa, :sofa, :chair, :house]), (values=["big", "small", 17, 34, 4, "small"],))

@test !MLJBase.istable(A)
@test !MLJBase.istable([1,2,3])
@test MLJBase.istable(df)
@test MLJBase.istable(tt)
@test MLJBase.istable(db)
@test MLJBase.isndsparse(nd)

## TABLE INDEXING

df.z  = 1:10
tt = Table(df)
db = JuliaDB.table(tt)

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
db = JuliaDB.table(tt)
@test selectcols(tt, :w) == v
@test selectcols(db, :w) == v

@test MLJBase.select(nd, 2, :house) isa Missing
@test MLJBase.select(nd, 1, :house) == "small"
@test all(MLJBase.select(nd, :, :house) .=== ["small", missing, missing, "small", missing, "big"])
@test all(MLJBase.select(nd, [2,4], :house) .=== [missing, "small"])
@test all(selectcols(nd, :house) .=== MLJBase.select(nd, :, :house))
@test nrows(nd) == 6
s = schema(nd)
@test s.names == (:chair, :house, :sofa)
@test s.types == (Union{Missing,Int64}, Union{Missing,String}, Union{Missing,Int64})



## MANIFESTING ARRAYS AS TABLES

A = hcat(v, v)
tab = MLJBase.table(A)
tab[1] == v
MLJBase.matrix(tab) == A

sparsearray = sparse([6, 1, 1, 2, 3, 4], [2, 2, 3, 3, 1, 2], ["big", "small", 17, 34, 4, "small"])
@test MLJBase.matrix(nd) == sparsearray
end # module

true
