module TestData

# using Revise
using Test
using MLJBase
using DataFrames
using TypedTables
using JuliaDB

import CategoricalArrays

A = broadcast(x->Char(65+mod(x,5)), rand(Int, 10, 5))
X = CategoricalArrays.categorical(A)
Xsmall = X[2:5,3:4]


## DECODER

# using specified output type:
decoder = MLJBase.CategoricalDecoder(X, Float16)
@test inverse_transform(decoder, transform(decoder, Xsmall)) == Xsmall

# using original type:
decoder = MLJBase.CategoricalDecoder(X)
@test inverse_transform(decoder, transform(decoder, Xsmall)) == Xsmall

v = CategoricalArrays.categorical(collect("asdfasdfasdf"))
decoder = MLJBase.CategoricalDecoder(v[1:2], Float16)
@test levels(decoder) == ['a', 'd', 'f', 's']
@test levels_seen(decoder) == ['a', 's']


## MATRIX

@test MLJBase.matrix(DataFrame(A)) == A


## ISTABLE

df = DataFrame(A)
tt = Table(df)
db = JuliaDB.table(tt)
nd = ndsparse((document=[1, 1, 2, 3, 4],
               word=[:house, :sofa, :sofa, :chair, :house]), (values=[23, 17, 34, 4, 5],))

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
@test MLJBase.select(nd, 1, :house) == 23
@test all(MLJBase.select(nd, :, :house) .=== [23, missing, missing, 5])
@test all(MLJBase.select(nd, [2,4], :house) .=== [missing, 5])
@test all(selectcols(nd, :house) .=== MLJBase.select(nd, :, :house))
@test nrows(nd) == 4
schema(nd)


## MANIFESTING ARRAYS AS TABLES

A = hcat(v, v)
tab = MLJBase.table(A)
tab[1] == v
MLJBase.matrix(tab) == A

end # module

true
