module TestData

# using Revise
using Test
using MLJBase
using DataFrames
using TypedTables

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


## TABLE INDEXING

df = DataFrame(A)
df.z  = 1:10

@test selectcols(df, 4:6) == selectcols(df[4:6], :)
@test selectcols(df, [:x1, :z]) == selectcols(df[[:x1, :z]], :)
@test selectcols(df, :x2) == df.x2
@test selectcols(df, 2) == df.x2
@test selectrows(df, 4:6) == selectrows(df[4:6, :], :)
@test selectrows(df, 1) == selectrows(df[1:1, :], :)
s = schema(df)
@test (s.nrows, s.ncols) == size(df)
@test s.names == tuple(names(df)...)
@test s.eltypes == (Char, Char, Char, Char, Char, Int64)

tt = Table(df)
@test selectcols(tt, 4:6) == selectcols(Table(x4=tt.x4, x5=tt.x5, z=tt.z), :)
@test selectcols(tt, [:x1, :z]) == selectcols(Table(x1=tt.x1, z=tt.z), :)
@test selectcols(tt, :x2) == tt.x2
@test selectcols(tt, 2) == tt.x2
@test selectrows(tt, 4:6) == selectrows(tt[4:6], :)
s = schema(tt)
@test (s.nrows, s.ncols) == size(df)
@test s.names == tuple(names(df)...)
@test s.eltypes == (Char, Char, Char, Char, Char, Int64)


A = rand(Int, 3, 4)
@test selectcols(A, 3:4) == A[:,3:4]
@test selectrows(A, 2:3) == A[2:3,:]
s = schema(A)
@test (s.nrows, s.ncols) == size(A)
@test s.names == (:x1, :x2, :x3, :x4)
@test s.eltypes == (Int, Int, Int, Int)

v = rand(Int, 4)
@test selectrows(v, 2:3) == v[2:3]
s = schema(v)
@test s.nrows == 4
@test s.ncols == 1
@test first(s.eltypes) == Int
@test s.names == (:x,)

v = categorical(collect("asdfasdf"))
@test selectrows(v, 2:3) == v[2:3]
s = schema(v)
@test s.nrows == 8
@test s.ncols == 1
@test first(s.eltypes) == Char
@test s.names == (:x,)
df = DataFrame(v=v, w=v)
@test selectcols(df, :w) == v
tt = TypedTables.Table(df)
@test selectcols(tt, :w) == v


## MANIFESTING ARRAYS AS TABLES

A = hcat(v, v)
tab = MLJBase.table(A)
tab[1] == v
MLJBase.matrix(tab) == A

end # module

true
