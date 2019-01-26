module TestData

# using Revise
using Test
using MLJBase
using DataFrames
using TypedTables

import CategoricalArrays

A = broadcast(x->Char(65+mod(x,5)), rand(Int, 10,5))
X = CategoricalArrays.categorical(A)
Xsmall = X[2:5,3:4]

decoder = MLJBase.CategoricalDecoder(X, eltype=Float16)
@test inverse_transform(decoder, transform(decoder, Xsmall)) == Xsmall

decoder = MLJBase.CategoricalDecoder(X)
@test inverse_transform(decoder, transform(decoder, Xsmall)) == Xsmall

@test MLJBase.matrix(DataFrame(A)) == A

df = DataFrame(A)
df.z  =1:10

@test select(df, Cols, 4:6) == select(df[4:6], Cols, :)
@test select(df, Cols, [:x1, :z]) == select(df[[:x1, :z]], Cols, :)
@test select(df, Cols, :x2) == df.x2
@test select(df, Cols, 2) == df.x2
@test select(df, Rows, 4:6) == select(df[4:6, :], Rows, :)
s = select(df, Schema)
@test (s.nrows, s.ncols) == size(df)
@test s.names == tuple(names(df)...)
@test s.eltypes == (Char, Char, Char, Char, Char, Int64)

tt = Table(df)
@test select(tt, Cols, 4:6) == select(Table(x4=tt.x4, x5=tt.x5, z=tt.z), Cols, :)
@test select(tt, Cols, [:x1, :z]) == select(Table(x1=tt.x1, z=tt.z), Cols, :)
@test select(tt, Cols, :x2) == tt.x2
@test select(tt, Cols, 2) == tt.x2
@test select(tt, Rows, 4:6) == select(tt[4:6], Rows, :)
s = select(tt, Schema)
@test (s.nrows, s.ncols) == size(df)
@test s.names == tuple(names(df)...)
@test s.eltypes == (Char, Char, Char, Char, Char, Int64)


A = rand(Int, 3, 4)
@test select(A, Cols, 3:4) == A[:,3:4]
@test select(A, Rows, 2:3) == A[2:3,:]
s = select(A, Schema)
@test (s.nrows, s.ncols) == size(A)
@test s.names == (:x1, :x2, :x3, :x4)
@test s.eltypes == (Int, Int, Int, Int)

v = rand(Int, 4)
@test select(v, Rows, 2:3) == v[2:3]
s = select(v, Schema)
@test s.nrows == 4
@test s.ncols == 1
@test first(s.eltypes) == Int
@test s.names == (:x,)

v = categorical(collect("asdfasdf"))
@test select(v, Rows, 2:3) == v[2:3]
s = select(v, Schema)
@test s.nrows == 8
@test s.ncols == 1
@test first(s.eltypes) == Char
@test s.names == (:x,)
df = DataFrame(v=v, w=v)
@test select(df, Cols, :w) == v
tt = TypedTables.Table(df)
@test select(tt, Cols, :w) == v

A = hcat(v, v)
tab = MLJBase.table(A)
tab[1] == v
MLJBase.matrix(tab) == A

end # module

true
